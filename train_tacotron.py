import torch
from torch import optim
import torch.nn.functional as F
from utils import hparams as hp
from utils.display import *
from utils.dataset import get_tts_datasets
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron, Tacotron_ss
import argparse
from utils import data_parallel_workaround, set_global_seeds
import os
from pathlib import Path
import time
import numpy as np
import sys
from utils.checkpoints import save_checkpoint, restore_checkpoint


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_af_online', action='store_true', help='Force the model to create EAL / AF features')
    parser.add_argument('--force_attn', '-a', action='store_true', help='Force the model to create attn_ref')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    hp.fix_compatibility()
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    if hasattr(hp, 'random_seed'):
        set_global_seeds(hp.random_seed)

    force_train = args.force_train
    force_gta = args.force_gta
    force_attn = args.force_attn
    force_af_online = args.force_af_online

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitialising Tacotron Model...\n')
    Taco = Tacotron_ss if hp.mode=='scheduled_sampling' else Tacotron
    model = Taco(embed_dims=hp.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hp.tts_encoder_dims,
                     decoder_dims=hp.tts_decoder_dims,
                     n_mels=hp.num_mels,
                     fft_bins=hp.num_mels,
                     postnet_dims=hp.tts_postnet_dims,
                     encoder_K=hp.tts_encoder_K,
                     lstm_dims=hp.tts_lstm_dims,
                     postnet_K=hp.tts_postnet_K,
                     num_highways=hp.tts_num_highways,
                     dropout=hp.tts_dropout,
                     stop_threshold=hp.tts_stop_threshold,
                     mode=hp.mode).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True,  init_weights_path=hp.tts_init_weights_path)

    # import pdb; pdb.set_trace()

    if hp.mode!='attention_forcing_online':
        model_tf = None
    else:
        model_tf = Tacotron(embed_dims=hp.tts_embed_dims,
                         num_chars=len(symbols),
                         encoder_dims=hp.tts_encoder_dims,
                         decoder_dims=hp.tts_decoder_dims,
                         n_mels=hp.num_mels,
                         fft_bins=hp.num_mels,
                         postnet_dims=hp.tts_postnet_dims,
                         encoder_K=hp.tts_encoder_K,
                         lstm_dims=hp.tts_lstm_dims,
                         postnet_K=hp.tts_postnet_K,
                         num_highways=hp.tts_num_highways,
                         dropout=hp.tts_dropout,
                         stop_threshold=hp.tts_stop_threshold,
                         mode='teacher_forcing').to(device)
        model_tf.load(hp.model_tf_path)

        # pdb.set_trace()


    if not (force_gta or force_attn or force_af_online):
        for i, session in enumerate(hp.tts_schedule):
            current_step = model.get_step()

            r, lr, max_step, batch_size = session

            training_steps = max_step - current_step

            # Do we need to change to the next session?
            if current_step >= max_step:
                # Are there no further sessions than the current one?
                if i == len(hp.tts_schedule)-1:
                    # There are no more sessions. Check if we force training.
                    if force_train:
                        # Don't finish the loop - train forever
                        training_steps = 999_999_999
                    else:
                        # We have completed training. Breaking is same as continue
                        break
                else:
                    # There is a following session, go to it
                    continue

            model.r = r
            # if model_tf is not None: model_tf.r = r

            simple_table([(f'Steps with r={r}', str(training_steps//1000) + 'k Steps'),
                            ('Batch Size', batch_size),
                            ('Learning Rate', lr),
                            ('Outputs/Step (r)', model.r)])

            train_set, attn_example = get_tts_datasets(paths.data, batch_size, r, data_split=hp.tts_data_split)
            tts_train_loop(paths, model, optimizer, train_set, lr, training_steps, attn_example, hp=hp, model_tf=model_tf)

        print('Training Complete.')
        print('To continue training increase tts_total_steps in hparams.py or use --force_train\n')

    train_set, attn_example = get_tts_datasets(paths.data, 8, model.r)
    if force_gta:
        print(f'Creating Ground Truth Aligned Dataset at {paths.gta_model}...\n')
        create_gta_features(model, train_set, paths.gta_model)
    elif force_attn:
        print(f'Creating Reference Attention at {paths.attn_model}...\n')
        create_attn_ref(model, train_set, paths.attn_model)
    elif force_af_online:
        print(f'Creating AF Dataset at {paths.gta_model}...\n')
        create_af_online_features(model, model_tf, train_set, paths.gta_model)

    print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')


def tts_train_loop(paths: Paths, model: Tacotron, optimizer, train_set, lr, train_steps, attn_example, hp=None, model_tf=None):
    if hp.mode=='teacher_forcing':
        tts_train_loop_tf(paths, model, optimizer, train_set, lr, train_steps, attn_example)
    elif hp.mode=='attention_forcing_online':
        tts_train_loop_af_online(paths, model, model_tf, optimizer, train_set, lr, train_steps, attn_example, hp=hp)
    elif hp.mode=='attention_forcing_offline':
        tts_train_loop_af_offline(paths, model, optimizer, train_set, lr, train_steps, attn_example, hp=hp)
    elif hp.mode=='scheduled_sampling':
        tts_train_loop_ss(paths, model, optimizer, train_set, lr, train_steps, attn_example, hp=hp)
    else:
        raise NotImplementedError(f'hp.mode={hp.mode} is not yet implemented')


def tts_train_loop_tf(paths: Paths, model: Tacotron, optimizer, train_set, lr, train_steps, attn_example):
    # import pdb; pdb.set_trace()

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    # print(total_iters * hp.tts_batch_size)
    # pdb.set_trace()

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0

        # accumulte grad
        optimizer.zero_grad()

        # Perform 1 epoch
        for i, (x, m, ids, _) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)
            # pdb.set_trace()

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                m1_hat, m2_hat, attention = model(x, m)

            # print(x.size())
            # print(m.size())
            # print(m1_hat.size(), m2_hat.size())
            # print(attention.size(), attention.size(1)*model.r)
            # pdb.set_trace()

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)

            loss = m1_loss + m2_loss

            # optimizer.zero_grad()
            # loss.backward()
            # if hp.tts_clip_grad_norm is not None:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
            #     if np.isnan(grad_norm):
            #         print('grad_norm was NaN!')
            # optimizer.step()

            # accumulte grad
            (loss / hp.tts_batch_acu).backward()
            if (i+1)%hp.tts_batch_acu == 0:
                if hp.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                    if np.isnan(grad_norm):
                        print('grad_norm was NaN!')
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                save_checkpoint('tts', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attention[idx][:, :160]), paths.tts_attention/f'{step}')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        print(' ')


def tts_train_loop_ss(paths: Paths, model: Tacotron, optimizer, train_set, lr, train_steps, attn_example, hp=None):
    # import pdb; pdb.set_trace()

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    # print(total_iters * hp.tts_batch_size)
    # pdb.set_trace()

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0

        # accumulte grad
        optimizer.zero_grad()

        # Perform 1 epoch
        for i, (x, m, ids, _) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)

            # compute p_tf
            p_tf = get_p_tf(hp.tts_ss_dct, model.get_step())

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m, False, attn_ref, p_tf)
            else:
                m1_hat, m2_hat, attention = model(x, m, p_tf=p_tf)

            # print(x.size())
            # print(m.size())
            # print(m1_hat.size(), m2_hat.size())
            # print(attention.size(), attention.size(1)*model.r)
            # pdb.set_trace()

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)

            loss = m1_loss + m2_loss

            # accumulte grad
            (loss / hp.tts_batch_acu).backward()
            if (i+1)%hp.tts_batch_acu == 0:
                if hp.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                    if np.isnan(grad_norm):
                        print('grad_norm was NaN!')
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                save_checkpoint('tts', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attention[idx][:, :160]), paths.tts_attention/f'{step}')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        print(' ')


def tts_train_loop_af_online(paths: Paths, model: Tacotron, model_tf: Tacotron, optimizer, train_set, lr, train_steps, attn_example, hp=None):
    # setattr(model, 'mode', 'attention_forcing')
    # setattr(model, 'mode', 'teacher_forcing')
    # import pdb; pdb.set_trace()

    def smooth(d, eps = float(1e-10)):
        u = 1.0 / float(d.size()[2])
        return eps * u + (1-eps) * d

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss_out, running_loss_attn = 0, 0

        optimizer.zero_grad()

        # Perform 1 epoch
        for i, (x, m, ids, _) in enumerate(train_set, 1):
            # print(i)
            # import pdb; pdb.set_trace()

            x, m = x.to(device), m.to(device)
            # pdb.set_trace()

            # print(model.r, model_tf.r)
            # import pdb; pdb.set_trace()

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                with torch.no_grad(): _, _, attn_ref = data_parallel_workaround(model_tf, x, m)
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m, True, attn_ref)
            else:
                with torch.no_grad(): _, _, attn_ref = model_tf(x, m, generate_gta=False)
                # pdb.set_trace()

                # setattr(model, 'mode', 'teacher_forcing')
                # with torch.no_grad(): _, _, attn_ref = model(x, m)

                # setattr(model, 'mode', 'attention_forcing_online')
                m1_hat, m2_hat, attention = model(x, m, generate_gta=False, attn_ref=attn_ref)
                # m1_hat, m2_hat, attention = model(x, m, generate_gta=False, attn_ref=None)
                # pdb.set_trace()

            # print(x.size())
            # print(m.size())
            # print(m1_hat.size(), m2_hat.size())
            # print(attention.size(), attention.size(1)*model.r)
            # print(attn_ref.size())
            # pdb.set_trace()

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)
            attn_loss = F.kl_div(torch.log(smooth(attention)), smooth(attn_ref), reduction='none') # 'batchmean'
            attn_loss = attn_loss.sum(2).mean()
            # attn_loss = F.l1_loss(smooth(attention), smooth(attn_ref))

            loss_out = m1_loss + m2_loss
            loss_attn = attn_loss * hp.attn_loss_coeff
            loss = loss_out + loss_attn

            # if i%100==0:
            #     save_attention(np_now(attn_ref[0][:, :160]), paths.tts_attention/f'asup_{step}_tf')
            #     save_attention(np_now(attention[0][:, :160]), paths.tts_attention/f'asup_{step}_af')

            #     model_tf.r = 2
            #     with torch.no_grad(): _, _, attn_ref = model_tf(x, m)
            #     save_attention(np_now(attn_ref[0][:, :160]), paths.tts_attention/f'asup_{step}_tf_r2')
            #     model_tf.r = model.r
            #     pdb.set_trace()

            # optimizer.zero_grad()
            # loss.backward()
            # if hp.tts_clip_grad_norm is not None:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
            #     if np.isnan(grad_norm):
            #         print('grad_norm was NaN!')
            # optimizer.step()

            # accumulte grad
            (loss / hp.tts_batch_acu).backward()
            if (i+1)%hp.tts_batch_acu == 0:
                if hp.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                    if np.isnan(grad_norm):
                        print('grad_norm was NaN!')
                optimizer.step()
                optimizer.zero_grad()

            running_loss_out += loss_out.item()
            avg_loss_out = running_loss_out / i
            running_loss_attn += loss_attn.item()
            avg_loss_attn = running_loss_attn / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                save_checkpoint('tts', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attn_ref[idx][:, :160]), paths.tts_attention/f'{step}_tf')
                save_attention(np_now(attention[idx][:, :160]), paths.tts_attention/f'{step}_af')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss_out: {avg_loss_out:#.4}; Loss_attn: {avg_loss_attn:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        print(' ')


def tts_train_loop_af_offline(paths: Paths, model: Tacotron, optimizer, train_set, lr, train_steps, attn_example, hp=None):
    # setattr(model, 'mode', 'attention_forcing')
    # import pdb
    if hp.tts_batch_acu>1:
        raise NotImplementedError(f'gradient accumulation is not yet implemented')

    def smooth(d, eps = float(1e-10)):
        u = 1.0 / float(d.size()[2])
        return eps * u + (1-eps) * d

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss_out, running_loss_attn = 0, 0

        # Perform 1 epoch
        for i, (x, m, ids, _, attn_ref) in enumerate(train_set, 1):

            # print(x.size())
            # print(m.size())
            # print(attn_ref.size())
            # # print(m1_hat.size(), m2_hat.size())
            # # print(attention.size(), attention.size(1)*model.r)
            # pdb.set_trace()

            x, m, attn_ref = x.to(device), m.to(device), attn_ref.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m, False, attn_ref)
            else:
                m1_hat, m2_hat, attention = model(x, m, generate_gta=False, attn_ref=attn_ref)

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)
            # attn_loss = F.kl_div(torch.log(smooth(attention)), smooth(attn_ref), reduction='mean') # 'batchmean'
            attn_loss = F.l1_loss(smooth(attention), smooth(attn_ref))

            loss_out = m1_loss + m2_loss
            loss_attn = attn_loss * hp.attn_loss_coeff
            loss = loss_out + loss_attn

            optimizer.zero_grad()
            loss.backward()
            if hp.tts_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')

            optimizer.step()

            running_loss_out += loss_out.item()
            avg_loss_out = running_loss_out / i
            running_loss_attn += loss_attn.item()
            avg_loss_attn = running_loss_attn / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                save_checkpoint('tts', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attn_ref[idx][:, :160]), paths.tts_attention/f'{step}_tf')
                save_attention(np_now(attention[idx][:, :160]), paths.tts_attention/f'{step}_af')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss_out: {avg_loss_out:#.4}; Output_attn: {avg_loss_attn:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        print(' ')


def create_gta_features(model: Tacotron, train_set, save_path: Path):
    save_path.mkdir(parents=False, exist_ok=True)
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens) in enumerate(train_set, 1):

        x, mels = x.to(device), mels.to(device)

        with torch.no_grad(): _, gta, _ = model(x, mels) # might want to set generate_gta=True

        gta = gta.cpu().numpy()

        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            np.save(save_path/f'{item_id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)

def create_af_online_features(model: Tacotron, model_tf: Tacotron, train_set, save_path: Path):
    save_path.mkdir(parents=False, exist_ok=True)
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens) in enumerate(train_set, 1):

        x, mels = x.to(device), mels.to(device)

        with torch.no_grad():
            _, _, attn_ref = model_tf(x, mels, generate_gta=True)
            _, gta, _ = model(x, mels, generate_gta=True, attn_ref=attn_ref)

        gta = gta.cpu().numpy()

        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            np.save(save_path/f'{item_id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)

def create_attn_ref(model: Tacotron, train_set, save_path: Path):
    # import pdb; pdb.set_trace()
    save_path.mkdir(parents=False, exist_ok=True)
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens) in enumerate(train_set, 1):

        x, mels = x.to(device), mels.to(device)

        with torch.no_grad(): _, m2_hat, attn_ref = model(x, mels)

        # print(x.size())
        # print(mels.size())
        # print(m2_hat.size())
        # print(attn_ref.size())
        # print(mel_lens)
        # pdb.set_trace()

        attn_ref = attn_ref.cpu().numpy()

        for j, item_id in enumerate(ids):
            # attn_ref_tmp = attn_ref[j][:mel_lens[j]//model.r, :]
            attn_ref_tmp = attn_ref[j][:, :]
            np.save(save_path/f'{item_id}.npy', attn_ref_tmp, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


def get_p_tf(ss_dct, step):
    progress = min(1.0, 1.0 * step / ss_dct['steps'])
    ss_max = ss_dct['max_p_fr'] # 0.2 0.1
    # linear schedule
    if ss_dct['type']=='linear':
        p_tf = 1.0 - ss_max * progress
    # Inverse sigmoid decay
    elif ss_dct['type']=='sigmoid':
        ss_k = 10 # 5
        p_tf = 1.0 - ss_max + ss_max * (ss_k / (ss_k + np.exp(progress * 100 / ss_k)))
    return p_tf


if __name__ == "__main__":
    main()
