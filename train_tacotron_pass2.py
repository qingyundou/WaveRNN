import torch
from torch import optim
import torch.nn.functional as F
from utils import hparams as hp
from utils.display import *
from utils.dataset import get_tts_datasets
from utils.text.symbols import symbols
from utils.paths import Paths, Paths_multipass
from models.tacotron import Tacotron, Tacotron_pass2
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
    parser.add_argument('--force_attn', '-a', action='store_true', help='Force the model to create attn_ref')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    # paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    paths = Paths_multipass(hp.data_path, hp.voc_model_id, hp.tts_model_id, 'pass1')
    paths_pass2 = Paths_multipass(hp.data_path, hp.voc_model_id, hp.tts_model_id, 'pass2')

    if hasattr(hp, 'random_seed'):
        set_global_seeds(hp.random_seed)

    force_train = args.force_train
    force_gta = args.force_gta
    force_attn = args.force_attn

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model & Tacotron_pass2 Model
    print('\nInitialising Tacotron Model...\n')
    model = Tacotron(embed_dims=hp.tts_embed_dims,
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
                     mode=hp.tts_mode_train_pass1).to(device)

    model_pass2 = Tacotron_pass2(embed_dims=hp.tts_embed_dims,
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
                     mode=hp.tts_mode_train_pass2,
                     encoder_reduction_factor=hp.tts_encoder_reduction_factor).to(device)

    # tmp = model.named_parameters()
    # names_p1 = [x[0] for x in tmp]
    # tmp = model_pass2.named_parameters()
    # names_p2 = [x[0] for x in tmp]
    # print(len(names_p1), len(names_p2))
    # for n in names_p2:
    #     if n not in names_p1:
    #         print(n)
    
    # cnt = 0
    # for name, param in model_pass2.named_parameters():
    #     # print(parameter)
    #     if name not in names_p1:
    #         print('before', name, param.data.size())
    #         print(param.data[0,:10])
    #         break
    #     # cnt += 1
    #     # if cnt>3: break
    # import pdb; pdb.set_trace()

    # optimizer = optim.Adam(list(model.parameters()) + list(model_pass2.parameters()))
    optimizer = optim.Adam(model.parameters())
    optimizer_pass2 = optim.Adam(model_pass2.parameters())


    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True, init_weights_path=hp.tts_init_weights_path)
    restore_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, create_if_missing=True, init_weights_path=hp.tts_init_weights_path_pass2)

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


    if not (force_gta or force_attn):
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

            train_set, attn_example = get_tts_datasets(paths.data, batch_size, r)
            tts_train_loop(paths, paths_pass2, model, model_pass2, optimizer, optimizer_pass2, train_set, lr, training_steps, attn_example, hp=hp, model_tf=model_tf)

        print('Training Complete.')
        print('To continue training increase tts_total_steps in hparams.py or use --force_train\n')

    train_set, attn_example = get_tts_datasets(paths.data, 8, model.r)
    if force_gta:
        print(f'Creating Ground Truth Aligned Dataset at {paths.gta_model}...\n')
        create_gta_features(model, train_set, paths.gta_model)
    elif force_attn:
        print(f'Creating Reference Attention at {paths.attn_model}...\n')
        create_attn_ref(model, train_set, paths.attn_model)

    print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')


def tts_train_loop(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, optimizer, optimizer_pass2, 
    train_set, lr, train_steps, attn_example, hp=None, model_tf=None):
    if hp.mode=='teacher_forcing':
        tts_train_loop_tf(paths, paths_pass2, model, model_pass2, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example)
    # elif hp.mode=='attention_forcing_online':
    #     tts_train_loop_af_online(paths, model, model_tf, optimizer, train_set, lr, train_steps, attn_example, hp=hp)
    # elif hp.mode=='attention_forcing_offline':
    #     tts_train_loop_af_offline(paths, model, optimizer, train_set, lr, train_steps, attn_example, hp=hp)
    else:
        raise NotImplementedError(f'hp.mode={hp.mode} is not yet implemented')


def tts_train_loop_tf(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example):
    # import pdb; pdb.set_trace()
    # model.mode = hp.mode_pass1 # 'free_running', asup

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0

        # optimizer.zero_grad()
        optimizer_pass2.zero_grad()

        # Perform 1 epoch
        for i, (x, m, ids, _) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)
            # pdb.set_trace()

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                with torch.no_grad(): _, m2_hat, attention = model(x, m)

                if hp.tts_pass2_input_train=='y1':
                    x = x * 0
                elif hp.tts_pass2_input_train in ['x_y1', 'x_y1_xNy1']:
                    if hp.tts_pass2_input_train=='x_y1' or np.random.randint(2):
                        mask = np.random.randint(2)
                        x = x * mask
                        m2_hat = m2_hat * (1-mask)
                elif hp.tts_pass2_input_train=='xNy1':
                	pass
                    # import pdb; pdb.set_trace()

                # m1_hat_p2, m2_hat_p2, attention_p2, attention_vc = model_pass2(x * 0, m, m)
                m1_hat_p2, m2_hat_p2, attention_p2, attention_vc = model_pass2(x, m, m2_hat)

            # print(x.size())
            # print(m.size())
            # print(m2_hat.size())
            # print(m1_hat_p2.size(), m2_hat_p2.size())
            # print(attention_p2.size(), attention_p2.size(1)*model.r)
            # print(attention_vc.size(), attention_vc.size(1)*model.r)
            # pdb.set_trace()
            # import pdb; pdb.set_trace()

            m1_loss = F.l1_loss(m1_hat_p2, m)
            m2_loss = F.l1_loss(m2_hat_p2, m)
            loss = (m1_loss + m2_loss) / hp.tts_batch_acu

            # optimizer.zero_grad()
            # loss.backward()
            # if hp.tts_clip_grad_norm is not None:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
            #     if np.isnan(grad_norm):
            #         print('grad_norm was NaN!')
            # optimizer.step()

            # optimizer_pass2.zero_grad()
            loss.backward()
            if (i+1)%hp.tts_batch_acu == 0:
                # grad = model_pass2.encoder.embedding.weight.grad
                # print(grad.size())
                # print(grad[0, :3])
                # pdb.set_trace()
                # clip grad only once before updating the params with step()
                if hp.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model_pass2.parameters(), hp.tts_clip_grad_norm)
                    if np.isnan(grad_norm):
                        print('grad_norm was NaN!')
                optimizer_pass2.step()
                optimizer_pass2.zero_grad()

            running_loss += loss.item() * hp.tts_batch_acu
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            # step = model.get_step()
            step = model_pass2.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                # save_checkpoint('tts', paths, model, optimizer, name=ckpt_name, is_silent=True)
                save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, name=ckpt_name, is_silent=True)

            # save_attention(np_now(attention_vc[0][:, :]), paths.tts_attention/f'{step}_speech')
            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attention[idx][:, :160]), paths.tts_attention/f'{step}_text_p1')
                save_attention(np_now(attention_p2[idx][:, :160]), paths.tts_attention/f'{step}_text_p2')
                save_attention(np_now(attention_vc[idx][:, :]), paths.tts_attention/f'{step}_speech')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}_p1', 600)
                save_spectrogram(np_now(m2_hat_p2[idx]), paths.tts_mel_plot/f'{step}_p2', 600)
            for idx, tmp in enumerate(ids):
                # import pdb; pdb.set_trace()
                if tmp in ['LJ035-0011', 'LJ016-0320']: # selected egs
                    save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}_{tmp}_p1')
                    save_spectrogram(np_now(m2_hat_p2[idx]), paths.tts_mel_plot/f'{step}_{tmp}_p2')

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        # save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        # model.log(paths.tts_log, msg)
        save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, is_silent=True)
        model_pass2.log(paths_pass2.tts_log, msg)
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
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m, False, attn_ref)
            else:
                with torch.no_grad(): _, _, attn_ref = model_tf(x, m)
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

        with torch.no_grad(): _, gta, _ = model(x, mels)

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


if __name__ == "__main__":
    main()
