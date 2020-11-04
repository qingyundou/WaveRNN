import torch
from torch import optim
import torch.nn.functional as F
from utils import hparams as hp
from utils.display import *
from utils.dataset import get_tts_datasets
from utils.text.symbols import symbols
from utils.paths import Paths, Paths_multipass
from models.tacotron import Tacotron, Tacotron_pass2, Tacotron_pass1
import argparse
from utils import data_parallel_workaround, set_global_seeds, smooth
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
            _, _, _, batch_size, *extension = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model & Tacotron_pass2 Model
    print('\nInitialising Tacotron Model...\n')
    Taco = Tacotron_pass1 if 's1' in hp.tts_pass2_input_train else Tacotron
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
                     encoder_reduction_factor=hp.tts_encoder_reduction_factor,
                     encoder_reduction_factor_s=hp.tts_encoder_reduction_factor_s,
                     pass2_input=hp.tts_pass2_input_train).to(device)

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

            r, lr, max_step, batch_size, *extension = session
            if extension: hp.tts_extension_dct['input_prob_lst'] = extension[0] if extension else [0, 0, 1]

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
            model_pass2.r = r
            if model_tf is not None: model_tf.r = r

            simple_table([(f'Steps with r={r}', str(training_steps//1000) + 'k Steps'),
                            ('Batch Size', batch_size),
                            ('Learning Rate', lr),
                            ('Outputs/Step (r)', model.r),
                            ('p2_input_prob_lst', hp.tts_extension_dct['input_prob_lst'])])

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


def prepare_pass2_input(x, y_p1, s_p1, input_prob_lst):
    # unpack the flexible lst, if it is not empty
    s_p1 = s_p1[0] if s_p1 else None

    if sum(input_prob_lst)!=1: print(f'qd212 warning: input_prob_lst {input_prob_lst} doesnt sum to 1')
    p_x, p_y, p_both = [float(p) for p in input_prob_lst]

    if np.random.uniform(high=1.0)<p_both:
        pass
    elif np.random.uniform(high=p_both)<p_x:
        x = x * 0
    else:
        y_p1 = y_p1 * 0
        if s_p1 is not None: s_p1 = s_p1 * 0
    return x, y_p1, s_p1


def tts_train_loop(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, optimizer, optimizer_pass2, 
    train_set, lr, train_steps, attn_example, hp=None, model_tf=None):
    if hp.mode=='teacher_forcing':
        tts_train_loop_tf(paths, paths_pass2, model, model_pass2, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example)
    elif hp.mode=='attention_forcing_online':
        tts_train_loop_af_online(paths, paths_pass2, model, model_pass2, model_tf, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example, hp=hp)
    elif hp.mode=='attention_forcing_offline':
        tts_train_loop_af_offline(paths, paths_pass2, model, model_pass2, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example, hp=hp)
    else:
        raise NotImplementedError(f'hp.mode={hp.mode} is not yet implemented')


def tts_train_loop_tf(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example):
    # import pdb; pdb.set_trace()

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr
    for g in optimizer_pass2.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    passModelOptimizer_lst = []
    if hp.tts_updateP1: passModelOptimizer_lst.append((paths, model, optimizer))
    if hp.tts_updateP2: passModelOptimizer_lst.append((paths_pass2, model_pass2, optimizer_pass2))

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0

        optimizer.zero_grad()
        optimizer_pass2.zero_grad()

        # Perform 1 epoch
        for i, (x, m, ids, _) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)
            # pdb.set_trace()

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                # pass1
                if hp.tts_updateP1:
                    m1_hat, m2_hat, attention, *s_p1 = model(x, m)
                else:
                    with torch.no_grad(): _, m2_hat, attention, *s_p1 = model(x, m)

                # mask
                x, m2_hat, s_p1 = prepare_pass2_input(x, m2_hat, s_p1, hp.tts_extension_dct['input_prob_lst'])
                # import pdb; pdb.set_trace()

                # pass 2
                m1_hat_p2, m2_hat_p2, attention_p2, attention_vc = model_pass2(x, m, m2_hat, s_p1=s_p1)

            # print(x.size())
            # print(m.size())
            # print(m2_hat.size())
            # print(m1_hat_p2.size(), m2_hat_p2.size())
            # print(attention_p2.size(), attention_p2.size(1)*model.r)
            # print(attention_vc.size(), attention_vc.size(1)*model.r)
            # pdb.set_trace()
            # import pdb; pdb.set_trace()

            # m1_loss, m2_loss = F.l1_loss(m1_hat_p2, m), F.l1_loss(m2_hat_p2, m)
            # loss = (m1_loss + m2_loss) / hp.tts_batch_acu

            # if hp.tts_mode_train_pass1=='teacher_forcing':
            loss = (F.l1_loss(m1_hat_p2, m) + F.l1_loss(m2_hat_p2, m) + F.l1_loss(m1_hat, m) + F.l1_loss(m2_hat, m)) / hp.tts_batch_acu

            loss.backward()

            # w = model.encoder.embedding.weight.data
            # print(w.size(), w[0, :3])
            # w = model_pass2.encoder.embedding.weight.data
            # print(w.size(), w[0, :3])
            # pdb.set_trace()

            # grad = model.encoder.embedding.weight.grad
            # print(grad.size(), grad[0, :3])
            # grad = model_pass2.encoder.embedding.weight.grad
            # print(grad.size(), grad[0, :3])
            # pdb.set_trace()

            if (i+1)%hp.tts_batch_acu == 0:
                for _paths, _model, _optimizer in passModelOptimizer_lst:
                    # clip grad only once before updating the params with step()
                    if hp.tts_clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(), hp.tts_clip_grad_norm)
                        if np.isnan(grad_norm):
                            print('grad_norm was NaN!')
                    _optimizer.step()
                    _optimizer.zero_grad()
                #     print('wooooooooooooow')
                # w = model.encoder.embedding.weight.data
                # print(w.size(), w[0, :3])
                # w = model_pass2.encoder.embedding.weight.data
                # print(w.size(), w[0, :3])
                # pdb.set_trace()

            running_loss += loss.item() * hp.tts_batch_acu
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            # step = model.get_step()
            step = model_pass2.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                for _paths, _model, _optimizer in passModelOptimizer_lst:
                    save_checkpoint('tts', _paths, _model, _optimizer, name=ckpt_name, is_silent=True)
                # save_checkpoint('tts', paths, model, optimizer, name=ckpt_name, is_silent=True)
                # save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, name=ckpt_name, is_silent=True)

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
        for _paths, _model, _optimizer in passModelOptimizer_lst:
            save_checkpoint('tts', _paths, _model, _optimizer, is_silent=True)
            _model.log(_paths.tts_log, msg)
        # save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        # model.log(paths.tts_log, msg)
        # save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, is_silent=True)
        # model_pass2.log(paths_pass2.tts_log, msg)
        print(' ')


def tts_train_loop_af_online(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, model_tf: Tacotron, 
    optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example, hp=None):
    # import pdb; pdb.set_trace()

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr
    for g in optimizer_pass2.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    passModelOptimizer_lst = []
    if hp.tts_updateP1: passModelOptimizer_lst.append((paths, model, optimizer))
    if hp.tts_updateP2: passModelOptimizer_lst.append((paths_pass2, model_pass2, optimizer_pass2))

    for e in range(1, epochs+1):

        start = time.time()
        running_loss_out, running_loss_attn = 0, 0

        optimizer.zero_grad()
        optimizer_pass2.zero_grad()

        # Perform 1 epoch
        for i, (x, m, ids, _) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)
            # pdb.set_trace()

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                # reference attention
                with torch.no_grad(): _, _, attn_ref = model_tf(x, m)

                # pass1
                if hp.tts_updateP1:
                    m1_hat, m2_hat, attention, *s_p1 = model(x, m, generate_gta=False, attn_ref=attn_ref)
                else:
                    with torch.no_grad(): _, m2_hat, attention, *s_p1 = model(x, m, generate_gta=False, attn_ref=attn_ref)

                # mask
                x, m2_hat, s_p1 = prepare_pass2_input(x, m2_hat, s_p1, hp.tts_extension_dct['input_prob_lst'])
                # import pdb; pdb.set_trace()

                # pass 2
                m1_hat_p2, m2_hat_p2, attention_p2, attention_vc = model_pass2(x, m, m2_hat, s_p1=s_p1, generate_gta=False, attn_ref=attn_ref)

            # print(x.size())
            # print(m.size())
            # print(m2_hat.size())
            # print(m1_hat_p2.size(), m2_hat_p2.size())
            # print(attention_p2.size(), attention_p2.size(1)*model.r)
            # print(attention_vc.size(), attention_vc.size(1)*model.r)
            # pdb.set_trace()
            # import pdb; pdb.set_trace()

            # m1_loss, m2_loss = F.l1_loss(m1_hat_p2, m), F.l1_loss(m2_hat_p2, m)
            # loss = (m1_loss + m2_loss) / hp.tts_batch_acu

            # if hp.tts_mode_train_pass1=='teacher_forcing':
            loss_out = (F.l1_loss(m1_hat_p2, m) + F.l1_loss(m2_hat_p2, m) + F.l1_loss(m1_hat, m) + F.l1_loss(m2_hat, m)) 

            attn_loss = F.kl_div(torch.log(smooth(attention)), smooth(attn_ref), reduction='none') # 'batchmean'
            attn_loss = attn_loss.sum(2).mean()
            attn_loss_p2 = F.kl_div(torch.log(smooth(attention_p2)), smooth(attn_ref), reduction='none') # 'batchmean'
            attn_loss_p2 = attn_loss_p2.sum(2).mean()
            loss_attn = (attn_loss + attn_loss_p2) * hp.attn_loss_coeff

            loss = loss_out + loss_attn

            (loss / hp.tts_batch_acu).backward()

            # w = model.encoder.embedding.weight.data
            # print(w.size(), w[0, :3])
            # w = model_pass2.encoder.embedding.weight.data
            # print(w.size(), w[0, :3])
            # pdb.set_trace()

            # grad = model.encoder.embedding.weight.grad
            # print(grad.size(), grad[0, :3])
            # grad = model_pass2.encoder.embedding.weight.grad
            # print(grad.size(), grad[0, :3])
            # pdb.set_trace()

            if (i+1)%hp.tts_batch_acu == 0:
                for _paths, _model, _optimizer in passModelOptimizer_lst:
                    # clip grad only once before updating the params with step()
                    if hp.tts_clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(), hp.tts_clip_grad_norm)
                        if np.isnan(grad_norm):
                            print('grad_norm was NaN!')
                    _optimizer.step()
                    _optimizer.zero_grad()
                #     print('wooooooooooooow')
                # w = model.encoder.embedding.weight.data
                # print(w.size(), w[0, :3])
                # w = model_pass2.encoder.embedding.weight.data
                # print(w.size(), w[0, :3])
                # pdb.set_trace()

            running_loss_out += loss_out.item()
            avg_loss_out = running_loss_out / i
            running_loss_attn += loss_attn.item()
            avg_loss_attn = running_loss_attn / i

            speed = i / (time.time() - start)

            # step = model.get_step()
            step = model_pass2.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                for _paths, _model, _optimizer in passModelOptimizer_lst:
                    save_checkpoint('tts', _paths, _model, _optimizer, name=ckpt_name, is_silent=True)
                # save_checkpoint('tts', paths, model, optimizer, name=ckpt_name, is_silent=True)
                # save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, name=ckpt_name, is_silent=True)

            # save_attention(np_now(attention_vc[0][:, :]), paths.tts_attention/f'{step}_speech')
            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attn_ref[idx][:, :160]), paths.tts_attention/f'{step}_text_ref')
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

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss_out: {avg_loss_out:#.4}; Output_attn: {avg_loss_attn:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        for _paths, _model, _optimizer in passModelOptimizer_lst:
            save_checkpoint('tts', _paths, _model, _optimizer, is_silent=True)
            _model.log(_paths.tts_log, msg)
        # save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        # model.log(paths.tts_log, msg)
        # save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, is_silent=True)
        # model_pass2.log(paths_pass2.tts_log, msg)
        print(' ')


def tts_train_loop_af_offline(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, 
    optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example, hp=None):
    # import pdb; pdb.set_trace()

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr
    for g in optimizer_pass2.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    passModelOptimizer_lst = []
    if hp.tts_updateP1: passModelOptimizer_lst.append((paths, model, optimizer))
    if hp.tts_updateP2: passModelOptimizer_lst.append((paths_pass2, model_pass2, optimizer_pass2))

    for e in range(1, epochs+1):

        start = time.time()
        running_loss_out, running_loss_attn = 0, 0

        optimizer.zero_grad()
        optimizer_pass2.zero_grad()

        # Perform 1 epoch
        for i, (x, m, ids, _, attn_ref) in enumerate(train_set, 1):

            x, m, attn_ref = x.to(device), m.to(device), attn_ref.to(device)
            # pdb.set_trace()

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                # pass1
                if hp.tts_updateP1:
                    m1_hat, m2_hat, attention, *s_p1 = model(x, m, generate_gta=False, attn_ref=attn_ref)
                else:
                    with torch.no_grad(): _, m2_hat, attention, *s_p1 = model(x, m, generate_gta=False, attn_ref=attn_ref)

                # mask
                x, m2_hat, s_p1 = prepare_pass2_input(x, m2_hat, s_p1, hp.tts_extension_dct['input_prob_lst'])
                # import pdb; pdb.set_trace()

                # pass 2
                m1_hat_p2, m2_hat_p2, attention_p2, attention_vc = model_pass2(x, m, m2_hat, s_p1=s_p1, generate_gta=False, attn_ref=attn_ref)

            # print(x.size())
            # print(m.size())
            # print(m2_hat.size())
            # print(m1_hat_p2.size(), m2_hat_p2.size())
            # print(attention_p2.size(), attention_p2.size(1)*model.r)
            # print(attention_vc.size(), attention_vc.size(1)*model.r)
            # pdb.set_trace()
            # import pdb; pdb.set_trace()

            # m1_loss, m2_loss = F.l1_loss(m1_hat_p2, m), F.l1_loss(m2_hat_p2, m)
            # loss = (m1_loss + m2_loss) / hp.tts_batch_acu

            # if hp.tts_mode_train_pass1=='teacher_forcing':
            loss_out = (F.l1_loss(m1_hat_p2, m) + F.l1_loss(m2_hat_p2, m) + F.l1_loss(m1_hat, m) + F.l1_loss(m2_hat, m)) 

            # attn_loss = F.kl_div(torch.log(smooth(attention)), smooth(attn_ref), reduction='none') + F.kl_div(torch.log(smooth(attention_p2)), smooth(attn_ref), reduction='none') # 'batchmean'
            # attn_loss = attn_loss.sum(2).mean()
            attn_loss = F.l1_loss(smooth(attention), smooth(attn_ref)) + F.l1_loss(smooth(attention_p2), smooth(attn_ref))
            loss_attn = attn_loss * hp.attn_loss_coeff

            loss = loss_out + loss_attn

            (loss / hp.tts_batch_acu).backward()

            # w = model.encoder.embedding.weight.data
            # print(w.size(), w[0, :3])
            # w = model_pass2.encoder.embedding.weight.data
            # print(w.size(), w[0, :3])
            # pdb.set_trace()

            # grad = model.encoder.embedding.weight.grad
            # print(grad.size(), grad[0, :3])
            # grad = model_pass2.encoder.embedding.weight.grad
            # print(grad.size(), grad[0, :3])
            # pdb.set_trace()

            if (i+1)%hp.tts_batch_acu == 0:
                for _paths, _model, _optimizer in passModelOptimizer_lst:
                    # clip grad only once before updating the params with step()
                    if hp.tts_clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(), hp.tts_clip_grad_norm)
                        if np.isnan(grad_norm):
                            print('grad_norm was NaN!')
                    _optimizer.step()
                    _optimizer.zero_grad()
                #     print('wooooooooooooow')
                # w = model.encoder.embedding.weight.data
                # print(w.size(), w[0, :3])
                # w = model_pass2.encoder.embedding.weight.data
                # print(w.size(), w[0, :3])
                # pdb.set_trace()

            running_loss_out += loss_out.item()
            avg_loss_out = running_loss_out / i
            running_loss_attn += loss_attn.item()
            avg_loss_attn = running_loss_attn / i

            speed = i / (time.time() - start)

            # step = model.get_step()
            step = model_pass2.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                for _paths, _model, _optimizer in passModelOptimizer_lst:
                    save_checkpoint('tts', _paths, _model, _optimizer, name=ckpt_name, is_silent=True)
                # save_checkpoint('tts', paths, model, optimizer, name=ckpt_name, is_silent=True)
                # save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, name=ckpt_name, is_silent=True)

            # save_attention(np_now(attention_vc[0][:, :]), paths.tts_attention/f'{step}_speech')
            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attn_ref[idx][:, :160]), paths.tts_attention/f'{step}_text_ref')
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

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss_out: {avg_loss_out:#.4}; Output_attn: {avg_loss_attn:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        for _paths, _model, _optimizer in passModelOptimizer_lst:
            save_checkpoint('tts', _paths, _model, _optimizer, is_silent=True)
            _model.log(_paths.tts_log, msg)
        # save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        # model.log(paths.tts_log, msg)
        # save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, is_silent=True)
        # model_pass2.log(paths_pass2.tts_log, msg)
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
