import torch
from torch import optim
import torch.nn.functional as F
from utils import hparams as hp
from utils.display import *
from utils.dataset import get_tts_datasets
from utils.text.symbols import symbols
from utils.paths import Paths, Paths_multipass
from models.tacotron import Tacotron, Tacotron_pass2, Tacotron_pass1, Tacotron_pass2_concat, Tacotron_pass2_delib, Tacotron_pass2_delib_shareEnc, Tacotron_pass2_attn, Tacotron_pass2_attnAdv
from models.tacotron import Tacotron_pass1_smartKV, Tacotron_pass2_attnAdv_smartKV
import argparse
from utils import data_parallel_workaround, set_global_seeds
import os
from pathlib import Path
import time
import numpy as np
import sys
from utils.checkpoints import save_checkpoint, restore_checkpoint

sys.path.append("/home/dawna/tts/qd212/models/espnet")
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import GuidedAttentionLoss

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
    hp.fix_compatibility()
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
    # Taco = Tacotron_pass1 # if 's1' in hp.tts_pass2_input_train else Tacotron
    Taco_dct = {'Tacotron':Tacotron, 'Tacotron_pass1':Tacotron_pass1, 'Tacotron_pass1_smartKV':Tacotron_pass1_smartKV}
    Taco = Taco_dct[hp.tts_model_pass1]
    # tmp_dct = {}
    # if 'Tacotron_pass1' in hp.tts_model_pass1: tmp_dct['share_encoder'] = True
    # if 'smartKV' in hp.tts_model_pass1: tmp_dct['output_context'] = True

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
                     mode=hp.tts_mode_train_pass1,
                     fr_length_ratio=hp.tts_fr_length_ratio,
                     share_encoder=hp.tts_pass2_delib_shareEnc).to(device)

    Taco_p2 = Tacotron_pass2
    if hp.tts_pass2_concat:
        Taco_p2 = Tacotron_pass2_concat
    if hp.tts_pass2_delib:
        Taco_p2 = Tacotron_pass2_delib
    if hp.tts_pass2_delib_shareEnc:
        Taco_p2 = Tacotron_pass2_delib_shareEnc
    if hp.tts_pass2_attn:
        Taco_p2 = Tacotron_pass2_attn
    if hp.tts_pass2_attnAdv:
        Taco_p2 = Tacotron_pass2_attnAdv
    Taco_p2_dct = {'Tacotron_pass2':Tacotron_pass2, 'Tacotron_pass2_concat':Tacotron_pass2_concat, 
    'Tacotron_pass2_delib':Tacotron_pass2_delib, 'Tacotron_pass2_delib_shareEnc':Tacotron_pass2_delib_shareEnc, 
    'Tacotron_pass2_attn':Tacotron_pass2_attn, 'Tacotron_pass2_attnAdv':Tacotron_pass2_attnAdv, 
    'Tacotron_pass2_attnAdv_smartKV':Tacotron_pass2_attnAdv_smartKV}
    if hp.tts_model_pass2 in Taco_p2_dct.keys(): Taco_p2 = Taco_p2_dct[hp.tts_model_pass2]

    model_pass2 = Taco_p2(embed_dims=hp.tts_embed_dims,
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

    # import pdb; pdb.set_trace()
    # tmp=model_pass2.parameters()
    # print(type(tmp), len(list(tmp)))
    # tmp=model_pass2.named_parameters()
    # print(type(tmp), len(dict(tmp)))
    # tmp = optimizer_pass2.param_groups
    # print(type(tmp), len(tmp), len(tmp[0]['params']))
    # tmp_rg = [int(x.requires_grad) for x in tmp[0]['params']]
    # print(tmp_rg)


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

    if hp.tts_use_guided_attn_loss:
        guided_attn_loss = GuidedAttentionLoss(
            sigma=hp.tts_guided_attn_loss_sigma,
            alpha=hp.tts_guided_attn_loss_lambda,
            ).to(device)
    else:
        guided_attn_loss = None


    if not (force_gta or force_attn):
        for i, session in enumerate(hp.tts_schedule):
            current_step = model.get_step()

            r, lr, max_step, batch_size, *extension = session
            if len(extension)>0: hp.tts_extension_dct['input_prob_lst'] = extension[0]
            if len(extension)>1: hp.tts_extension_dct['gal_coeff'] = extension[1]
            if len(extension)>2:
                hp.tts_extension_dct['params_to_train'] = extension[2]
                # optimizer_pass2 = update_optimizer(model_pass2, optimizer_pass2, hp.tts_extension_dct['params_to_train'])
                update_model(model_pass2, hp.tts_extension_dct['params_to_train'])

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
                            ('Batch Accumulation', hp.tts_batch_acu),
                            ('Learning Rate', lr),
                            ('Outputs/Step (r)', model.r),
                            ('p2_input_prob_lst', hp.tts_extension_dct['input_prob_lst']),
                            ('params_to_train', hp.tts_extension_dct['params_to_train'])])

            train_set, attn_example = get_tts_datasets(paths.data, batch_size, r)
            tts_train_loop(paths, paths_pass2, model, model_pass2, optimizer, optimizer_pass2, train_set, lr, training_steps, attn_example, 
                hp=hp, model_tf=model_tf, guided_attn_loss=guided_attn_loss)

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


def prepare_pass2_input(x, y_p1, inter_p1, input_prob_lst):
    # unpack the flexible lst, if it is not empty
    k_lst = ['s_p1', 'e_p1', 'e_p_p1', 'c_p1']
    inter_p1_dct = {}
    for i,v in enumerate(inter_p1):
        inter_p1_dct[k_lst[i]] = v

    # s_p1, e_p1, e_p_p1 = None, None, None
    # nb_inter = len(inter_p1)
    # if nb_inter>0: s_p1 = inter_p1[0]
    # if nb_inter>1: e_p1 = inter_p1[1]
    # if nb_inter>2: e_p_p1 = inter_p1[2]

    if sum(input_prob_lst)!=1: print(f'qd212 warning: input_prob_lst {input_prob_lst} doesnt sum to 1')
    p_x, p_y, p_both = [float(p) for p in input_prob_lst]

    if np.random.uniform(high=1.0)<p_both:
        pass
    elif np.random.uniform(high=p_both)<p_x:
        x = x * 0
        if 'e_p1' in inter_p1_dct: inter_p1_dct['e_p1'] = inter_p1_dct['e_p1'] * 0
        if 'e_p_p1' in inter_p1_dct: inter_p1_dct['e_p_p1'] = inter_p1_dct['e_p_p1'] * 0
        # if (e_p1 is not None) and (e_p_p1 is not None): e_p1, e_p_p1 = e_p1*0, e_p_p1*0
    else:
        y_p1 = y_p1 * 0
        if 's_p1' in inter_p1_dct: inter_p1_dct['s_p1'] = inter_p1_dct['s_p1'] * 0
        # if s_p1 is not None: s_p1 = s_p1 * 0
    return x, y_p1, inter_p1_dct

def pass2_input_lst2mask(input_prob_lst):
    if sum(input_prob_lst)!=1: print(f'qd212 warning: input_prob_lst {input_prob_lst} doesnt sum to 1')
    p_mask_x, p_mask_y, p_mask_none = [float(p) for p in input_prob_lst]

    input_mask_dct = {'input':1, 'output_p1':1}
    if np.random.uniform(high=1.0)<p_mask_none:
        pass
    elif np.random.uniform(high=1-p_mask_none)<p_mask_x:
        input_mask_dct['input'] = 0
    else:
        input_mask_dct['output_p1'] = 0
    return input_mask_dct

def prepare_pass2_input_w_mask(x, y_p1, inter_p1, input_mask_dct, mask_target='input'):
    # unpack the flexible lst, if it is not empty
    k_lst = ['s_p1', 'e_p1', 'e_p_p1', 'c_p1']
    inter_p1_dct = {}
    for i,v in enumerate(inter_p1):
        inter_p1_dct[k_lst[i]] = v

    input_mask_x, input_mask_y = input_mask_dct['input'], input_mask_dct['output_p1']
    
    if mask_target=='input':
        x = x * input_mask_x
        if 'e_p1' in inter_p1_dct: inter_p1_dct['e_p1'] = inter_p1_dct['e_p1'] * input_mask_x
        if 'e_p_p1' in inter_p1_dct: inter_p1_dct['e_p_p1'] = inter_p1_dct['e_p_p1'] * input_mask_x
        y_p1 = y_p1 * input_mask_y
        if 's_p1' in inter_p1_dct: inter_p1_dct['s_p1'] = inter_p1_dct['s_p1'] * input_mask_y

    elif mask_target=='context':
        inter_p1_dct['input_mask_x'] = input_mask_x
        inter_p1_dct['input_mask_y'] = input_mask_y

    return x, y_p1, inter_p1_dct

def update_optimizer(model, optimizer, group_name_lst):
    if 'all' in group_name_lst:
        params_lst = [ p for p in model.parameters() ]
    else:
        params_lst = [ p for n, p in model.named_parameters() if 'decoder.' in n ]
        if 'enc_vc0' in group_name_lst:
            params_lst += [ p for n, p in model.named_parameters() if 'encoder_vc.' in n or 'encoder_proj_vc.' in n ]
        if 'enc_vc1' in group_name_lst:
            params_lst += [ p for n, p in model.named_parameters() if 'encoder_vc_global.' in n or 'encoder_proj_vc_global.' in n ]
    optimizer.param_groups[0]['params'] = params_lst
    return optimizer

def update_model(model, group_name_lst):
    if 'all' in group_name_lst:
        n_lst = [ n for n, p in model.named_parameters() ]
    else:
        n_lst = [ n for n, p in model.named_parameters() if 'decoder.' in n]
        if 'enc_vc0' in group_name_lst:
            n_lst += [ n for n, p in model.named_parameters() if 'encoder_vc.' in n or 'encoder_proj_vc.' in n]
        if 'enc_vc1' in group_name_lst:
            n_lst += [ n for n, p in model.named_parameters() if 'encoder_vc_global.' in n or 'encoder_proj_vc_global.' in n]

    for n, p in model.named_parameters():
        p.requires_grad = True if n in n_lst else False
    # return model


def tts_train_loop(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, optimizer, optimizer_pass2, 
    train_set, lr, train_steps, attn_example, hp=None, model_tf=None, guided_attn_loss=None):
    if hp.mode=='teacher_forcing':
        if hp.tts_use_guided_attn_loss:
            tts_train_loop_tf_gal(paths, paths_pass2, model, model_pass2, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example, guided_attn_loss=guided_attn_loss)
        else:
            tts_train_loop_tf(paths, paths_pass2, model, model_pass2, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example)
    # elif hp.mode=='attention_forcing_online':
    #     tts_train_loop_af_online(paths, model, model_tf, optimizer, train_set, lr, train_steps, attn_example, hp=hp)
    # elif hp.mode=='attention_forcing_offline':
    #     tts_train_loop_af_offline(paths, model, optimizer, train_set, lr, train_steps, attn_example, hp=hp)
    else:
        raise NotImplementedError(f'hp.mode={hp.mode} is not yet implemented')


def tts_train_loop_tf(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, optimizer, optimizer_pass2, train_set, lr, train_steps, attn_example):
    # import pdb; pdb.set_trace()

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0

        # optimizer.zero_grad()
        optimizer_pass2.zero_grad()
        input_mask_dct = pass2_input_lst2mask(hp.tts_extension_dct['input_prob_lst'])

        # Perform 1 epoch
        for i, (x, m, ids, _) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)
            # pdb.set_trace()

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                # pass1
                with torch.no_grad(): _, m2_hat, attention, *inter_p1 = model(x, m)

                # mask
                # x, m2_hat, inter_p1_dct = prepare_pass2_input(x, m2_hat, inter_p1, hp.tts_extension_dct['input_prob_lst'])
                x, m2_hat, inter_p1_dct = prepare_pass2_input_w_mask(x, m2_hat, inter_p1, input_mask_dct, hp.tts_mask_target)
                # for k,v in inter_p1_dct.items():
                #     print(k, v.size())
                # import pdb; pdb.set_trace()

                # pass 2
                m1_hat_p2, m2_hat_p2, attention_p2, *attention_vc_lst = model_pass2(x, m, m2_hat, **inter_p1_dct)

            # print(x.size())
            # print(m.size())
            # print(m2_hat.size())
            # print(m1_hat_p2.size(), m2_hat_p2.size())
            # print(attention_p2.size(), attention_p2.size(1)*model.r)
            # for attention_vc in attention_vc_lst:
            #     print(attention_vc.size(), attention_vc.size(1)*model.r)
            # pdb.set_trace()

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
                # grad = model_pass2.decoder.attn_net.L.weight.grad
                # grad = model_pass2.postnet.conv_project1.conv.weight #.grad
                # print(grad.size(), grad[0, :3])
                # pdb.set_trace()
                # clip grad only once before updating the params with step()
                if hp.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model_pass2.parameters(), hp.tts_clip_grad_norm)
                    if np.isnan(grad_norm):
                        print('grad_norm was NaN!')
                optimizer_pass2.step()
                optimizer_pass2.zero_grad()
                input_mask_dct = pass2_input_lst2mask(hp.tts_extension_dct['input_prob_lst'])

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
                for tmp_i, attention_vc in enumerate(attention_vc_lst):
                    save_attention(np_now(attention_vc[idx][:, :]), paths.tts_attention/f'{step}_speech_{tmp_i}')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}_p1', 600)
                save_spectrogram(np_now(m2_hat_p2[idx]), paths.tts_mel_plot/f'{step}_p2', 600)
            # for idx, tmp in enumerate(ids):
            #     if tmp in ['LJ035-0011', 'LJ016-0320']: # selected egs
            #         save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}_{tmp}_p1')
            #         save_spectrogram(np_now(m2_hat_p2[idx]), paths.tts_mel_plot/f'{step}_{tmp}_p2')

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        # save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        # model.log(paths.tts_log, msg)
        save_checkpoint('tts', paths_pass2, model_pass2, optimizer_pass2, is_silent=True)
        model_pass2.log(paths_pass2.tts_log, msg)
        print(' ')


def tts_train_loop_tf_gal(paths: Paths, paths_pass2: Paths, model: Tacotron, model_pass2: Tacotron_pass2, optimizer, optimizer_pass2, 
    train_set, lr, train_steps, attn_example, guided_attn_loss=None):
    # import pdb; pdb.set_trace()
    # model.mode = hp.mode_pass1 # 'free_running', asup

    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss, running_loss_gal = 0, 0

        # optimizer.zero_grad()
        optimizer_pass2.zero_grad()

        # Perform 1 epoch
        for i, (x, m, ids, mlens) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)
            # pdb.set_trace()

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                _, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                # pass1
                with torch.no_grad(): _, m2_hat, attention, *inter_p1 = model(x, m)

                # mask
                x, m2_hat, inter_p1_dct = prepare_pass2_input(x, m2_hat, inter_p1, hp.tts_extension_dct['input_prob_lst'])

                # pass2
                m1_hat_p2, m2_hat_p2, attention_p2, *attention_vc_lst = model_pass2(x, m, m2_hat, **inter_p1_dct)
                attention_vc = attention_vc_lst[0]

            # print(x.size())
            # print(m.size())
            # print(m1_hat.size(), m2_hat.size())
            # print(m1_hat_p2.size(), m2_hat_p2.size())
            # print(attention_p2.size(), attention_p2.size(1)*model.r)
            # print(attention_vc.size(), attention_vc.size(1)*model.r)
            # pdb.set_trace()
            # for idx, tmp in enumerate(ids):
            #     save_spectrogram(np_now(m1_hat[idx]), paths.tts_mel_plot/f'nomask_{idx}_{tmp}_p1_m1')
            #     save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'nomask_{idx}_{tmp}_p1_m2')
            #     save_spectrogram(np_now(m[idx]), paths.tts_mel_plot/f'{idx}_{tmp}_ref')

            m1_loss = F.l1_loss(m1_hat_p2, m)
            m2_loss = F.l1_loss(m2_hat_p2, m)
            loss_y = (m1_loss + m2_loss)

            ilens = ((m2_hat.size(-1) - (m2_hat < model.stop_threshold).int().long().prod(1).sum(-1)) // model_pass2.encoder_reduction_factor).to(device)
            # ilens = ((m1_hat.size(-1) - (m1_hat==-4).int().long().prod(1).sum(-1)) // model_pass2.encoder_reduction_factor).to(device)
            olens = (torch.tensor([l//model_pass2.r for l in mlens]).long()).to(device)

            tmp = (ilens * model_pass2.encoder_reduction_factor) / float(m.size(-1))
            if (tmp<0.5).any(): print('warning: len(mel_fr) < 0.5 * len(mel_ref)')
            # print(ilens)
            # print(olens)
            # print(tmp)
            # print(m2_hat.size(-1) / float(m.size(-1)))
            ilens[ilens==torch.max(ilens)] = attention_vc.size(2)
            olens[olens==torch.max(olens)] = attention_vc.size(1)
            # if hp.tts_bin_lengths:
            #     ilens[:] = attention_vc.size(2)
            #     olens[:] = attention_vc.size(1)
            # else:
            #     ilens[ilens==torch.max(ilens)] = attention_vc.size(2)
            #     olens[olens==torch.max(olens)] = attention_vc.size(1)
            # print(ilens)
            # print(olens)
            # pdb.set_trace()
            loss_gal = guided_attn_loss(attention_vc, ilens, olens)

            loss = loss_y + loss_gal * hp.tts_extension_dct['gal_coeff']

            (loss / hp.tts_batch_acu).backward()
            if (i+1)%hp.tts_batch_acu == 0:
                if hp.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model_pass2.parameters(), hp.tts_clip_grad_norm)
                    if np.isnan(grad_norm):
                        print('grad_norm was NaN!')
                        import pdb; pdb.set_trace()
                optimizer_pass2.step()
                optimizer_pass2.zero_grad()

            running_loss += loss_y.item()
            avg_loss = running_loss / i
            running_loss_gal += loss_gal.item()
            avg_loss_gal = running_loss_gal / i

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
                for tmp_i, attention_vc in enumerate(attention_vc_lst):
                    save_attention(np_now(attention_vc[idx][:, :]), paths.tts_attention/f'{step}_speech_{tmp_i}')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}_p1', 600)
                save_spectrogram(np_now(m2_hat_p2[idx]), paths.tts_mel_plot/f'{step}_p2', 600)
            # for idx, tmp in enumerate(ids):
            #     # import pdb; pdb.set_trace()
            #     if tmp in ['LJ035-0011', 'LJ016-0320']: # selected egs
            #         save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}_{tmp}_p1')
            #         save_spectrogram(np_now(m2_hat_p2[idx]), paths.tts_mel_plot/f'{step}_{tmp}_p2')

            tmp = hp.tts_extension_dct['gal_coeff']
            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | Guided attn loss: {avg_loss_gal:#.4} * {tmp} | {speed:#.2} steps/s | Step: {k}k | '
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
