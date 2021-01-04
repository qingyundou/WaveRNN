import torch
from models.fatchord_version import WaveRNN
from utils import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths, Paths_multipass
from models.tacotron import Tacotron, Tacotron_pass2, Tacotron_pass1, Tacotron_pass2_concat, Tacotron_pass2_delib, Tacotron_pass2_delib_shareEnc, Tacotron_pass2_attn, Tacotron_pass2_attnAdv
from models.tacotron import Tacotron_pass1_smartKV, Tacotron_pass2_attnAdv_smartKV
import argparse
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table, save_spectrogram
from utils.dsp import reconstruct_waveform, save_wav
import numpy as np
import os
from utils import get_gv
from train_tacotron_pass2 import prepare_pass2_input

if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--tts_weights', type=str, help='[string/path] Load in different Tacotron weights')
    parser.add_argument('--tts_weights_pass2', type=str, help='[string/path] Load in different Tacotron weights')
    parser.add_argument('--save_attention', '-a', dest='save_attn', action='store_true', help='Save Attention Plots')
    parser.add_argument('--save_mel', '-m', action='store_true', help='Save Mel Plots')
    parser.add_argument('--save_gv', action='store_true', help='Save the global variance of all mels in a npy array')
    parser.add_argument('--skip_wav', action='store_true', help='Skip wavform generation')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    parser.add_argument('--use_standard_names', action='store_true', help='use hp.test_sentences_names to name the generated audio samples')

    parser.set_defaults(input_text=None)
    parser.set_defaults(weights_path=None)

    # name of subcommand goes to args.vocoder
    subparsers = parser.add_subparsers(required=True, dest='vocoder')

    wr_parser = subparsers.add_parser('wavernn', aliases=['wr'])
    wr_parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    wr_parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    wr_parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    wr_parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    wr_parser.add_argument('--voc_weights', type=str, help='[string/path] Load in different WaveRNN weights')
    wr_parser.set_defaults(batched=None)

    gl_parser = subparsers.add_parser('griffinlim', aliases=['gl'])
    gl_parser.add_argument('--iters', type=int, default=32, help='[int] number of griffinlim iterations')

    args = parser.parse_args()

    if args.vocoder in ['griffinlim', 'gl']:
        args.vocoder = 'griffinlim'
    elif args.vocoder in ['wavernn', 'wr']:
        args.vocoder = 'wavernn'
    else:
        raise argparse.ArgumentError('Must provide a valid vocoder type!')

    hp.configure(args.hp_file)  # Load hparams from file
    hp.fix_compatibility()
    # set defaults for any arguments that depend on hparams
    if args.vocoder == 'wavernn':
        if args.target is None:
            args.target = hp.voc_target
        if args.overlap is None:
            args.overlap = hp.voc_overlap
        if args.batched is None:
            args.batched = hp.voc_gen_batched

        batched = args.batched
        target = args.target
        overlap = args.overlap

    input_text = args.input_text
    tts_weights = args.tts_weights
    tts_weights_pass2 = args.tts_weights_pass2
    save_attn = args.save_attn
    save_mel = args.save_mel
    save_gv = args.save_gv or hp.tts_save_gv
    if save_gv: gv_lst_p1, gv_lst_p2 = [], []

    # paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    paths = Paths_multipass(hp.data_path, hp.voc_model_id, hp.tts_model_id, 'pass1')
    paths_pass2 = Paths_multipass(hp.data_path, hp.voc_model_id, hp.tts_model_id, 'pass2')

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    if args.vocoder == 'wavernn':
        print('\nInitialising WaveRNN Model...\n')
        # Instantiate WaveRNN Model
        voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                            fc_dims=hp.voc_fc_dims,
                            bits=hp.bits,
                            pad=hp.voc_pad,
                            upsample_factors=hp.voc_upsample_factors,
                            feat_dims=hp.num_mels,
                            compute_dims=hp.voc_compute_dims,
                            res_out_dims=hp.voc_res_out_dims,
                            res_blocks=hp.voc_res_blocks,
                            hop_length=hp.hop_length,
                            sample_rate=hp.sample_rate,
                            mode=hp.voc_mode).to(device)

        voc_load_path = args.voc_weights if args.voc_weights else paths.voc_latest_weights
        voc_model.load(voc_load_path)

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    # Taco = Tacotron_pass1 if 's1' in hp.tts_pass2_input_train else Tacotron
    Taco_dct = {'Tacotron':Tacotron, 'Tacotron_pass1':Tacotron_pass1, 'Tacotron_pass1_smartKV':Tacotron_pass1_smartKV}
    Taco = Taco_dct[hp.tts_model_pass1]
    tts_model = Taco(embed_dims=hp.tts_embed_dims,
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
                         mode=hp.tts_mode_gen_pass1,
                         share_encoder=hp.tts_pass2_delib_shareEnc).to(device)

    # tts_model.load(hp.tts_init_weights_path)
    # for i, (name, param) in enumerate(tts_model.named_parameters()):
    #     print('tts_model before', name, param.data.size())
    #     print(param.data[0,:3]) if len(param.data.size())>1 else print(param.data[:3])
    #     if i>1: break

    tts_load_path = tts_weights if tts_weights else paths.tts_latest_weights
    tts_model.load(tts_load_path)

    # for i, (name, param) in enumerate(tts_model.named_parameters()):
    #     print('tts_model after', name, param.data.size())
    #     print(param.data[0,:3]) if len(param.data.size())>1 else print(param.data[:3])
    #     if i>1: break
    # import pdb; pdb.set_trace()

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
    tts_model_pass2 = Taco_p2(embed_dims=hp.tts_embed_dims,
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
                     mode=hp.tts_mode_gen_pass2,
                     encoder_reduction_factor=hp.tts_encoder_reduction_factor,
                     encoder_reduction_factor_s=hp.tts_encoder_reduction_factor_s,
                     pass2_input=hp.tts_pass2_input_train).to(device)

    tts_load_path = tts_weights_pass2 if tts_weights_pass2 else paths_pass2.tts_latest_weights
    tts_model_pass2.load(tts_load_path)

    if input_text:
        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    else:
        test_sentences_file = hp.test_sentences_file if hasattr(hp, 'test_sentences_file') else 'test_sentences/sentences.txt'
        with open(test_sentences_file) as f:
            inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]

    if args.vocoder == 'wavernn':
        voc_k = voc_model.get_step() // 1000
        # tts_k = tts_model.get_step() // 1000
        tts_k = tts_model_pass2.get_step() // 1000

        simple_table([('Tacotron', str(tts_k) + 'k'),
                    ('r', tts_model.r),
                    ('Vocoder Type', 'WaveRNN'),
                    ('WaveRNN', str(voc_k) + 'k'),
                    ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                    ('Target Samples', target if batched else 'N/A'),
                    ('Overlap Samples', overlap if batched else 'N/A')])

    elif args.vocoder == 'griffinlim':
        # tts_k = tts_model.get_step() // 1000
        tts_k = tts_model_pass2.get_step() // 1000

        simple_table([('Tacotron', str(tts_k) + 'k'),
                    ('r', tts_model.r),
                    ('Vocoder Type', 'Griffin-Lim'),
                    ('GL Iters', args.iters)])

    tmp = paths.tts_output/hp.tts_pass2_input_gen
    print(f'output dir: {tmp}')
    os.makedirs(tmp, exist_ok=True)
    for i, x in enumerate(inputs, 1):

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention, *inter_p1 = tts_model.generate(x)
        # print(m)
        # print(m.size())
        # print(m.unsqueeze().size())
        # for i in inter_p1:
        #     print(i.size())
        # import pdb; pdb.set_trace()

        # if 'x' not in hp.tts_pass2_input_gen:
        #     x = [0 for x_i in x]
        # if 'y1' not in hp.tts_pass2_input_gen:
        #     m = m * 0
        #     if 's1' in hp.tts_pass2_input_train: s = s * 0

        input_prob_lst = [0., 0., 1.]
        if 'x' not in hp.tts_pass2_input_gen:
            input_prob_lst = [0., 1., 0.]
        if 'y1' not in hp.tts_pass2_input_gen and 's1' not in hp.tts_pass2_input_gen:
            input_prob_lst = [1., 0., 0.]

        x, m, inter_p1_dct = prepare_pass2_input(x, m, [i.to(device) for i in inter_p1], input_prob_lst)
        _, m_p2, attention_p2, *attention_vc_lst = tts_model_pass2.generate(x, torch.tensor(m).unsqueeze(0).to(device), **inter_p1_dct)

        # if 's1' in hp.tts_pass2_input_train:
        #     _, m_p2, attention_p2, attention_vc = tts_model_pass2.generate(x, torch.tensor(m).unsqueeze(0).to(device), s)
        # else:
        #     _, m_p2, attention_p2, attention_vc = tts_model_pass2.generate(x, torch.tensor(m).unsqueeze(0).to(device))

        # Fix mel spectrogram scaling to be from 0 to 1
        m = (m + 4) / 8
        np.clip(m, 0, 1, out=m)
        m_p2 = (m_p2 + 4) / 8
        np.clip(m_p2, 0, 1, out=m_p2)

        if args.vocoder == 'griffinlim':
            v_type = args.vocoder
        elif args.vocoder == 'wavernn' and args.batched:
            v_type = 'wavernn_batched'
        else:
            v_type = 'wavernn_unbatched'

        if input_text:
            save_path = paths.tts_output/f'{hp.tts_pass2_input_gen}'/f'__input_{input_text[:10]}_{v_type}_{tts_k}k.wav'
        else:
            save_path = paths.tts_output/f'{hp.tts_pass2_input_gen}'/f'{i}_{v_type}_{tts_k}k.wav'
        if args.use_standard_names:
            save_path = paths.tts_output/f'{hp.tts_pass2_input_gen}'/f'{hp.test_sentences_names[i-1]}.wav'
        save_path_p2 = str(save_path).replace('.wav', '_p2.wav')

        if save_attn:
            save_attention(attention, save_path.parent/(save_path.stem+'_attn_p1'))
            save_attention(attention_p2, save_path.parent/(save_path.stem+'_attn_p2'))
            for tmp_i, attention_vc in enumerate(attention_vc_lst):
                save_attention(attention_vc, save_path.parent/(save_path.stem+f'_attn_vc_{tmp_i}'))

        if save_mel:
            save_spectrogram(m, str(save_path).replace('.wav', '_p1'), 600)
            save_spectrogram(m_p2, str(save_path).replace('.wav', '_p2'), 600)

        if save_gv:
            gv_lst_p1.append(get_gv(m))
            gv_lst_p2.append(get_gv(m_p2))

        if not args.skip_wav:
            if args.vocoder == 'wavernn':
                m = torch.tensor(m).unsqueeze(0)
                voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
                m_p2 = torch.tensor(m_p2).unsqueeze(0)
                voc_model.generate(m_p2, save_path_p2, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
            elif args.vocoder == 'griffinlim':
                wav = reconstruct_waveform(m, n_iter=args.iters)
                save_wav(wav, save_path)
                wav = reconstruct_waveform(m_p2, n_iter=args.iters)
                save_wav(wav, save_path_p2)

        if save_gv:
            np.save(paths.tts_output/f'{hp.tts_pass2_input_gen}'/f'gv_p1_array_{tts_k}k', np.array(gv_lst_p1))
            np.save(paths.tts_output/f'{hp.tts_pass2_input_gen}'/f'gv_p2_array_{tts_k}k', np.array(gv_lst_p2))

    print('\n\nDone.\n')
