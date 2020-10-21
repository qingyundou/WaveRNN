import torch
from models.fatchord_version import WaveRNN
from utils import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths, Paths_multipass
from models.tacotron import Tacotron, Tacotron_pass2
import argparse
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table, save_spectrogram
from utils.dsp import reconstruct_waveform, save_wav
import numpy as np
import os

if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--tts_weights', type=str, help='[string/path] Load in different Tacotron weights')
    parser.add_argument('--save_attention', '-a', dest='save_attn', action='store_true', help='Save Attention Plots')
    parser.add_argument('--save_mel', '-m', action='store_true', help='Save Mel Plots')
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
    save_attn = args.save_attn
    save_mel = args.save_mel

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
    tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
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
                         stop_threshold=hp.tts_stop_threshold).to(device)

    tts_load_path = tts_weights if tts_weights else paths.tts_latest_weights
    tts_model.load(tts_load_path)

    tts_model_pass2 = Tacotron_pass2(embed_dims=hp.tts_embed_dims,
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
                     encoder_reduction_factor=hp.tts_encoder_reduction_factor).to(device)

    tts_load_path = tts_weights if tts_weights else paths_pass2.tts_latest_weights
    tts_model_pass2.load(tts_load_path)

    if input_text:
        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    else:
        test_sentences_file = hp.test_sentences_file if hasattr(hp, 'test_sentences_file') else 'test_sentences/sentences.txt'
        with open(test_sentences_file) as f:
            inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]

    if args.vocoder == 'wavernn':
        voc_k = voc_model.get_step() // 1000
        tts_k = tts_model.get_step() // 1000

        simple_table([('Tacotron', str(tts_k) + 'k'),
                    ('r', tts_model.r),
                    ('Vocoder Type', 'WaveRNN'),
                    ('WaveRNN', str(voc_k) + 'k'),
                    ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                    ('Target Samples', target if batched else 'N/A'),
                    ('Overlap Samples', overlap if batched else 'N/A')])

    elif args.vocoder == 'griffinlim':
        tts_k = tts_model.get_step() // 1000
        simple_table([('Tacotron', str(tts_k) + 'k'),
                    ('r', tts_model.r),
                    ('Vocoder Type', 'Griffin-Lim'),
                    ('GL Iters', args.iters)])

    tmp = paths.tts_output/hp.tts_pass2_input_gen
    print(f'output dir: {tmp}')
    os.makedirs(tmp, exist_ok=True)
    for i, x in enumerate(inputs, 1):

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x)
        # print(m)
        # print(m.size())
        # print(m.unsqueeze().size())
        # import pdb; pdb.set_trace()

        if hp.tts_pass2_input_gen=='y1':
            x = [0 for x_i in x]
        elif hp.tts_pass2_input_gen=='x':
            m = m * 0
        elif hp.tts_pass2_input_gen=='xNy1':
            pass

        # m = torch.tensor(m).unsqueeze(0).to(device)
        _, m_p2, attention_p2, attention_vc = tts_model_pass2.generate(x, torch.tensor(m).unsqueeze(0).to(device))
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
            save_attention(attention_vc, save_path.parent/(save_path.stem+'_attn_vc'))

        if save_mel:
            save_spectrogram(m, str(save_path).replace('.wav', '_p1'), 600)
            save_spectrogram(m_p2, str(save_path).replace('.wav', '_p2'), 600)

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

    print('\n\nDone.\n')
