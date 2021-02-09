import torch
from models.fatchord_version import WaveRNN
from utils import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
import argparse
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table, save_spectrogram
from utils.dsp import reconstruct_waveform, save_wav
import numpy as np
from utils import get_gv

if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--tts_weights', type=str, help='[string/path] Load in different Tacotron weights')
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

    wr_parser.add_argument('--use_standard_names', action='store_true', help='use hp.test_sentences_names to name the generated audio samples')

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
    save_gv = args.save_gv
    if save_gv: gv_lst = []

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

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

    for i, x in enumerate(inputs, 1):

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x)
        # Fix mel spectrogram scaling to be from 0 to 1
        m = (m + 4) / 8
        np.clip(m, 0, 1, out=m)

        if args.vocoder == 'griffinlim':
            v_type = args.vocoder
        elif args.vocoder == 'wavernn' and args.batched:
            v_type = 'wavernn_batched'
        else:
            v_type = 'wavernn_unbatched'

        if input_text:
            save_path = paths.tts_output/f'__input_{input_text[:10]}_{v_type}_{tts_k}k.wav'
        else:
            save_path = paths.tts_output/f'{i}_{v_type}_{tts_k}k.wav'
        if args.use_standard_names:
            save_path = paths.tts_output/f'{hp.test_sentences_names[i-1]}.wav'

        if save_attn: save_attention(attention, save_path)

        if save_mel: save_spectrogram(m, str(save_path).replace('.wav', '_mel'), 600)

        if save_gv: gv_lst.append(get_gv(m))

        if not args.skip_wav:
            if args.vocoder == 'wavernn':
                m = torch.tensor(m).unsqueeze(0)
                voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
            elif args.vocoder == 'griffinlim':
                wav = reconstruct_waveform(m, n_iter=args.iters)
                save_wav(wav, save_path)

    if save_gv: np.save(paths.tts_output/'gv_array', np.array(gv_lst))

    print('\n\nDone.\n')
