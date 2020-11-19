
# CONFIG -----------------------------------------------------------------------------------------------------------#

# Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
wav_path = '/home/dawna/tts/data/LJSpeech-1.1/webData/wavs/'
data_path = 'data/'

## hparams to tune
tts_batch_size = 10 # 32 64 100
tts_batch_acu = 10
_tts_adjust_steps = True
tts_encoder_reduction_factor = 4
tts_encoder_reduction_factor_s = tts_encoder_reduction_factor // 2 # quick fix
# tts_pass2_input_train = 'y1' # 'y1' 'x_y1' 'x_y1c1'

# model ids are separate - that way you can use a new tts with an old wavernn and vice versa
# NB: expect undefined behaviour if models were trained on different DSP settings
# exp_id = f'mp_lj_pass2_asup'


tts_batch_size = 16 # 32 64 100
tts_batch_acu = 2

tts_init_weights_path = '/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_pretrainGold_bs32.tacotron/latest_weights.pyt' # initial weights, usually from a pretrained model
tts_init_weights_path_pass2 = '/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_pretrainGold_bs32.tacotron/latest_weights.pyt'

_tmp = 1 if not _tts_adjust_steps else tts_batch_acu
tts_schedule = [(2,  1e-3, 10_000 * _tmp,  tts_batch_size, [1, 0, 0]),   # progressive training schedule
                (2,  1e-3, 20_000 * _tmp,  tts_batch_size, [0.25, 0.25, 0.5]),   # (r, lr, step, batch_size, tts_pass2_input_prob_lst: x_y_both)
                (2,  5e-4, 30_000 * _tmp,  tts_batch_size, [0.05, 0.05, 0.9]),
                (2,  5e-4, 40_000 * _tmp,  tts_batch_size, [0.05, 0.05, 0.9]),
                (2,  1e-4, 80_000 * _tmp,  tts_batch_size, [0.05, 0.05, 0.9])]

tts_pass2_input_train = 'xAOy1s1'
# tts_fr_length_ratio = 1.0
# exp_id = f'mp_lj_pass2_fixBestP1_BS{tts_batch_size*tts_batch_acu}_stepD{tts_batch_acu}_max80k_p1fr_re4_{tts_pass2_input_train}'

tts_fr_length_ratio = 1.2
# exp_id = f'mp_lj_pass2_fixBestP1_BS{tts_batch_size*tts_batch_acu}_stepD{tts_batch_acu}_max80k_p1fr_frL{tts_fr_length_ratio}_re4_{tts_pass2_input_train}'


## better schedule
# tts_schedule = [(2,  1e-3, 10_000 * _tmp,  tts_batch_size, [1, 0, 0]),   # progressive training schedule
#                 (2,  1e-3, 20_000 * _tmp,  tts_batch_size, [0.5, 0.0, 0.5]),   # (r, lr, step, batch_size, tts_pass2_input_prob_lst: x_y_both)
#                 (2,  5e-4, 30_000 * _tmp,  tts_batch_size, [0.3, 0.0, 0.7]),
#                 (2,  5e-4, 40_000 * _tmp,  tts_batch_size, [0.1, 0.0, 0.9]),
#                 (2,  1e-4, 80_000 * _tmp,  tts_batch_size, [0.1, 0.0, 0.9])]

# tts_init_weights_path = '/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_pretrainGold_bs32.tacotron/latest_weights.pyt' # initial weights, usually from a pretrained model
# tts_init_weights_path_pass2 = '/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_fixBestP1_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step20K_weights.pyt'
# # exp_id = f'mp_lj_pass2_fixBestP1_BS{tts_batch_size*tts_batch_acu}_stepD{tts_batch_acu}_max80k_betterSched_p1fr_frL{tts_fr_length_ratio}_re4_{tts_pass2_input_train}'


## no mask
tts_pass2_input_train = 'xAOy1s1'
tts_pass2_input_gen = 'xNy1s1'
tts_batch_size = 32 # 32 64 100
tts_batch_acu = 1
# exp_id = f'mp_lj_pass2_nomask_fixBestP1_BS{tts_batch_size*tts_batch_acu}_stepD{tts_batch_acu}_max80k_p1fr_frL{tts_fr_length_ratio}_re4_{tts_pass2_input_train}'

tts_batch_size = 32 # 32 64 100
tts_batch_acu = 4
exp_id = f'mp_lj_pass2_nomask_fixBestP1_BS{tts_batch_size*tts_batch_acu}_stepD{tts_batch_acu}_max80k_p1fr_frL{tts_fr_length_ratio}_re4_{tts_pass2_input_train}'

# tts_pass2_input_train = 'xAOy1'
# tts_pass2_input_gen = 'xNy1'
# tts_batch_size = 16 # 32 64 100
# tts_batch_acu = 2
# exp_id = f'mp_lj_pass2_nomask_fixBestP1_BS{tts_batch_size*tts_batch_acu}_stepD{tts_batch_acu}_max80k_p1fr_frL{tts_fr_length_ratio}_re4_{tts_pass2_input_train}'

# tts_pass2_input_train = 'xAOy1'
# tts_pass2_input_gen = 'xNy1'
# tts_batch_size = 16 # 32 64 100
# tts_batch_acu = 2
# tts_encoder_reduction_factor = 8
# tts_encoder_reduction_factor_s = tts_encoder_reduction_factor // 2 # quick fix
# exp_id = f'mp_lj_pass2_nomask_fixBestP1_BS{tts_batch_size*tts_batch_acu}_stepD{tts_batch_acu}_max80k_p1fr_frL{tts_fr_length_ratio}_re{tts_encoder_reduction_factor}_{tts_pass2_input_train}'

_tmp = 1 if not _tts_adjust_steps else tts_batch_acu
tts_schedule = [(2,  1e-3, 10_000 * _tmp,  tts_batch_size, [1, 0, 0]),   # progressive training schedule
                (2,  1e-3, 20_000 * _tmp,  tts_batch_size, [0.25, 0.25, 0.5]),   # (r, lr, step, batch_size, tts_pass2_input_prob_lst: x_y_both)
                (2,  5e-4, 30_000 * _tmp,  tts_batch_size, [0.05, 0.05, 0.9]),
                (2,  5e-4, 40_000 * _tmp,  tts_batch_size, [0.05, 0.05, 0.9]),
                (2,  1e-4, 80_000 * _tmp,  tts_batch_size, [0.05, 0.05, 0.9])]


tts_save_gv = False
# tts_save_gv = True




voc_model_id = exp_id + ''
tts_model_id = exp_id + ''

# set this to True if you are only interested in WaveRNN
ignore_tts = False
ignore_voc = True

# random seed
random_seed = 16


# DSP --------------------------------------------------------------------------------------------------------------#

# Settings for all models
sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 275                    # 12.5ms - in line with Tacotron 2 paper
win_length = 1100                   # 50ms - same reason as above
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False                   # Normalise to the peak of each wav file


# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#


# Model Hparams
voc_mode = 'MOL'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_upsample_factors = (5, 5, 11)   # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 32
voc_lr = 1e-4
voc_checkpoint_every = 25_000
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_total_steps = 50_000         # Total number of training steps
voc_test_samples = 50               # How many unseen samples to put aside for testing
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length
voc_clip_grad_norm = 4              # set to None if no gradient clipping needed
voc_init_weights_path = '/home/dawna/tts/qd212/models/WaveRNN/quick_start/voc_weights/latest_weights.pyt' # initial weights, usually from a pretrained model

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 11_000                 # target number of samples to be generated in each batch entry
voc_overlap = 550                   # number of samples for crossfading between batches


# TACOTRON/TTS -----------------------------------------------------------------------------------------------------#


# Model Hparams
tts_embed_dims = 256                # embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 128
tts_decoder_dims = 256
tts_postnet_dims = 128
tts_encoder_K = 16
tts_lstm_dims = 512
tts_postnet_K = 8
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ['english_cleaners']
tts_stop_threshold = -3.4           # Value below which audio generation ends.
                                    # For example, for a range of [-4, 4], this
                                    # will terminate the sequence at the first
                                    # frame that has all values < -3.4

# Training

# tts_schedule = [(2,  1e-3,  10_000 * _tmp,  tts_batch_size),   # progressive training schedule
#                 (2,  1e-3, 20_000 * _tmp,  tts_batch_size),   # (r, lr, step, batch_size)
#                 (2,  1e-4, 40_000 * _tmp,  tts_batch_size)]

tts_extension_dct = {}

tts_max_mel_len = 1250              # if you have a couple of extremely long spectrograms you might want to use this
tts_bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
tts_clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion - set to None if not needed
tts_checkpoint_every = 2_000 * tts_batch_acu       # checkpoints the model every X steps
# tts_init_weights_path = '/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_pretrainGold_bs32.tacotron/latest_weights.pyt' # initial weights, usually from a pretrained model
# tts_init_weights_path_pass2 = '/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_pretrainGold_bs32.tacotron/latest_weights.pyt'
# TODO: tts_phoneme_prob = 0.0              # [0 <-> 1] probability for feeding model phonemes vrs graphemes

tts_use_guided_attn_loss = False
tts_guided_attn_loss_sigma = 0.4
tts_guided_attn_loss_lambda = 1.0

mode = 'teacher_forcing' # overall training mode of the multipass system, inconsistent name kept for compatibility
tts_mode_train_pass1 = 'free_running'
tts_mode_train_pass2 = 'teacher_forcing'
tts_mode_gen_pass1 = 'free_running'
tts_mode_gen_pass2 = 'free_running'

# tts_pass2_input_gen = 'xNy1'

# Test
# test_sentences_file = 'test_sentences/sentences.txt'
# test_sentences_names = ['LJ001-0073', 'LJ010-0294', 'LJ020-0077', 'LJ030-0208', 'LJ040-0113']
test_sentences_file = 'test_sentences/sentences_espnet.txt'
test_sentences_names = ['LJ050-0029_gen', 'LJ050-0030_gen', 'LJ050-0031_gen', 'LJ050-0032_gen', 'LJ050-0033_gen']

if tts_save_gv:
    test_sentences_file = 'test_sentences/sentences_espnet_all250.txt'
    test_sentences_names = [f'LJ050-{29+i:04d}_gen' for i in range(250)]

# test_sentences_file = 'test_sentences/asup.txt'
# test_sentences_names = ['LJ050-0033_gen']


# ------------------------------------------------------------------------------------------------------------------#

