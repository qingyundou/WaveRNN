### env
unset LD_PRELOAD
# source activate p37_pt11_c9_tts
source activate p37_pt13_c10_tts

### gpu
AIR_FORCE_GPU=0
export MANU_CUDA_DEVICE=0 # 2,3 note on nausicaa no.2 is no.0
# select gpu when not on air
if [[ "$HOSTNAME" != *"air"* ]]  || [ $AIR_FORCE_GPU -eq 1 ]; then
  X_SGE_CUDA_DEVICE=$MANU_CUDA_DEVICE
  echo "manually set gpu $MANU_CUDA_DEVICE"
fi
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo "on $HOSTNAME, using gpu (no nb means cpu) $CUDA_VISIBLE_DEVICES"

### dir
EXP_DIR=/home/dawna/tts/qd212/models/WaveRNN
cd $EXP_DIR

### exp
# ------------------------------------------------- tf

## default setting: Taco + WRNN
voc_weights_gold=/home/dawna/tts/qd212/models/WaveRNN/quick_start/voc_weights/latest_weights.pyt

## default: Taco + NV
# voc_weights=/home/dawna/tts/qd212/models/WaveRNN/quick_start/voc_weights/latest_weights.pyt
# voc_weights=${EXP_DIR}/checkpoints/lj_pretrain.wavernn/latest_weights.pyt
# python train_tacotron.py
# python train_wavernn.py --gta
# python gen_tacotron.py griffinlim
# python gen_tacotron.py wavernn --voc_weights $voc_weights
# python gen_wavernn.py -s 3 # --gta

## ASNV
# python train_wavernn.py --hp_file scripts/hparams_asnv.py
# python gen_wavernn.py --hp_file scripts/hparams_asnv.py -s 3 # --gta

## pretrain / init
hp_file=scripts/hparams_init.py
# python train_wavernn.py --hp_file $hp_file --gta
# python gen_wavernn.py --hp_file $hp_file -s 3 --unbatched --gta

# gold init
hp_file=scripts/hparams_initGold.py
# voc_weights=${EXP_DIR}/checkpoints/ljspeech_mol.wavernn/wave_step1000K_weights.pyt
# voc_weights=${EXP_DIR}/checkpoints/lj_pretrainGold.wavernn/wave_step50K_weights.pyt
# python train_tacotron.py --hp_file $hp_file
# python train_wavernn.py --hp_file $hp_file --gta
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights --unbatched # -i "THAT IS REFLECTED IN DEFINITE AND COMPREHENSIVE OPERATING PROCEDURES."
# python gen_wavernn.py --hp_file $hp_file -s 3 --unbatched --gta

# gold init / tf big BS
hp_file=scripts/hparams_initGold_tuneBS.py
# python train_tacotron.py --hp_file $hp_file
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --batched --use_standard_names
# python gen_tacotron.py --hp_file $hp_file --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched --use_standard_names



## debug
hp_file=scripts/hparams_debug.py
# python train_tacotron.py --hp_file $hp_file
# python train_wavernn.py --hp_file $hp_file --gta
# python train_wavernn.py --hp_file $hp_file

# tts_weights=/home/dawna/tts/qd212/models/WaveRNN/quick_start/tts_weights/latest_weights.pyt
# voc_weights=/home/dawna/tts/qd212/models/WaveRNN/quick_start/voc_weights/latest_weights.pyt
# voc_weights=${EXP_DIR}/checkpoints/lj_pretrain.wavernn/latest_weights.pyt
# voc_weights=${EXP_DIR}/checkpoints/ljspeech_mol.wavernn/wave_step1000K_weights.pyt
# python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights griffinlim
# python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights wavernn --voc_weights $voc_weights
# python gen_wavernn.py --hp_file $hp_file -s 3 --voc_weights $voc_weights # --gta


## gold
hp_file=scripts/hparams_gold.py
# python train_tacotron.py --hp_file $hp_file --force_attn

# ------------------------------------------------- af

## gold init, af offline
hp_file=scripts/hparams_af_offline.py
# python train_tacotron.py --hp_file $hp_file
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold # --unbatched

## gold init, af online
hp_file=scripts/hparams_initGold_af.py
# python train_tacotron.py --hp_file $hp_file
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --unbatched --use_standard_names

## gold init, af online, kl
hp_file=scripts/hparams_af_online_kl.py
# python train_tacotron.py --hp_file $hp_file
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --unbatched --use_standard_names

# tune gamma
hp_file=scripts/hparams_af_online_kl_tune.py
# python train_tacotron.py --hp_file $hp_file
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --batched --use_standard_names

# tune batch size and lr
hp_file=scripts/hparams_af_online_tuneBS.py
# python train_tacotron.py --hp_file $hp_file
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --batched --use_standard_names
# python gen_tacotron.py --hp_file $hp_file --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched --use_standard_names



# ------------------------------------------------- multipass
# -------------------------------------------------

# pretrain tacotron_pass2
hp_file=scripts/hparams_pass2.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention griffinlim

hp_file=scripts/hparams_pass2_init.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched
tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_fixBestP1_BS100_stepD10_p1fr_re4_y1.tacotron/pass2/taco_step180K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_fixBestP1_BS100_stepD5_p1fr_re4_xAOy1s1.tacotron/pass2/taco_step160K_weights.pyt
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_pass2_init_dev.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention wavernn --voc_weights $voc_weights_gold --batched
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_BS32_stepD1_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step54K_weights.pyt
tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_BS128_stepD4_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step184K_weights.pyt
python gen_tacotron_multipass.py --hp_file $hp_file --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_pass2_init_gal.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_fixBestP1_GAL100.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step12K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_fixBestP1_GAL500.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step12K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_GAL1.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step24K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_GAL10.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step56K_weights.pyt
tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_GAL100.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step24K_weights.pyt
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched


hp_file=scripts/hparams_multipass.py
# python train_tacotron_multipass.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention --save_mel wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --batched --use_standard_names

hp_file=scripts/hparams_multipass_af.py
# python train_tacotron_multipass.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention --save_mel wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_multipass_af_sched.py
# python train_tacotron_multipass.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention --save_mel wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_multipass_af_tune.py
# python train_tacotron_multipass.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention --save_mel wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_multipass_switch.py
# python train_tacotron_multipass.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention --save_mel wavernn --voc_weights $voc_weights_gold --batched
