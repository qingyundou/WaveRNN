### env
unset LD_PRELOAD
export PATH=/home/dawna/tts/qd212/anaconda2/bin/:$PATH
# source activate p37_pt11_c9_tts
source activate p37_pt13_c10_tts

# export PATH=/home/miproj/4thyr.oct2020/zs323/miniconda3/bin/:$PATH
# source activate env1
# export PYTHONBIN=/home/miproj/4thyr.oct2020/zs323/miniconda3/envs/env1/bin/python3
# # source activate env3.7
# # export PYTHONBIN=/home/miproj/4thyr.oct2020/zs323/miniconda3/envs/env3.7/bin/python3

### gpu
AIR_FORCE_GPU=1
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
# ------------------------------------------------- process data
hp_file=scripts/hparams_200hz.py
# python preprocess.py --hp_file $hp_file

# ------------------------------------------------- tf

## default setting: Taco + WRNN
voc_weights_gold=/home/dawna/tts/qd212/models/WaveRNN/quick_start/voc_weights/latest_weights.pyt
voc_weights_tfnv100hz=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_pretrainGold.wavernn/wave_step450K_weights.pyt
voc_weights_afnv100hz=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_af_online_kl1.0_bs100_stepD2.wavernn/wave_step500K_weights.pyt
voc_weights_asnv200hz=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_asnv_bs100.wavernn/wave_step975K_weights.pyt
voc_weights_tfnv200hz=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_bs100_stepD5.wavernn/wave_step500K_weights.pyt
voc_weights_afnv200hz=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_af_bs100_stepD2_initAF100hz_noDropout.wavernn/wave_step225K_weights.pyt

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
# voc_weights=${EXP_DIR}/checkpoints/lj_pretrainGold.wavernn/wave_step450K_weights.pyt
# python train_tacotron.py --hp_file $hp_file
# python train_wavernn.py --hp_file $hp_file --gta
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights --unbatched # -i "THAT IS REFLECTED IN DEFINITE AND COMPREHENSIVE OPERATING PROCEDURES."
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights
# python gen_wavernn.py --hp_file $hp_file -s 3 --unbatched --gta
# python gen_wavernn.py --hp_file $hp_file -s 3 --gta

# gold init / tf big BS
hp_file=scripts/hparams_initGold_tuneBS.py
# python train_tacotron.py --hp_file $hp_file
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --batched --use_standard_names
# voc_weights=${EXP_DIR}/checkpoints/lj_pretrainGold.wavernn/wave_step450K_weights.pyt
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights --batched
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_tfnv100hz --batched
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
# python train_tacotron.py --hp_file $hp_file --force_af_online
# python train_wavernn.py --hp_file $hp_file --gta
# python train_wavernn.py --hp_file $hp_file
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --batched --use_standard_names
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_afnv100hz --batched
# python gen_tacotron.py --hp_file $hp_file --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched --use_standard_names
# python gen_wavernn.py --hp_file $hp_file -s 3 --gta

hp_file=scripts/hparams_100hz_afnv.py
# python train_wavernn.py --hp_file $hp_file --gta

# ------------------------------------------------- ss

hp_file=scripts/hparams_ss_initGold_tuneBS.py
# python train_tacotron.py --hp_file $hp_file
python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --batched --use_standard_names
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron.py --hp_file $hp_file --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched


# ------------------------------------------------- 200hz
hp_file=scripts/hparams_200hz_asnv.py
# python train_wavernn.py --hp_file $hp_file
# file=/home/dawna/tts/qd212/models/WaveRNN/data-200hz/gta_lj_v250t250_200hz_af_bs100_stepD2_initAF100hz/LJ050-0029.npy
# python gen_wavernn.py --hp_file $hp_file --custom_files --voc_weights $voc_weights_asnv200hz

hp_file=scripts/hparams_200hz_tf.py
# python train_tacotron.py --hp_file $hp_file
# python train_tacotron.py --hp_file $hp_file --force_gta
# python gen_tacotron.py --hp_file $hp_file --save_attention --save_mel --skip_wav --use_standard_names griffinlim
# python gen_tacotron.py --hp_file $hp_file --save_attention --save_mel griffinlim
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_tfnv200hz --batched --use_standard_names
# tts_weights=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_bs100_stepD5.tacotron/taco_step270K_weights.pyt
# python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights wavernn --voc_weights $voc_weights_asnv200hz --batched
# python gen_tacotron_tf.py --hp_file $hp_file --tts_weights $tts_weights wavernn --voc_weights $voc_weights_asnv200hz --batched
# tts_weights=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_bs100_stepD5.tacotron/taco_step400K_weights.pyt
# python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights wavernn --voc_weights $voc_weights_asnv200hz --batched --use_standard_names
# python train_wavernn.py --hp_file $hp_file --gta

hp_file=scripts/hparams_200hz_af.py
# python train_tacotron.py --hp_file $hp_file
# python train_tacotron.py --hp_file $hp_file --force_af_online
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_asnv200hz --batched
tts_weights=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_af_bs100_stepD2_initAF100hz.tacotron/taco_step160K_weights.pyt
# python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights wavernn --voc_weights $voc_weights_asnv200hz --batched --use_standard_names
# voc_weights=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_af_bs100_stepD2_initAF100hz.wavernn/wave_step450K_weights.pyt
# voc_weights=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_af_bs100_stepD2_initAF100hz_tuneNV.wavernn/wave_step375K_weights.pyt
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights --batched
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights --batched --use_standard_names
# python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights wavernn --voc_weights $voc_weights_tfnv200hz --batched --use_standard_names
# python train_wavernn.py --hp_file $hp_file --gta

hp_file=scripts/hparams_200hz_af_v2.py
# python train_wavernn.py --hp_file $hp_file --gta

hp_file=scripts/hparams_200hz_af_v3.py
# python train_wavernn.py --hp_file $hp_file --gta

hp_file=scripts/hparams_200hz_af_dataAug.py
# python train_wavernn.py --hp_file $hp_file --gtaNnat

hp_file=scripts/hparams_200hz_af_dataAug_v2.py
# python train_wavernn.py --hp_file $hp_file --gtaNnat

hp_file=scripts/hparams_200hz_af_noDropout.py
# python train_tacotron.py --hp_file $hp_file
# python train_tacotron.py --hp_file $hp_file --force_af_online
# python gen_tacotron.py --hp_file $hp_file wavernn --voc_weights $voc_weights_asnv200hz --batched
tts_weights=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/lj_v250t250_200hz_af_bs100_stepD2_initAF100hz_noDropout.tacotron/taco_step160K_weights.pyt
# python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights wavernn --voc_weights $voc_weights_tfnv200hz --batched --use_standard_names
# python train_wavernn.py --hp_file $hp_file --gta
# python train_wavernn.py --hp_file $hp_file --gta --grab_memory --gpu_id $CUDA_VISIBLE_DEVICES


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
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_BS128_stepD4_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step320K_weights.pyt
tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_BS32_stepD2_max80k_p1fr_frL1.2_re8_xAOy1.tacotron/pass2/taco_step40K_weights.pyt
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_pass2_init_gal.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav wavernn --voc_weights $voc_weights_gold --batched
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_fixBestP1_GAL100.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step12K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_fixBestP1_GAL500.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step12K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_GAL1.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step24K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_GAL10.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step12K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_GAL100.0_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step4K_weights.pyt
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_pass2_nomask_fixBestP1_GAL10.0schedDiag_BS32_stepD2_max80k_p1fr_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step24K_weights.pyt
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_gv --skip_wav --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_delib_pass2.py
# python train_tacotron_pass2.py --hp_file $hp_file
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_delib_shareEnc_init_pass2_BS32_stepD2_max80k_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step100K_weights.pyt
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_delib_pass2_xAOs1.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_delib_pass2_xAOs1_attn.py
# python train_tacotron_pass2.py --hp_file $hp_file

hp_file=scripts/hparams_delib_pass2_xAOs1_attnAdv.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_delib_pass2_xAOs1_attnAdv_smartKV.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention wavernn --voc_weights $voc_weights_gold --batched
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_attnAdv_smartKV_pass2_BS32_stepD2_max80k_frL1.2_re8N16_xAOs1.tacotron/pass2/taco_step20K_weights.pyt
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_delib_pass2_xAOs1_attnAdv_smartKV_trainByGroup.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention wavernn --voc_weights $voc_weights_gold --batched


hp_file=scripts/hparams_delib_pass2_xAOs1_attn_gal.py
# python train_tacotron_pass2.py --hp_file $hp_file
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_attn_GAL1.0sched_pass2_BS32_stepD2_max80k_frL1.2_re8_xAOs1.tacotron/pass2/taco_step20K_weights.pyt
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched

hp_file=scripts/hparams_delib_pass2_gal.py
# python train_tacotron_pass2.py --hp_file $hp_file
# python gen_tacotron_multipass.py --hp_file $hp_file --use_standard_names --save_attention wavernn --voc_weights $voc_weights_gold --batched
# tts_weights_pass2=/home/dawna/tts/qd212/models/WaveRNN/checkpoints/mp_lj_delib_shareEnc_pass2_GAL10.0sched_BS32_stepD2_max80k_frL1.2_re4_xAOy1s1.tacotron/pass2/taco_step8K_weights.pyt
# python gen_tacotron_multipass.py --hp_file $hp_file --save_attention --tts_weights_pass2 $tts_weights_pass2 wavernn --voc_weights $voc_weights_gold --batched


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
