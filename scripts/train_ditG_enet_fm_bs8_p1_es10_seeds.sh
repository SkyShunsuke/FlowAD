#!/bin/bash
# DiT-G counterparts of the DiT-B multi-seed MVTec AD / VisA runs (same setting: 1 GPU x bs8, patch 1, fr16, fp32 + TF32,
# eval every epoch, early-stop patience 10). 10 runs on 8 GPUs: VisA (longest) alone on GPUs 0-4, AD on 5-7 (s3/s4 share 5/6).
#   python scripts/gen_ditG_enet_fm_bs8_p1_es10_configs.py && bash scripts/train_ditG_enet_fm_bs8_p1_es10_seeds.sh
# Finished runs (DONE) are skipped; unfinished ones restart from scratch (no checkpoints kept).
cd "$(dirname "$0")/.."
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH HF_HUB_OFFLINE=1
TAG=ditG_enet_fm_fr16_tf32_bs8_p1_es10_seeds
CFG_DIR=configs/train/$TAG
T=logs/$TAG
mkdir -p $T/_out
run() { local gpu=$1 job=$2
  [ -f $T/$job/DONE ] && return
  rm -rf $T/$job
  echo "[gpu$gpu] start $job $(date '+%F %T')" >> $T/launcher.log
  CUDA_VISIBLE_DEVICES=$gpu python main.py --config_file $CFG_DIR/$job.yaml --task train > $T/_out/$job.out 2>&1
  local rc=$?; [ $rc -eq 0 ] && [ -s $T/$job/best_results.json ] && touch $T/$job/DONE
  echo "[gpu$gpu] end $job rc=$rc $(date '+%F %T')" >> $T/launcher.log; }
JOBS=(visa_s0 visa_s1 visa_s2 visa_s3 visa_s4 ad_s0 ad_s1 ad_s2 ad_s3 ad_s4)
GPUS=(0 1 2 3 4 5 6 7 5 6)
for i in "${!JOBS[@]}"; do run ${GPUS[$i]} ${JOBS[$i]} & sleep 3; done
wait
echo "all done $(date '+%F %T')" >> $T/launcher.log
