#!/bin/bash
# Multi-seed DiT-B + EfficientNet-B4 FM on MVTec AD / VisA / MPDD (model-scale ablation of ditG_enet_fm_fr16_tf32_bs8_p1_es10).
# 1 GPU x bs8 per run, no DDP. DiT-B is small, so all 15 runs go at once (2 per GPU on 8 GPUs). Finished runs (DONE) are skipped.
#   python scripts/gen_ditB_enet_fm_seed_configs.py && bash scripts/train_ditB_enet_fm_seeds.sh
cd "$(dirname "$0")/.."
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH HF_HUB_OFFLINE=1
TAG=ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds
CFG_DIR=configs/train/$TAG
T=logs/$TAG
NGPU=${NGPU:-8}
mkdir -p $T/_out
run() { local gpu=$1 job=$2
  [ -f $T/$job/DONE ] && return
  rm -rf $T/$job  # no checkpoints kept, so an unfinished run restarts from scratch
  echo "[gpu$gpu] start $job $(date '+%F %T')" >> $T/launcher.log
  CUDA_VISIBLE_DEVICES=$gpu python main.py --config_file $CFG_DIR/$job.yaml --task train > $T/_out/$job.out 2>&1
  local rc=$?; [ $rc -eq 0 ] && [ -s $T/$job/best_results.json ] && touch $T/$job/DONE
  echo "[gpu$gpu] end $job rc=$rc $(date '+%F %T')" >> $T/launcher.log; }
i=0
for ds in visa ad mpdd; do for s in 0 1 2 3 4; do
  run $((i % NGPU)) ${ds}_s$s & i=$((i + 1)); sleep 3
done; done
wait
echo "all done $(date '+%F %T')" >> $T/launcher.log
