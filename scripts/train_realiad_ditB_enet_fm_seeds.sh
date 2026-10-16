#!/bin/bash
# RealIAD DiT-B + EfficientNet-B4 FM, one model per category, multi-seed (same setting as the MVTec AD / VisA / MPDD DiT-B runs).
#   python scripts/gen_realiad_ditB_enet_fm_seed_configs.py && bash scripts/train_realiad_ditB_enet_fm_seeds.sh [seed ...]
# NGPU x SLOTS workers pull category runs from a shared seed-major queue (1 GPU x bs8 each, no DDP).
# Finished runs (DONE) are skipped; unfinished ones restart from scratch (no checkpoints), so re-run to resume.
cd "$(dirname "$0")/.."
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH HF_HUB_OFFLINE=1
TAG=realiad_ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds
CFG_DIR=configs/train/$TAG
T=logs/$TAG
NGPU=${NGPU:-8}; SLOTS=${SLOTS:-2}
SEEDS=${@:-$(cd $CFG_DIR && ls -d seed* | sed 's/seed//' | sort -n)}
mkdir -p $T/_out
QUEUE=$T/_out/queue.txt
(for s in $SEEDS; do for cfg in $CFG_DIR/seed$s/*.yaml; do echo seed$s/$(basename $cfg .yaml); done; done) > $QUEUE

worker() {
  local gpu=$1
  while true; do
    local job
    job=$(flock $QUEUE.lock bash -c "head -n1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$job" ] && break
    [ -f $T/$job/DONE ] && continue
    rm -rf $T/$job; mkdir -p $T/_out/$(dirname $job)
    echo "[gpu$gpu] start $job $(date '+%F %T')" >> $T/launcher.log
    CUDA_VISIBLE_DEVICES=$gpu python main.py --config_file $CFG_DIR/$job.yaml --task train > $T/_out/$job.out 2>&1
    local rc=$?; [ $rc -eq 0 ] && [ -s $T/$job/best_results.json ] && touch $T/$job/DONE
    echo "[gpu$gpu] end $job rc=$rc $(date '+%F %T')" >> $T/launcher.log
  done
}

for k in $(seq 1 $SLOTS); do for g in $(seq 0 $((NGPU - 1))); do worker $g & sleep 3; done; done
wait
echo "all done $(date '+%F %T')" >> $T/launcher.log
