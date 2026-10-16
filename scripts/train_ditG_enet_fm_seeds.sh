#!/bin/bash
# Multi-seed DiT-G + EfficientNet-B4 FM on MVTec AD / VisA / MPDD, one run per GPU, no DDP (plain python, 1 GPU x 64).
#   python scripts/gen_ditG_enet_fm_seed_configs.py 16 && FR=16 bash scripts/train_ditG_enet_fm_seeds.sh
# NGPU workers pull runs from a shared queue (longest datasets first). Finished runs (DONE) are skipped, so re-running resumes.
cd "$(dirname "$0")/.."
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH HF_HUB_OFFLINE=1
FR=${FR:-16}
TAG=ditG_enet_fm_fr${FR}_tf32_seeds
CFG_DIR=configs/train/$TAG
LOG_ROOT=logs/$TAG
NGPU=${NGPU:-8}
GPU0=${GPU0:-0}
mkdir -p $LOG_ROOT/_out
QUEUE=$LOG_ROOT/_out/queue.txt
# longest first (VisA -> AD -> MPDD) so the tail of the queue is short
(for ds in visa ad mpdd; do for s in 0 1 2 3 4; do [ -f $CFG_DIR/${ds}_s$s.yaml ] && echo ${ds}_s$s; done; done) > $QUEUE

worker() {
  local gpu=$1
  while true; do
    local job
    job=$(flock $QUEUE.lock bash -c "head -n1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$job" ] && break
    [ -f $LOG_ROOT/$job/DONE ] && continue
    rm -rf $LOG_ROOT/$job  # no checkpoints kept, so an unfinished run restarts from scratch
    echo "[gpu$gpu] start $job $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES=$gpu python main.py --config_file $CFG_DIR/$job.yaml --task train > $LOG_ROOT/_out/$job.out 2>&1
    local rc=$?
    [ $rc -eq 0 ] && [ -s $LOG_ROOT/$job/best_results.json ] && touch $LOG_ROOT/$job/DONE
    echo "[gpu$gpu] end $job rc=$rc $(date '+%F %T')"
  done
}

for g in $(seq $GPU0 $((GPU0 + NGPU - 1))); do worker $g & sleep 5; done
wait
echo "all done"
