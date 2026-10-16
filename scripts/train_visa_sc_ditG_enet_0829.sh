#!/bin/bash
# Single-class DiT-G + EfficientNet-B4 FM on VisA (0829 setting), one category per GPU, no DDP (plain python, 1 GPU x 8).
#   python scripts/gen_visa_sc_ditG_enet_0829_configs.py && bash scripts/train_visa_sc_ditG_enet_0829.sh
# NGPU workers pull categories from a shared queue (largest train sets first). Finished runs (DONE) are skipped, so re-running resumes.
cd "$(dirname "$0")/.."
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH HF_HUB_OFFLINE=1
TAG=visa_sc_ditG_enet_0829
CFG_DIR=configs/train/$TAG
LOG_ROOT=logs/$TAG
NGPU=${NGPU:-8}
GPU0=${GPU0:-0}
mkdir -p $LOG_ROOT/_out
QUEUE=$LOG_ROOT/_out/queue.txt
printf '%s\n' pcb3 pcb4 pcb1 pcb2 candle macaroni1 macaroni2 capsules chewinggum cashew fryum pipe_fryum > $QUEUE

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
