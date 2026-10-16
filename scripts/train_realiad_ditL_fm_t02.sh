#!/bin/bash
# DiT-L, plain FM (uniform t), one-step recon from t=0.2 on RealIAD, one model per category.
# Eval every 25 epochs, early stopping (patience 50), best ckpt saved.
# 8 GPU workers pull categories from a shared queue (1 GPU / category, global batch 64).
# Finished categories (logs/.../<cat>/DONE) are skipped, so the script can be re-run to resume.
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH
CFG_DIR=configs/train/realiad_ditL_fm_t02
LOG_ROOT=logs/realiad_ditL_enet_fm_t02
NGPU=${NGPU:-8}
mkdir -p $LOG_ROOT/_out
QUEUE=$LOG_ROOT/_out/queue.txt
ls $CFG_DIR | sed 's/.yaml$//' > $QUEUE

worker() {
  local gpu=$1
  while true; do
    local cat
    cat=$(flock $QUEUE.lock bash -c "head -n1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$cat" ] && break
    [ -f $LOG_ROOT/$cat/DONE ] && continue
    echo "[gpu$gpu] start $cat $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc-per-node=1 \
      main.py --config_file $CFG_DIR/$cat.yaml --task train > $LOG_ROOT/_out/$cat.out 2>&1
    local rc=$?
    [ $rc -eq 0 ] && touch $LOG_ROOT/$cat/DONE
    echo "[gpu$gpu] end $cat rc=$rc $(date '+%F %T')"
  done
}

for g in $(seq 0 $((NGPU-1))); do worker $g & sleep 5; done
wait
echo "all done"
