#!/bin/bash
# Noise-level pilot: DiT-S on DINOv2-G features, plain FM, 50 epochs, one model per RealIAD category,
# then 1-step recon eval of ckpt_epoch_50 at inv_stop_time t in {0.1..0.9}.
# 8 GPU workers pull categories from a shared queue; each trains its category then runs the 9 evals.
# Finished steps (ckpt / eval_results.yaml) are skipped, so the script can be re-run to resume.
#   python scripts/gen_realiad_dinoG_configs.py pilot && bash scripts/train_realiad_ditS_dinoG_pilot.sh
cd "$(dirname "$0")/.."
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH HF_HUB_OFFLINE=1
CFG_DIR=configs/train/realiad_ditS_dinoG_pilot
EVAL_DIR=configs/eval/realiad_ditS_dinoG_pilot_sweep
LOG_ROOT=logs/realiad_ditS_dinoG_pilot
NGPU=${NGPU:-8}
mkdir -p $LOG_ROOT/_out
QUEUE=$LOG_ROOT/_out/queue.txt
ls $CFG_DIR | sed 's/.yaml$//' > $QUEUE

worker() {
  local gpu=$1 port=$((12600 + $1))
  while true; do
    local cat
    cat=$(flock $QUEUE.lock bash -c "head -n1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$cat" ] && break
    if [ ! -f $LOG_ROOT/$cat/checkpoints/ckpt_epoch_50.pth ]; then
      echo "[gpu$gpu] train $cat $(date '+%F %T')"
      CUDA_VISIBLE_DEVICES=$gpu torchrun --nnodes=1 --nproc-per-node=1 --master_port=$port \
        main.py --config_file $CFG_DIR/$cat.yaml --task train > $LOG_ROOT/_out/$cat.out 2>&1
      [ -f $LOG_ROOT/$cat/checkpoints/ckpt_epoch_50.pth ] || { echo "[gpu$gpu] train FAILED $cat"; continue; }
    fi
    for cfg in $EVAL_DIR/${cat}_t*.yaml; do
      local d
      d=$(grep "^  log_dir:" $cfg | awk '{print $2}')
      [ -f $d/eval_results.yaml ] && continue
      mkdir -p $d
      CUDA_VISIBLE_DEVICES=$gpu torchrun --nnodes=1 --nproc-per-node=1 --master_port=$port \
        main.py --config_file $cfg --task eval > ${d%/}.log 2>&1
    done
    echo "[gpu$gpu] done $cat $(date '+%F %T')"
  done
}

for g in $(seq 0 $((NGPU-1))); do worker $g & sleep 5; done
wait
echo "all done"
