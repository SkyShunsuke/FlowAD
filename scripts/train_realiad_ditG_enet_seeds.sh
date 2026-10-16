#!/bin/bash
# DiT-G + EfficientNet-B4, logistic-t FM, 1-step recon from t=0.1 on RealIAD, one model per category, extra seeds.
#   python scripts/gen_realiad_ditG_enet_seed_configs.py 1 2 3 4 && bash scripts/train_realiad_ditG_enet_seeds.sh [seed ...]
# Same layout as seed 0: each job uses all 8 GPUs x batch 8 (1 GPU x 64 diverges to NaN under bf16).
# Eval every 25 epochs, early stopping (patience 50); best epoch/metrics in best_results.json (no ckpt, home quota).
# Jobs run one after another, seed-major; finished jobs (logs/.../seed<s>/<cat>/DONE) are skipped, so re-run to resume.
cd "$(dirname "$0")/.."
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH HF_HUB_OFFLINE=1
CFG_DIR=configs/train/realiad_ditG_enet_seeds
LOG_ROOT=logs/realiad_ditG_enet_seeds
SEEDS=${@:-$(cd $CFG_DIR && ls -d seed* | sed 's/seed//' | sort -n)}
mkdir -p $LOG_ROOT/_out

for s in $SEEDS; do
  for cfg in $CFG_DIR/seed$s/*.yaml; do
    job=seed$s/$(basename $cfg .yaml)
    [ -f $LOG_ROOT/$job/DONE ] && continue
    mkdir -p $LOG_ROOT/_out/seed$s
    echo "start $job $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc-per-node=8 --master_port=13000 \
      main.py --config_file $cfg --task train > $LOG_ROOT/_out/$job.out 2>&1
    rc=$?
    [ $rc -eq 0 ] && [ -s $LOG_ROOT/$job/best_results.json ] && touch $LOG_ROOT/$job/DONE
    echo "end $job rc=$rc $(date '+%F %T')"
  done
done
echo "all done"
