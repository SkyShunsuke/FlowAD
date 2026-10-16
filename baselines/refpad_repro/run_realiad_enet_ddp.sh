#!/bin/bash
# ReFP-AD (EfficientNet-B4) on RealIAD — 8-GPU DDP training + sharded eval.
# Assumes the dataset layout and token caches already exist (phases 1-2 of
# run_realiad_enet.sh); it is safe to re-run those first, they are idempotent.
# Kill any running single-GPU training before launching this (same recipe,
# separate out_dir, so the single-GPU run is not clobbered either way).
set -e
cd "$(dirname "$0")"
PY=~/projects/FlowAD/flowad/bin/python
ROOT=/home/acg17237wy/projects/FlowAD
DATA=$ROOT/data/realiad_1cls
CACHE=$ROOT/baselines/refpad_repro/cache_tokens_enet
OUT=$ROOT/baselines/refpad_repro/out/realiad_unified_enet
mkdir -p logs

TORCHRUN=$(dirname "$PY")/torchrun
nohup "$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 \
  run_repro_enet_realiad_ddp.py \
  --data_path "$DATA" --out_dir "$OUT" --cache_dir "$CACHE" \
  > logs/full_realiad_enet_ddp.log 2>&1 &
echo "Launched DDP training. tail -f logs/full_realiad_enet_ddp.log"
echo "(non-rank-0 output: $OUT/rank*.log)"
