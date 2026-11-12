#!/bin/bash
# OmiAD re-implementation on VIADUCT (multi-class, 49 categories), single GPU, bs 32.
# Same two stages and settings as run_btad.sh (300 epochs each). Run from the FlowAD repo root.
set -e
export PATH=/home/shunsuke_sakai/G-Suite/miniforge/3/envs/flowad/bin:$PATH
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=baselines/amdm_repro/out

torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=1 \
  baselines/amdm_repro/train_amdm.py --config_file baselines/amdm_repro/configs/amdm_viaduct.yaml \
  > $OUT/train_amdm_viaduct.log 2>&1

python baselines/amdm_repro/train_omiad.py --config_file baselines/amdm_repro/configs/omiad_viaduct.yaml \
  > $OUT/train_omiad_viaduct.log 2>&1
