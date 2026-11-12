#!/bin/bash
# OmiAD re-implementation on BTAD (multi-class, 3 categories), single GPU, bs 32.
#   stage 1: AMDM teacher, 300 epochs            (train_amdm.py,  configs/amdm_btad.yaml)
#   stage 2: adversarial score distillation, 300 epochs -> one-step generator
#                                                 (train_omiad.py, configs/omiad_btad.yaml)
# Run from the FlowAD repo root. data/btad -> extracted btad.zip (bottle / glass / wood, PNG copies).
# baselines/amdm_repro/out is a symlink to /var/tmp/shunsuke_sakai/amdm_out (home quota).
set -e
export PATH=/home/shunsuke_sakai/G-Suite/miniforge/3/envs/flowad/bin:$PATH
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=baselines/amdm_repro/out

torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=1 \
  baselines/amdm_repro/train_amdm.py --config_file baselines/amdm_repro/configs/amdm_btad.yaml \
  2>&1 | tee $OUT/train_amdm_btad.log

python baselines/amdm_repro/train_omiad.py --config_file baselines/amdm_repro/configs/omiad_btad.yaml \
  2>&1 | tee $OUT/train_omiad_btad.log
