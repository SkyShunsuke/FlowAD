#!/bin/bash
# AMDM: MVTec-AD (fresh, clip+EMA) then VisA extension (resume ep1000 -> 2000),
# both 4-GPU DDP (4 x bs8 = global 32, paper batch). Run from the repo root.
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate flowad
export CUDA_VISIBLE_DEVICES=0,1,2,6
mkdir -p logs baselines/amdm_repro/out

torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=4 \
  baselines/amdm_repro/train_amdm.py \
  --config_file baselines/amdm_repro/configs/amdm_mvtec.yaml \
  2>&1 | tee baselines/amdm_repro/out/train_mvtec.log

torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=4 \
  baselines/amdm_repro/train_amdm.py \
  --config_file baselines/amdm_repro/configs/amdm_visa_ext.yaml \
  2>&1 | tee baselines/amdm_repro/out/train_visa_ext.log
