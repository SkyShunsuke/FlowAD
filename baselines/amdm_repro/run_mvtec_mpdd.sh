#!/bin/bash
# AMDM reproduction, MVTec-AD then MPDD multi-class: single GPU, bs 32
# (exactly the paper's single-GPU setting). Run from the FlowAD repo root.
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate flowad
export CUDA_VISIBLE_DEVICES=7
mkdir -p logs baselines/amdm_repro/out

for ds in mvtec mpdd; do
  echo "=== AMDM ${ds} ==="
  torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=1 \
    baselines/amdm_repro/train_amdm.py \
    --config_file baselines/amdm_repro/configs/amdm_${ds}.yaml \
    2>&1 | tee baselines/amdm_repro/out/train_${ds}.log
done
