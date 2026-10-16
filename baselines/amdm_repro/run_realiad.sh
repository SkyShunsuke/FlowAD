#!/bin/bash
# AMDM reproduction on RealIAD multi-class: 8-GPU DDP (per-GPU batch 4 =
# paper's global batch 32, so the optimization recipe matches the single-GPU
# runs). Protocol: NO intermediate eval — train all 1000 epochs, then ONE
# final evaluation (EMA primary + raw secondary) of the last checkpoint.
# Results: <log_dir>/eval_results_final.csv (EMA) / eval_results_final_raw.csv
# Run from the FlowAD repo root.
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p logs baselines/amdm_repro/out

torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=8 \
  baselines/amdm_repro/train_amdm.py \
  --config_file baselines/amdm_repro/configs/amdm_realiad.yaml \
  2>&1 | tee baselines/amdm_repro/out/train_realiad.log

# To re-run only the final evaluation later (e.g. on a specific snapshot):
#   torchrun ... train_amdm.py --config_file .../amdm_realiad.yaml \
#     --eval_only --eval_ckpt baselines/amdm_repro/out/amdm_mc_unet_enet_realiad_0922/checkpoints/ckpt_epoch_800.pth
