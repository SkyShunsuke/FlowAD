#!/bin/bash
# Waits for the VisA extension to finish, then fine-tunes MVTec from the
# pre-collapse best checkpoint (ep250) at lr 1e-4 on GPUs 0,1,2,6.
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate flowad

pat="amdm_visa""_ext.yaml"   # split so this script doesn't match itself
while pgrep -f "$pat" > /dev/null; do sleep 300; done
sleep 60   # let the final eval / teardown finish

export CUDA_VISIBLE_DEVICES=0,1,2,6
cd "$(dirname "$0")/../.."   # repo root
mkdir -p logs baselines/amdm_repro/out

torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=4 \
  baselines/amdm_repro/train_amdm.py \
  --config_file baselines/amdm_repro/configs/amdm_mvtec_ft.yaml \
  2>&1 | tee baselines/amdm_repro/out/train_mvtec_ft.log
