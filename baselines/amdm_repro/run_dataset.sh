#!/bin/bash
# OmiAD re-implementation, both stages (AMDM teacher 300 ep -> adversarial score distillation 300 ep)
# for one dataset tag with configs/{amdm,omiad}_<tag>.yaml. Run from the FlowAD repo root:
#   GPU=1 bash baselines/amdm_repro/run_dataset.sh brats     # tags: btad viaduct brats resc rsna
set -e
TAG=$1
export PATH=/home/shunsuke_sakai/G-Suite/miniforge/3/envs/flowad/bin:$PATH
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8   # several runs on one 192-core box oversubscribe the CPU otherwise
OUT=baselines/amdm_repro/out

torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=1 \
  baselines/amdm_repro/train_amdm.py --config_file baselines/amdm_repro/configs/amdm_${TAG}.yaml \
  > $OUT/train_amdm_${TAG}.log 2>&1

python baselines/amdm_repro/train_omiad.py --config_file baselines/amdm_repro/configs/omiad_${TAG}.yaml \
  > $OUT/train_omiad_${TAG}.log 2>&1
