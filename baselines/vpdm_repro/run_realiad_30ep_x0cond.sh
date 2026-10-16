#!/bin/bash
# VPDM + input-feature path (jitter), seed 2 (best MVTec variant) on multi-class Real-IAD.
# 30 epochs = ~34k iterations, same as the 300-epoch MVTec runs. Trains on the first
# visible GPU; the final evaluation splits over all visible GPUs. Run from the repo root.
PY=~/venv/jit/bin/python
CFG=baselines/vpdm_repro/configs/vpdm_realiad_30ep_x0cond.yaml
OUT=baselines/vpdm_repro/out
CUDA_VISIBLE_DEVICES=${GPUS:-3,4,5,6,7} $PY baselines/vpdm_repro/train_vpdm.py --config_file $CFG \
  --override x0_cond.mode=jitter opt.global_seed=2 \
  > $OUT/vpdm30x_realiad_jitter_s2.log 2>&1
