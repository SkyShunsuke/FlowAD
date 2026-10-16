#!/bin/bash
# Step 2: denoiser additionally conditioned on the input features (NOT in the
# paper; CARD-style). 300-epoch protocol. jitter / mask x 3 seeds, clean (no
# corruption = identity-shortcut upper bound) x 1 seed. Run from the repo root.
PY=~/venv/jit/bin/python
CFG=baselines/vpdm_repro/configs/vpdm_mvtec_300ep_x0cond.yaml
OUT=baselines/vpdm_repro/out
launch() {  # gpu mode seed
  CUDA_VISIBLE_DEVICES=$1 nohup $PY baselines/vpdm_repro/train_vpdm.py --config_file $CFG \
    --override x0_cond.mode=$2 opt.global_seed=$3 logging.log_dir=$OUT/vpdm300x_mvtec_$2_s$3/ \
    > $OUT/vpdm300x_mvtec_$2_s$3.log 2>&1 &
}
launch 0 jitter 0; launch 1 jitter 1; launch 2 jitter 2
launch 3 mask 0;   launch 6 mask 1;   launch 7 mask 2
launch 4 clean 0
wait
