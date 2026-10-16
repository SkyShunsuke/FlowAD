#!/bin/bash
# VPDM reproduction on multi-class MVTec-AD: 3 seeds (paper reports mean±std)
# + Table 3 "N(0,I)" (standard DDPM endpoint) ablation. One GPU per run.
# Run from the FlowAD repo root.
PY=~/venv/jit/bin/python
CFG=baselines/vpdm_repro/configs/vpdm_mvtec.yaml
mkdir -p baselines/vpdm_repro/out
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s nohup $PY baselines/vpdm_repro/train_vpdm.py --config_file $CFG \
    --override opt.global_seed=$s logging.log_dir=baselines/vpdm_repro/out/vpdm_mvtec_s$s/ \
    > baselines/vpdm_repro/out/vpdm_mvtec_s$s.log 2>&1 &
done
CUDA_VISIBLE_DEVICES=3 nohup $PY baselines/vpdm_repro/train_vpdm.py --config_file $CFG \
  --override diffusion.endpoint=standard logging.log_dir=baselines/vpdm_repro/out/vpdm_mvtec_ablN01_s0/ \
  > baselines/vpdm_repro/out/vpdm_mvtec_ablN01_s0.log 2>&1 &
wait
