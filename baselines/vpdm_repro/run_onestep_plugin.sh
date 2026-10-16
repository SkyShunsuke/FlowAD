#!/bin/bash
# FlowAD one-step reconstruction plugged into trained VPDM checkpoints (NOT in the paper).
# Variant oN: x_N ~ q(x_N | x0, y_hat), one eps prediction -> x0_hat, VPDM's own score
# (||x0 - x0_hat||_2 map, avg-pool max image score). N=658 matches FlowAD's t_FM=0.1 SNR.
# Writes to out/onestep_<run>/ (checkpoints symlinked) so the original run dirs are untouched.
# Run from the repo root: bash baselines/vpdm_repro/run_onestep_plugin.sh
PY=~/venv/jit/bin/python
OUT=baselines/vpdm_repro/out
VARS='[o50_batch,o100_batch,o200_batch,o300_batch,o400_batch,o527_batch,o658_batch,o762_batch,o900_batch]'
RUNS=(vpdm300_mvtec_s0 vpdm300_mvtec_s1 vpdm300_mvtec_s2
      vpdm300x_mvtec_jitter_s2 vpdm300x_visa_jitter_s2 vpdm300x_mpdd_jitter_s2 vpdm30x_realiad_jitter_s2)
gpu=1  # GPU 0: InvAD plug-in eval
for run in "${RUNS[@]}"; do
  dst=$OUT/onestep_$run
  mkdir -p $dst
  [ -e $dst/checkpoints ] || ln -s $(realpath $OUT/$run/checkpoints) $dst/checkpoints
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY baselines/vpdm_repro/train_vpdm.py --config_file $OUT/$run/config.yaml --eval_only \
    --override logging.log_dir=$dst/ "logging.eval.final_variants=$VARS" 'logging.eval.ema_final_variants=[]' \
      logging.eval.devices=1 logging.eval.save_maps=false \
    > $dst.log 2>&1 &
  gpu=$((gpu + 1))
done
wait
