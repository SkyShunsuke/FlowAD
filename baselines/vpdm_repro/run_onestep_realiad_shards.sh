#!/bin/bash
# FlowAD one-step plug-in on out/vpdm30x_realiad_jitter_s2 (raw weights), one process per GPU, then merge.
# Uses out/onestep_vpdm30x_realiad_jitter_s2 (config.yaml + symlinked checkpoints). Run from the repo root.
PY=~/venv/jit/bin/python
RUN=baselines/vpdm_repro/out/onestep_vpdm30x_realiad_jitter_s2
V="o50_batch o100_batch o200_batch o300_batch o400_batch o527_batch o658_batch o762_batch o900_batch"
GPU_LIST=(${GPUS:-2 3 4 5 6 7})
N=${#GPU_LIST[@]}
for i in "${!GPU_LIST[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} $PY baselines/vpdm_repro/eval_shard.py --log_dir $RUN --variants $V \
    --shard $i --num_shards $N > $RUN/eval_shard$i.log 2>&1 &
done
wait
$PY baselines/vpdm_repro/eval_shard.py --log_dir $RUN --variants $V --num_shards $N --merge > $RUN/eval_merge.log 2>&1
