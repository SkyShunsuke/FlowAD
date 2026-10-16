#!/bin/bash
# Final evaluation of out/vpdm30x_realiad_jitter_s2 (raw weights, Alg. 2 full_batch + y_hat
# diagnostic) with one process per GPU, then merge. Run from the repo root.
PY=~/venv/jit/bin/python
RUN=baselines/vpdm_repro/out/vpdm30x_realiad_jitter_s2
GPU_LIST=(${GPUS:-3 4 5 6 7})
N=${#GPU_LIST[@]}
for i in "${!GPU_LIST[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} $PY baselines/vpdm_repro/eval_shard.py --log_dir $RUN \
    --shard $i --num_shards $N > $RUN/eval_shard$i.log 2>&1 &
done
wait
$PY baselines/vpdm_repro/eval_shard.py --log_dir $RUN --num_shards $N --merge > $RUN/eval_merge.log 2>&1
