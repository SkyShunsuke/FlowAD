#!/bin/bash
# 2D recon sweep (noise strength x recon_steps) of the best FM DiT-G + EnetB4 model, 8 GPUs.
#   python scripts/gen_ditG_2d_sweep_configs.py visa mpdd && bash scripts/eval_ditG_2d_sweep.sh visa mpdd
#   python scripts/summarize_ditG_2d_sweep.py visa mpdd   # -> <run>/eval_2d_sweep/{results,mad_grid}.csv
# Shared job queue (most recon steps first); configs with an existing eval_results.yaml are skipped.
cd "$(dirname "$0")/.."
TR=~/venv/jit/bin/torchrun
DS=${@:-mvtec}
Q=$(mktemp -d)
for d in $DS; do ls configs/eval/ditG_2d_sweep_$d/*.yaml; done | awk -F_n '{print $NF "\t" $0}' | sort -t. -k1,1nr | cut -f2 > $Q/queue
: > $Q/pos
next() {  # pop one config path from the queue (atomic)
  flock $Q/lock bash -c "n=\$(wc -l < $Q/pos); sed -n \"\$((n + 1))p\" $Q/queue; echo >> $Q/pos"
}
worker() {  # gpu
  local g=$1 cfg d
  while cfg=$(next) && [ -n "$cfg" ]; do
    d=$(grep "^  log_dir:" $cfg | awk '{print $2}')
    [ -f $d/eval_results.yaml ] && continue
    mkdir -p $d
    echo "$(date +%T) gpu$g start $cfg"
    CUDA_VISIBLE_DEVICES=$g $TR --nnodes=1 --nproc-per-node=1 --master_port=$((12600 + g)) \
      main.py --config_file $cfg --task eval > ${d%/}.log 2>&1
    echo "$(date +%T) gpu$g done  $cfg (exit $?)"
  done
}
for g in 0 1 2 3 4 5 6 7; do worker $g & done
wait
rm -rf $Q
