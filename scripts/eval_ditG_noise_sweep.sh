#!/bin/bash
# Flow + DiT-G + 1-step recon: sweep the noise level inv_stop_time t in {0.0..0.9}
# (x_t = t*x + (1-t)*eps, then one Euler step to t=1).
#   bash scripts/eval_ditG_noise_sweep.sh                                        # MVTec AD / VisA / MPDD
#   bash scripts/eval_ditG_noise_sweep.sh configs/eval/ditG_noise_sweep_realiad  # RealIAD (30 per-class models)
# Ensemble size follows the original eval configs (MVTec 8 / VisA 4 / MPDD 1 / RealIAD 1).
# Configs run ordered by t, so each noise level finishes for all classes before the next.
# Results: <log_dir>/eval_results.yaml, log_dir = <run>/eval_noise_sweep/t<t>/. 8 GPUs, one job queue per GPU.
cd "$(dirname "$0")/.."
TR=~/venv/jit/bin/torchrun
C=${1:-configs/eval/ditG_noise_sweep}
mapfile -t CFGS < <(ls $C/*.yaml | awk -F_t '{print $NF "\t" $0}' | sort | cut -f2)
worker() {  # gpu
  local g=$1 i=0
  for cfg in "${CFGS[@]}"; do
    if [ $((i % 8)) -eq $g ]; then
      d=$(grep "^  log_dir:" $cfg | awk '{print $2}')
      [ -f $d/eval_results.yaml ] || {
        mkdir -p $d
        CUDA_VISIBLE_DEVICES=$g $TR --nnodes=1 --nproc-per-node=1 --master_port=$((12500 + g)) \
          main.py --config_file $cfg --task eval > ${d%/}.log 2>&1
      }
    fi
    i=$((i + 1))
  done
}
for g in 0 1 2 3 4 5 6 7; do worker $g & done
wait
