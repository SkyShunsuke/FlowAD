#!/bin/bash
# Inversion- and density-based scoring for logs/ad_ditG_enet_0829 (same checkpoint),
# steps in {1,2,4,8,20}. `bash scripts/eval_ad_ditG_invdens.sh recon` runs recon-based scoring instead. One GPU per chain; results in logs/ad_ditG_enet_0829/eval_{inv,density}_s*/.
cd "$(dirname "$0")/.."
TR=~/venv/jit/bin/torchrun
C=configs/eval/ad_ditG_enet_0829_invdens
chain() {  # gpu port cfg...
  local gpu=$1 port=$2; shift 2
  for cfg in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu $TR --nnodes=1 --nproc-per-node=1 --master_port=$port \
      main.py --config_file $C/$cfg.yaml --task eval > logs/ad_ditG_enet_0829/eval_$cfg.log 2>&1
  done
}
if [ "$1" == "recon" ]; then
  chain 1 12411 recon_s1 recon_s2 &
  chain 3 12412 recon_s4 &
  chain 4 12413 recon_s8 &
  chain 5 12414 recon_s20 &
  wait; exit
fi
chain 3 12401 inv_s1 inv_s2 inv_s4 inv_s8 inv_s20 &
chain 4 12402 density_s20 &
chain 5 12403 density_s8 &
chain 6 12404 density_s4 &
chain 7 12405 density_s2 density_s1 &
wait
