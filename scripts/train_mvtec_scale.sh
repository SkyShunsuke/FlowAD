#!/bin/bash
# Model-scale study on MVTec AD (multi-class): DiT-B / DiT-L / DiT-H, one-step recon.
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p logs/scale
for S in B L H; do
  echo "==== DiT-$S start $(date) ===="
  torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    main.py --config_file ./configs/train/mvtec_mc_dit${S}_enet.yaml --task train \
    > logs/scale/mvtec_mc_dit${S}_enet.out 2>&1
  echo "==== DiT-$S done (exit $?) $(date) ===="
done
