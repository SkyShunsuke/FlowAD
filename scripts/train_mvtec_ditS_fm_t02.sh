#!/bin/bash
# DiT-S, plain FM (uniform t), one-step recon from t=0.2; early stopping + best ckpt.
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p logs/scale
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  main.py --config_file ./configs/train/mvtec_mc_ditS_enet_fm_t02.yaml --task train \
  > logs/scale/mvtec_mc_ditS_enet_fm_t02.out 2>&1
echo "exit $?"
