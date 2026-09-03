#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  main.py --config_file ./configs/train/realiad/realiad_ditG_enet.yaml --task train