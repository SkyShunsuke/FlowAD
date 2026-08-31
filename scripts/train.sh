#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  main.py --config_file ./configs/train/mpdd_ditG_enet.yaml --task train