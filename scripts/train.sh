#!/bin/bash
export WANDB_API_KEY=ec54c112b7a4f2600ac4e283c5dfd1f8f06df606
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  main.py --config_file ./configs/train/ad_ditB_vp.yaml --task train