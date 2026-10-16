torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=8 \
  baselines/amdm_repro/train_amdm.py \
  --config_file baselines/amdm_repro/configs/amdm_realiad.yaml \
  --eval_only \
  --eval_ckpt baselines/amdm_repro/out/amdm_mc_unet_enet_realiad_0922/checkpoints/ckpt_epoch_300.pth