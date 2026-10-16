"""Single-class (one model per category) DiT-G + EfficientNet-B4 FM on MVTec AD, following logs/ad_ditG_enet_0829
(linear-t FM, patch 1, feature_res 16 -> 256 tokens, lr/warmup/300 epochs unchanged). Changes vs 0829:
  - dataset_name mvtec_ad + category (one DiT per class), use_class_labels false (a single class carries no information)
  - 1 GPU x 8 (0829: 8 GPUs x 8; global 64 would leave toothbrush (60 imgs) with 0 iters/epoch under drop_last)
  - fp32 + TF32 (single-GPU bf16 DiT-G diverged to NaN)
  - eval every 10 epochs, early stop when mAD has not improved for 20 epochs; best eval epoch is reported
  - no best checkpoint written (home quota), best_results.json only
    python scripts/gen_ad_sc_ditG_enet_0829_configs.py
"""
import os

import yaml

BASE = 'logs/ad_ditG_enet_0829/config.yaml'
TAG = 'ad_sc_ditG_enet_0829'
OUT = f'configs/train/{TAG}'
CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
           'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

os.makedirs(OUT, exist_ok=True)
for c in CLASSES:
    with open(BASE) as f:
        cfg = yaml.safe_load(f)
    cfg['data'].update(dataset_name='mvtec_ad', category=c, train_batch_size=8)
    cfg['model']['backbone']['feature_res'] = 16
    cfg['model']['use_class_labels'] = False
    cfg['opt'].update(use_bfloat16=False, use_tf32=True)
    cfg['logging']['ckpt'].update(save_best=True, save_best_ckpt=False, save_model_only=True, ckpt_epoch_freq=500)
    cfg['logging']['eval'].update(eval_epoch_freq=10, early_stop_patience=20)
    cfg['logging']['log_dir'] = f'logs/{TAG}/{c}/'
    cfg['logging']['wandb']['run_name'] = f'{TAG}_{c}'
    with open(f'{OUT}/{c}.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
print(f'{len(CLASSES)} classes -> {OUT}')
