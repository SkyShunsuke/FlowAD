"""Single-class (one model per category) DiT-G + EfficientNet-B4 FM on VisA, following logs/visa_ditG_enet_0829
(linear-t FM, patch 1, feature_res 16 -> 256 tokens, lr/warmup/300 epochs unchanged). Changes vs 0829:
  - dataset_name visa + category (one DiT per class), use_class_labels false (a single class carries no information)
  - 1 GPU x 8 (0829: 8 GPUs x 8)
  - fp32 + TF32 (single-GPU bf16 DiT-G diverged to NaN)
  - eval every 10 epochs, early stop when mAD has not improved for 20 epochs; best eval epoch is reported
  - no best checkpoint written (home quota), best_results.json only
    python scripts/gen_visa_sc_ditG_enet_0829_configs.py
"""
import os

import yaml

BASE = 'logs/visa_ditG_enet_0829/config.yaml'
TAG = 'visa_sc_ditG_enet_0829'
OUT = f'configs/train/{TAG}'
CLASSES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
           'pipe_fryum']

os.makedirs(OUT, exist_ok=True)
for c in CLASSES:
    with open(BASE) as f:
        cfg = yaml.safe_load(f)
    cfg['data'].update(dataset_name='visa', category=c, train_batch_size=8)
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
