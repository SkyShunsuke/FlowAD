"""Multi-seed DiT-G + EfficientNet-B4 (logistic-t FM, 1-step recon from t=0.1) configs for MVTec AD / VisA / MPDD.
Based on configs/train/ad_ditG_enet.yaml (the 0903 setting); only dataset, batch layout (1 GPU x 64 = same global
batch 64, no DDP), precision (fp32 + TF32: single-GPU bf16 diverged to NaN on AD/VisA), feature_res, seed and log paths change. Best epoch = eval epoch (every 25) with the highest mAD.
    python scripts/gen_ditG_enet_fm_seed_configs.py [feature_res=32] [seeds...]
"""
import os
import sys

import yaml

FR = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SEEDS = [int(s) for s in sys.argv[2:]] or [0, 1, 2, 3, 4]
BASE = 'configs/train/ad_ditG_enet.yaml'
TAG = f'ditG_enet_fm_fr{FR}_tf32_seeds'
OUT = f'configs/train/{TAG}'
DATASETS = {'ad': ('data/mvtec_ad', 'mvtec_ad_all'), 'visa': ('data/visa', 'visa_all'), 'mpdd': ('data/mpdd', 'mpdd_all')}

os.makedirs(OUT, exist_ok=True)
for ds, (root, name) in DATASETS.items():
    for s in SEEDS:
        with open(BASE) as f:
            cfg = yaml.safe_load(f)
        run = f'{ds}_s{s}'
        cfg['data'].update(data_root=root, dataset_name=name, train_batch_size=64)
        cfg['model']['backbone']['feature_res'] = FR
        cfg['opt']['global_seed'] = s
        cfg['opt'].update(use_bfloat16=False, use_tf32=True)
        # home quota is full: keep best-epoch tracking (best_results.json) but skip the best checkpoint
        cfg['logging']['ckpt'].update(save_best=True, save_best_ckpt=False, ckpt_epoch_freq=500)
        cfg['logging']['log_dir'] = f'logs/{TAG}/{run}/'
        cfg['logging']['wandb']['run_name'] = f'{ds}_ditG_enet_fm_fr{FR}_tf32_seed{s}'
        with open(f'{OUT}/{run}.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
print(f'{len(DATASETS)} datasets x {len(SEEDS)} seeds -> {OUT}')
