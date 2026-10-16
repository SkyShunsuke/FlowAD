"""DiT-G counterparts of the DiT-B multi-seed MVTec AD / VisA runs (configs/train/ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds),
for a like-for-like model-scale comparison: 1 GPU x bs8, patch 1, feature_res 16, fp32 + TF32, eval every epoch,
early-stop patience 10. Only the DiT size (B -> G: depth 16 / hidden 2048 / 16 heads), log paths and run names change.
Written next to the existing MPDD DiT-G configs of the same setting.
    python scripts/gen_ditG_enet_fm_bs8_p1_es10_configs.py [seeds...]
"""
import os
import sys

import yaml

SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
SRC = 'configs/train/ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds'
TAG = 'ditG_enet_fm_fr16_tf32_bs8_p1_es10_seeds'
OUT = f'configs/train/{TAG}'

os.makedirs(OUT, exist_ok=True)
for ds in ['ad', 'visa']:
    for s in SEEDS:
        run = f'{ds}_s{s}'
        with open(f'{SRC}/{run}.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['model'].update(depth=16, hidden_size=2048, num_heads=16)
        cfg['logging']['log_dir'] = f'logs/{TAG}/{run}/'
        cfg['logging']['wandb']['run_name'] = f'{ds}_ditG_enet_fm_fr16_tf32_bs8_p1_es10_seed{s}'
        with open(f'{OUT}/{run}.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
print(f'2 datasets x {len(SEEDS)} seeds -> {OUT}')
