"""DiT-G counterparts of the per-category RealIAD DiT-B fr32/patch-2 multi-seed runs
(configs/train/realiad_ditB_enet_fm_fr32_tf32_bs8_p2_es10_seeds). Only the DiT size (B -> G: depth 16 / hidden 2048 /
16 heads), log paths and run names change.
    python scripts/gen_realiad_ditG_enet_fm_fr32_p2_seed_configs.py [seeds...]
"""
import os
import sys

import yaml

SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
SRC = 'configs/train/realiad_ditB_enet_fm_fr32_tf32_bs8_p2_es10_seeds'
TAG = 'realiad_ditG_enet_fm_fr32_tf32_bs8_p2_es10_seeds'
OUT = f'configs/train/{TAG}'

n = 0
for s in SEEDS:
    os.makedirs(f'{OUT}/seed{s}', exist_ok=True)
    for fn in sorted(os.listdir(f'{SRC}/seed{s}')):
        c = fn[:-5]
        with open(f'{SRC}/seed{s}/{fn}') as f:
            cfg = yaml.safe_load(f)
        cfg['model'].update(depth=16, hidden_size=2048, num_heads=16)
        cfg['logging']['log_dir'] = f'logs/{TAG}/seed{s}/{c}/'
        cfg['logging']['wandb']['run_name'] = f'realiad_{c}_ditG_enet_fm_fr32_tf32_bs8_p2_es10_seed{s}'
        with open(f'{OUT}/seed{s}/{fn}', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        n += 1
print(f'{n} configs ({len(SEEDS)} seeds) -> {OUT}')
