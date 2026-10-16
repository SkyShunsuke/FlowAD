"""Per-category RealIAD DiT-B + EfficientNet-B4 FM multi-seed configs, feature_res 32 + patch 2.
Same as scripts/gen_realiad_ditB_enet_fm_seed_configs.py (fr16, patch 1) except EfficientNet feature_res 16 -> 32 and
DiT patch_size 1 -> 2 (token grid stays 16x16). FM scheduler: affine_prob, logistic t (mu=-2.197, sigma=0.35), linear infer.
    python scripts/gen_realiad_ditB_enet_fm_fr32_p2_seed_configs.py [seeds...]
"""
import os
import sys

import yaml

SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
REF = 'configs/train/realiad_ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds'
TAG = 'realiad_ditB_enet_fm_fr32_tf32_bs8_p2_es10_seeds'
OUT = f'configs/train/{TAG}'
CATS = sorted(f[:-5] for f in os.listdir(f'{REF}/seed0'))

for s in SEEDS:
    os.makedirs(f'{OUT}/seed{s}', exist_ok=True)
    for c in CATS:
        with open(f'{REF}/seed0/{c}.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['model']['backbone']['feature_res'] = 32
        cfg['model']['patch_size'] = 2
        cfg['flow_matching']['scheduler'] = dict(name='affine_prob', t_mu=-2.1972246, t_scheduler_infer='linear',
                                                 t_scheduler_train='logistic', t_sigma=0.35)
        cfg['opt']['global_seed'] = s
        cfg['logging']['log_dir'] = f'logs/{TAG}/seed{s}/{c}/'
        cfg['logging']['wandb']['run_name'] = f'realiad_{c}_ditB_enet_fm_fr32_tf32_bs8_p2_es10_seed{s}'
        with open(f'{OUT}/seed{s}/{c}.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
print(f'{len(SEEDS)} seeds x {len(CATS)} categories -> {OUT}')
