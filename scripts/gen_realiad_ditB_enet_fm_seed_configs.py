"""Per-category RealIAD DiT-B + EfficientNet-B4 (logistic-t FM, 1-step recon from t=0.1) multi-seed configs.
Same setting as the MVTec AD / VisA / MPDD DiT-B runs (configs/train/ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds):
1 GPU x bs8, patch 1, feature_res 16, fp32 + TF32, eval every epoch, early-stop patience 10. One model per category;
the best epoch (highest mAD) is selected per category and recorded in best_results.json (no checkpoint, home quota).
    python scripts/gen_realiad_ditB_enet_fm_seed_configs.py [seeds...]
"""
import os
import sys

import yaml

SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
BASE = 'configs/train/ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds/ad_s0.yaml'
TAG = 'realiad_ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds'
OUT = f'configs/train/{TAG}'
META = 'data/realiad_meta/realiad_jsons'
CATS = sorted(f[:-5] for f in os.listdir(META) if f.endswith('.json'))

for s in SEEDS:
    os.makedirs(f'{OUT}/seed{s}', exist_ok=True)
    for c in CATS:
        with open(BASE) as f:
            cfg = yaml.safe_load(f)
        cfg['data'].update(category=c, data_root='data/realiad', dataset_name='realiad', meta_dir=f'./{META}')
        cfg['opt']['global_seed'] = s
        cfg['logging']['log_dir'] = f'logs/{TAG}/seed{s}/{c}/'
        cfg['logging']['wandb']['run_name'] = f'realiad_{c}_ditB_enet_fm_fr16_tf32_bs8_p1_es10_seed{s}'
        with open(f'{OUT}/seed{s}/{c}.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
print(f'{len(SEEDS)} seeds x {len(CATS)} categories -> {OUT}')
