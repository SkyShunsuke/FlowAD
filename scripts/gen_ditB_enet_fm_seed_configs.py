"""Multi-seed DiT-B + EfficientNet-B4 (logistic-t FM, 1-step recon from t=0.1) configs for MVTec AD / VisA / MPDD.
Model-scale ablation of the best DiT-G multi-seed setting (configs/train/ditG_enet_fm_fr16_tf32_bs8_p1_es10_seeds):
1 GPU x bs8, patch 1, feature_res 16, fp32 + TF32, eval every epoch with early-stop patience 10. Only the DiT size
(G: depth 16 / hidden 2048 / 16 heads -> B: depth 12 / hidden 768 / 12 heads), dataset, seed and log paths change.
    python scripts/gen_ditB_enet_fm_seed_configs.py [seeds...]
"""
import os
import sys

import yaml

SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
BASE = 'configs/train/ditG_enet_fm_fr16_tf32_bs8_p1_es10_seeds/mpdd_s0.yaml'
TAG = 'ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds'
OUT = f'configs/train/{TAG}'
DATASETS = {'ad': ('data/mvtec_ad', 'mvtec_ad_all'), 'visa': ('data/visa', 'visa_all'), 'mpdd': ('data/mpdd', 'mpdd_all')}

os.makedirs(OUT, exist_ok=True)
for ds, (root, name) in DATASETS.items():
    for s in SEEDS:
        with open(BASE) as f:
            cfg = yaml.safe_load(f)
        run = f'{ds}_s{s}'
        cfg['data'].update(data_root=root, dataset_name=name)
        cfg['model'].update(depth=12, hidden_size=768, num_heads=12)
        cfg['opt']['global_seed'] = s
        cfg['logging']['log_dir'] = f'logs/{TAG}/{run}/'
        cfg['logging']['wandb']['run_name'] = f'{ds}_ditB_enet_fm_fr16_tf32_bs8_p1_es10_seed{s}'
        with open(f'{OUT}/{run}.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
print(f'{len(DATASETS)} datasets x {len(SEEDS)} seeds -> {OUT}')
