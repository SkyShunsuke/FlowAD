"""Multi-seed DiT-B + EfficientNet-B4 FM on MVTec AD with a FIXED training timestep t = 0.1 (noisy side; t=0 is noise).
Ablation of configs/train/ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds/ad_s*.yaml, whose training t ~ logistic(mu=logit(0.1),
sigma=0.35) over (0, 1). Only t_scheduler_train (-> constant, t_mu = 0.1), seed and log paths change; inference is untouched
(1-step recon from inv_stop_time 0.1).
    python scripts/gen_ditB_enet_fm_tfix01_seed_configs.py [seeds...]
"""
import os
import sys

import yaml

SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
SRC = 'configs/train/ditB_enet_fm_fr16_tf32_bs8_p1_es10_seeds'
TAG = 'ditB_enet_fm_fr16_tf32_bs8_p1_es10_tfix01_seeds'
OUT = f'configs/train/{TAG}'
T_FIX = 0.1

os.makedirs(OUT, exist_ok=True)
for s in SEEDS:
    with open(f'{SRC}/ad_s{s}.yaml') as f:
        cfg = yaml.safe_load(f)
    run = f'ad_s{s}'
    cfg['flow_matching']['scheduler'].update(t_scheduler_train='constant', t_mu=T_FIX)
    cfg['flow_matching']['scheduler'].pop('t_sigma', None)
    assert cfg['opt']['global_seed'] == s
    cfg['logging']['log_dir'] = f'logs/{TAG}/{run}/'
    cfg['logging']['wandb']['run_name'] = f'ad_ditB_enet_fm_fr16_tf32_bs8_p1_es10_tfix01_seed{s}'
    with open(f'{OUT}/{run}.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
print(f'MVTec AD x {len(SEEDS)} seeds (train t fixed to {T_FIX}) -> {OUT}')
