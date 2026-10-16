"""Per-category RealIAD DiT-G + EfficientNet-B4 (logistic-t FM, 1-step recon from t=0.1) configs for extra seeds.
Based on the seed-0 runs in logs/realiads/realiad_mc_ditG_enet_0903_<cat>; only the seed, early stopping,
checkpoint saving and log paths change. Run on 8 GPUs x 8 like seed 0: 1 GPU x 64 diverges to NaN under bf16."""
import os
import sys

import yaml

SEEDS = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4]
SRC = 'logs/realiads/realiad_mc_ditG_enet_0903_{}/config.yaml'
OUT = 'configs/train/realiad_ditG_enet_seeds'
CATS = sorted(d.split('0903_')[1] for d in os.listdir('logs/realiads') if d.startswith('realiad_mc_ditG_enet_0903_'))

for s in SEEDS:
    os.makedirs(f'{OUT}/seed{s}', exist_ok=True)
    for c in CATS:
        with open(SRC.format(c)) as f:
            cfg = yaml.safe_load(f)
        assert cfg['data']['category'] == c
        assert cfg['data']['train_batch_size'] == 8  # per GPU, x 8 GPUs
        # seed-0 checkpoints have 256 tokens (32x32 map at patch 2): pin it against default changes
        cfg['model']['backbone']['feature_res'] = 32
        cfg['opt']['global_seed'] = s
        # home quota is full: keep best-epoch tracking but skip the ~5GB best checkpoint
        cfg['logging']['ckpt'].update(save_best=True, save_best_ckpt=False, ckpt_epoch_freq=500)
        cfg['logging']['eval'].update(eval_epoch_freq=25, early_stop_patience=50)
        cfg['logging']['log_dir'] = f'logs/realiad_ditG_enet_seeds/seed{s}/{c}/'
        cfg['logging']['wandb']['run_name'] = f'realiad_{c}_ditG_enet_seed{s}'
        with open(f'{OUT}/seed{s}/{c}.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
print(f'{len(SEEDS)} seeds x {len(CATS)} categories -> {OUT}')
