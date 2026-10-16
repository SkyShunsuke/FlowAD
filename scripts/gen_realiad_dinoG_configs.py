"""Configs for RealIAD per-category FM + 1-step recon on DINOv2-G features.

  python scripts/gen_realiad_dinoG_configs.py pilot         # DiT-S 50 ep train + t-sweep eval configs
  python scripts/gen_realiad_dinoG_configs.py full --t 0.3  # DiT-G train configs (early stopping, best ckpt)

Base: configs/train/realiad_ditS_fm_t02 (plain FM, uniform t, 1-step inversion + 1-step recon, diff+sum).
Backbone: DINOv2-giant, final-LayerNorm output of the last block, 256px -> 224px -> (1536, 16, 16).
"""
import argparse
import copy
import os

import yaml

BASE_DIR = 'configs/train/realiad_ditS_fm_t02'
CATS = sorted(f[:-5] for f in os.listdir(BASE_DIR) if f.endswith('.yaml'))
TS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BACKBONE = {'model_name': 'dinov2-giant', 'pretrained': True, 'normalization': None,
            'out_blocks': [-1], 'final_norm': True}
DIT_S = {'hidden_size': 384, 'depth': 12, 'num_heads': 6, 'cond_channels': 384, 'patch_size': 1}
DIT_G = {'hidden_size': 2048, 'depth': 16, 'num_heads': 16, 'cond_channels': 768, 'patch_size': 1}

PILOT_TRAIN = 'configs/train/realiad_ditS_dinoG_pilot'
PILOT_EVAL = 'configs/eval/realiad_ditS_dinoG_pilot_sweep'
PILOT_LOG = 'logs/realiad_ditS_dinoG_pilot'
FULL_TRAIN = 'configs/train/realiad_ditG_dinoG_fm'
FULL_LOG = 'logs/realiad_ditG_dinoG_fm'


def base(cat):
    with open(os.path.join(BASE_DIR, f'{cat}.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['model']['backbone'] = copy.deepcopy(BACKBONE)
    return cfg


def dump(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=True)


def pilot():
    for cat in CATS:
        cfg = base(cat)
        cfg['model'].update(DIT_S)
        cfg['opt'].update({'epochs': 50, 'lr_warmup_epochs': 10})
        log = cfg['logging']
        log['log_dir'] = f'{PILOT_LOG}/{cat}/'
        log['eval']['eval_epoch_freq'] = 1000  # no eval during training; t-sweep evaluates ckpt_epoch_50
        log['eval'].pop('early_stop_patience', None)
        log['ckpt'].update({'ckpt_epoch_freq': 50, 'save_best': False})
        log['wandb']['use_wandb'] = False
        dump(cfg, f'{PILOT_TRAIN}/{cat}.yaml')

        for t in TS:
            ev = {
                'data': {**{k: v for k, v in cfg['data'].items() if k != 'train_batch_size'}, 'test_batch_size': 8},
                'flow_matching': cfg['flow_matching'],
                'meta': {'global_seed': 0, 'name': 'vfad', 'use_bfloat16': True},
                'model': cfg['model'],
                'logging': {
                    'log_dir': f'{PILOT_LOG}/{cat}/eval_noise_sweep/t{t}/',
                    'eval': {**{k: log['eval'][k] for k in ('img_score_agg', 'metrics', 'solver_name', 'steps', 'recon_steps')},
                             'eval_strategy': 'recon', 'inv_stop_time': t, 'num_recon_emsemble': 1,
                             'ablation_steps': [-1], 'save_dir': None},
                },
                'resume': {'resume_path': f'{PILOT_LOG}/{cat}/checkpoints/ckpt_epoch_50.pth'},
            }
            dump(ev, f'{PILOT_EVAL}/{cat}_t{t}.yaml')
    print(f'{len(CATS)} pilot train configs -> {PILOT_TRAIN}, {len(CATS) * len(TS)} eval configs -> {PILOT_EVAL}')


def full(t):
    for cat in CATS:
        cfg = base(cat)
        cfg['model'].update(DIT_G)
        log = cfg['logging']
        log['log_dir'] = f'{FULL_LOG}/{cat}/'
        log['eval'].update({'inv_stop_time': t, 'eval_epoch_freq': 25, 'early_stop_patience': 50})
        log['ckpt'].update({'ckpt_epoch_freq': 500, 'save_best': True})
        log['wandb']['run_name'] = f'realiad_{cat}_ditG_dinoG_fm_t{t}'
        dump(cfg, f'{FULL_TRAIN}/{cat}.yaml')
    print(f'{len(CATS)} DiT-G train configs (t={t}) -> {FULL_TRAIN}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['pilot', 'full'])
    p.add_argument('--t', type=float)
    a = p.parse_args()
    pilot() if a.stage == 'pilot' else full(a.t)
