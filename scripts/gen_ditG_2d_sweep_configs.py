"""2D recon sweep for the best FM DiT-G + EfficientNet-B4 model per dataset (final-epoch weights).
Grid: noise strength s in {0.1..0.9} x recon_steps in {1,2,4,8,20,40,80,100,200}.
Noise strength s = 1 - inv_stop_time (x_t = t*x + (1-t)*eps, so s is the noise weight), then recon_steps Euler steps t -> 1.
Ensemble 1, as in the models' training-time evaluation. feature_res is read from the ckpt pos_embed.
    python scripts/gen_ditG_2d_sweep_configs.py mvtec visa mpdd  # -> configs/eval/ditG_2d_sweep_<ds>/s<s>_n<steps>.yaml
    python scripts/gen_ditG_2d_sweep_configs.py realiad_ditB     # per-class models -> .../<cls>_s<s>_n<steps>.yaml
"""
import copy
import os
import sys

import torch
import yaml

RUNS = {  # dataset -> (run dir, ckpt)
    'mvtec': ('logs/ad_mc_ditG_enet_0903', 'ckpt_epoch_300.pth'),
    'visa': ('logs/visa_ditG_enet_0829', 'ckpt_model.pth'),
    'mpdd': ('logs/mpdd_ditG_enet_0829', 'ckpt_model.pth'),
    'mvtec0829': ('logs/ad_ditG_enet_0829', 'ckpt_model.pth'),
    'mvtec_ditB': ('logs/mvtec_mc_ditB_enet', 'ckpt_best.pth'),
    'visa_ditB': ('logs/visa_ditB_enet', 'ckpt_best.pth'),
    'mpdd_ditB': ('logs/mpdd_mc_ditB_enet', 'ckpt_best.pth'),
}
PER_CLASS = {  # dataset -> (run root with one run dir per class, ckpt, noise-sweep template dir, test_batch_size)
    'realiad_ditB': ('logs/realiad_ditB_enet', 'ckpt_best.pth', 'configs/eval/ditB_noise_sweep_realiad', 32),
}
TMPL = {'mvtec0829': 'mvtec', 'mvtec_ditB': 'mvtec', 'visa_ditB': 'visa', 'mpdd_ditB': 'mpdd'}  # dataset -> noise-sweep template key, if different
NOISE = [round(0.1 * i, 1) for i in range(1, 10)]
STEPS = [1, 2, 4, 8, 20, 40, 80, 100, 200]


def feature_res(ckpt, patch_size):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    n_tok = next(v.shape[1] for k, v in sd.items() if k.endswith('model.pos_embed'))
    return int(round(n_tok ** 0.5)) * patch_size


def make_base(run, ckpt, tmpl):
    train_cfg = yaml.safe_load(open(f'{run}/config.yaml'))
    tmpl = yaml.safe_load(open(tmpl))
    base = copy.deepcopy(tmpl)
    base['data'] = {**tmpl['data'], 'test_batch_size': train_cfg['data']['test_batch_size']}
    base['flow_matching'] = train_cfg['flow_matching']
    base['model'] = copy.deepcopy(train_cfg['model'])
    base['model']['backbone']['feature_res'] = feature_res(f'{run}/checkpoints/{ckpt}', base['model']['patch_size'])
    base['meta'] = {**tmpl['meta'], 'global_seed': train_cfg.get('opt', {}).get('global_seed', 0)}
    base['logging']['eval']['num_recon_emsemble'] = 1
    base['resume'] = {'resume_path': f'{run}/checkpoints/{ckpt}'}
    return base


def write_grid(base, run, out, prefix=''):
    for s in NOISE:
        for n in STEPS:
            cfg = copy.deepcopy(base)
            ev = cfg['logging']['eval']
            ev['inv_stop_time'] = round(1 - s, 1)
            ev['recon_steps'] = n
            cfg['logging']['log_dir'] = f'{run}/eval_2d_sweep/s{s}_n{n}/'
            with open(f'{out}/{prefix}s{s}_n{n}.yaml', 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=True)


for ds in sys.argv[1:] or RUNS:
    out = f'configs/eval/ditG_2d_sweep_{ds}'
    os.makedirs(out, exist_ok=True)
    if ds in PER_CLASS:  # one model per class; shared noise_base per run, so test_batch_size does not change scores
        root, ckpt, tmpl_dir, bs = PER_CLASS[ds]
        classes = sorted(c for c in os.listdir(root) if os.path.exists(f'{root}/{c}/checkpoints/{ckpt}'))
        for c in classes:
            base = make_base(f'{root}/{c}', ckpt, f'{tmpl_dir}/{c}_t0.1.yaml')
            base['data']['test_batch_size'] = bs
            base['logging']['wandb']['use_wandb'] = False
            write_grid(base, f'{root}/{c}', out, prefix=f'{c}_')
        print(f'{ds}: {len(classes)} classes, feature_res {base["model"]["backbone"]["feature_res"]}, '
              f'wrote {len(classes) * len(NOISE) * len(STEPS)} configs to {out}')
        continue
    run, ckpt = RUNS[ds]
    base = make_base(run, ckpt, f'configs/eval/ditG_noise_sweep/{TMPL.get(ds, ds)}_t0.1.yaml')
    write_grid(base, run, out)
    print(f'{ds}: feature_res {base["model"]["backbone"]["feature_res"]}, wrote {len(NOISE) * len(STEPS)} configs to {out}')
