"""OmiAD stage 2: Adversarial Score Distillation (ASD) of a pre-trained AMDM teacher into a
one-step generator (Feng et al., ICML 2025, Sec. 3.3 + Alg. 1). The official repository
(github.com/luolundashu/OmiAD) is README-only, so this is a re-implementation from the paper,
built on the AMDM teacher reproduction in this directory (``amdm.py`` / ``train_amdm.py``).

Networks (all the same guided-diffusion U-Net, x0-parameterized as in ``amdm.py``):
  phi    frozen AMDM teacher                              f_phi(x_t, t)
  psi    student ("fake") score network, init <- phi      f_psi(x_t, t)
  D      discriminator = encoder (input blocks + middle block) of psi, shared weights;
         its (C, H', W') output is averaged over channels -> (H', W') logit map (Sec. 3.3.2);
         asd.disc_norm: rms (default) rescales that output by its per-sample RMS first, none = literal
  theta  one-step generator, init <- phi:
         x_g = g_theta(sqrt(ab_tinit) x_0 + sqrt(1 - ab_tinit) eps), time embedding fixed to t_init

One iteration (Alg. 1), t ~ U{0..t_max-1}, x_t = sqrt(ab_t) x_g + sqrt(1-ab_t) eps_t,
y_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps_t (same eps_t):
  psi step:   L_psi = ||f_psi(x_t,t) - x_g||^2                                (Eq. 13, x0 form)
                      + lambda_adv_psi * mean[softplus(-D(y_t)) + softplus(D(x_t))]   (Eq. 17)
  theta step: L_theta = mean[(f_phi - f_psi) * ((f_phi - x_g) - alpha (f_phi - f_psi))] / mean|x_g - f_phi|
                      + lambda_adv_theta * mean[softplus(-D(x_t))]            (Eq. 15/27 + Eq. 18)

Notes on the choices the paper leaves open:
  * Eq. 15 is written with eps-networks. With eps = (x_t - sqrt(ab) f) / sqrt(1-ab) it equals
    ab/(1-ab) * [(f_phi - f_psi)(f_phi - x_g) - alpha ||f_phi - f_psi||^2]; the t-dependent factor is
    absorbed into gamma(t), for which we use SiD's L1 normaliser 1 / mean|x_g - f_phi| (stop-grad).
  * As in SiD, the generator gradient flows through x_t into both (frozen-parameter) score nets.
  * Eq. 17/18 are implemented as the usual non-saturating logistic GAN losses on the logit map (SiDA).
  * Optimiser: Adam(lr 1e-5, betas (0, 0.999)) for theta and psi (SiD / SiDA defaults; the paper
    only states the AMDM optimiser). No masking in this stage.

The saved generator is an ordinary ``AMDM`` state dict, so inference is
``AMDM.reconstruct(x0, t_init, steps=1)`` = Alg. 2, and evaluation reuses ``evaluate_amdm``
(FlowAD 7-metric mAD protocol) with ``recon_steps: 1``.

Run from the repo root (single process, single GPU):
  CUDA_VISIBLE_DEVICES=0 python baselines/amdm_repro/train_omiad.py \
      --config_file baselines/amdm_repro/configs/omiad_btad.yaml
"""

import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.getcwd())

from src.backbones import BackboneWrapper, get_normalization_func
from src.backbones.efficientnet import build_efficient
from src.datasets import build_dataset

from baselines.amdm_repro.amdm import evaluate_amdm, write_results_csv
from baselines.amdm_repro.guided_diffusion.nn import timestep_embedding
from baselines.amdm_repro.guided_diffusion.unet import AttentionBlock
from baselines.amdm_repro.train_amdm import build_amdm

# guided-diffusion's AttentionBlock always runs under its gradient-checkpoint Function, whose backward
# fails when the block's parameters are frozen but its input needs a gradient (teacher / psi in the
# generator step). Calling _forward directly is numerically identical.
AttentionBlock.forward = lambda self, x: self._forward(x)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('omiad')


DISC_NORM = 'rms'   # set from asd.disc_norm in main()


def unet_forward(net, x, t, with_logits=False, encoder_only=False):
    """guided-diffusion UNetModel.forward that can also return the shared-weight discriminator
    logit map (channel mean of the encoder/bottleneck output)."""
    hs = []
    emb = net.time_embed(timestep_embedding(t, net.model_channels))
    h = x
    for module in net.input_blocks:
        h = module(h, emb)
        hs.append(h)
    h = net.middle_block(h, emb)
    if DISC_NORM == 'rms':
        # parameter-free rescaling: the bottleneck residual stream of this U-Net is un-normalised
        # (|h| ~ 1e2-1e3 on raw EfficientNet features), so its plain channel mean gives logits in the
        # hundreds / thousands and the GAN terms swamp the distillation loss. Dividing by the per-sample
        # RMS and scaling the channel mean by sqrt(C) gives O(1) logits without adding parameters.
        hn = h / h.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
        logits = hn.mean(dim=1) * (h.shape[1] ** 0.5)
    else:
        logits = h.mean(dim=1)                  # (B, H', W'), literal Sec. 3.3.2
    if encoder_only:
        return logits
    for module in net.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb)
    out = net.out(h)
    return (out, logits) if with_logits else out


def load_amdm_weights(model, path, key):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    key = key if key in ck else 'model'
    model.load_state_dict({k.replace('module.', ''): v for k, v in ck[key].items()})
    return ck.get('epoch', -1), key


def main(params, args):
    device = torch.device('cuda')
    log_dir = params['logging']['log_dir']
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(params, f)

    seed = params['opt']['global_seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # -- data
    train_bs = params['data']['train_batch_size']
    test_bs = params['data'].get('test_batch_size', train_bs)
    train_dataset = build_dataset(train=True, **params['data'])
    test_dataset = build_dataset(train=False, **params['data'])
    nw = params['data'].get('num_workers', 4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False,
                                              num_workers=nw, pin_memory=True, drop_last=False)
    logger.info(f"[OmiAD] train: {len(train_dataset)}, test: {len(test_dataset)}")

    # -- frozen backbone (identical to train_amdm.py)
    bb = params['model']['backbone']
    fs = params['model']['feature_size']
    fe = BackboneWrapper(build_efficient(model_name=bb['model_name'], pretrained=bb['pretrained'],
                                         outblocks=bb['outblocks'], outstrides=bb['outstrides']),
                         target_size=(fs, fs))
    norm_fn = get_normalization_func(bb.get('normalization', None))
    fe.to(device).eval()
    for p in fe.parameters():
        p.requires_grad = False

    # -- networks: theta <- phi, psi <- phi
    asd = params['asd']
    teacher = build_amdm(params).to(device).eval()
    t_epoch, t_key = load_amdm_weights(teacher, asd['teacher_path'], asd.get('teacher_key', 'ema'))
    logger.info(f"[OmiAD] teacher {asd['teacher_path']} [{t_key}, epoch {t_epoch}]")
    for p in teacher.parameters():
        p.requires_grad = False
    gen = copy.deepcopy(teacher).train()
    fake = copy.deepcopy(teacher).train()
    for m in (gen, fake):
        for p in m.parameters():
            p.requires_grad = True
    gen_ema = copy.deepcopy(teacher).eval()

    sab, s1mab = teacher.sqrt_ab, teacher.sqrt_1mab
    t_init = int(asd.get('t_init', 960))
    ti = t_init - 1                                  # 0-based index, as AMDM.reconstruct(steps=1)
    t_max = int(asd.get('t_max', 990))
    alpha = float(asd.get('alpha', 1.0))
    lam_g = float(asd.get('lambda_adv_theta', 1.0))
    lam_d = float(asd.get('lambda_adv_psi', 0.01))
    ema_rate = float(asd.get('ema_rate', 0.999))
    global DISC_NORM
    DISC_NORM = asd.get('disc_norm', 'rms')
    logger.info(f"[OmiAD] t_init={t_init} (sqrt(ab)={sab[ti].item():.4f}), t_max={t_max}, alpha={alpha}, "
                f"lambda_adv_theta={lam_g}, lambda_adv_psi={lam_d}, ema={ema_rate}, disc_norm={DISC_NORM}")

    opt = params['opt']
    betas = tuple(opt.get('betas', [0.0, 0.999]))
    opt_g = torch.optim.Adam(gen.parameters(), lr=float(opt['lr_g']), betas=betas, eps=1e-8)
    opt_f = torch.optim.Adam(fake.parameters(), lr=float(opt['lr_psi']), betas=betas, eps=1e-8)
    clip = float(opt.get('clip_grad', 0.0))
    num_epochs = opt['epochs']

    def generate(x0, net):
        b = x0.shape[0]
        x_in = sab[ti] * x0 + s1mab[ti] * torch.randn_like(x0)
        return unet_forward(net.model, x_in, torch.full((b,), float(ti), device=device))

    def diffuse(x, t, eps):
        return sab[t].view(-1, 1, 1, 1) * x + s1mab[t].view(-1, 1, 1, 1) * eps

    eval_params = params['logging']['eval']
    eval_freq = eval_params['eval_epoch_freq']
    img_sz = (params['data']['img_size'], params['data']['img_size'])

    def run_eval(model, ep=None):
        return evaluate_amdm(vf=model, fe=fe, norm_fn=norm_fn, dataloader=test_loader, device=device,
                             img_sz=img_sz, verbose=False, distributed=False, eval_params=ep or eval_params)

    def save_ckpt(path, epoch):
        torch.save({'epoch': epoch, 'model': gen.state_dict(), 'ema': gen_ema.state_dict()}, path)

    HIST = ['mad', 'img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_aupro', 'px_ap', 'px_f1max']
    hist_paths = {k: os.path.join(log_dir, f'mad_history{s}.csv') for k, s in (('raw', ''), ('ema', '_ema'))}

    def append_history(which, epoch, avg):
        hp = hist_paths[which]
        if not os.path.exists(hp):
            with open(hp, 'w') as f:
                f.write('epoch,' + ','.join(HIST) + '\n')
        with open(hp, 'a') as f:
            f.write(f"{epoch}," + ','.join(f"{avg.get(k, 0):.6f}" for k in HIST) + '\n')

    best_ckpt = os.path.join(ckpt_dir, 'ckpt_best.pth')
    final_ckpt = os.path.join(ckpt_dir, f'ckpt_epoch_{num_epochs}.pth')

    if not args.eval_only:
        # -- training latents, extracted once and kept on the GPU (deterministic transforms)
        lat = []
        ex_loader = torch.utils.data.DataLoader(train_dataset, batch_size=test_bs, shuffle=False, num_workers=nw)
        with torch.no_grad():
            for batch in ex_loader:
                z, _ = fe(batch['img'].to(device))
                lat.append(norm_fn(z))
        lat = torch.cat(lat, dim=0)
        N = lat.shape[0]
        ipe = N // train_bs
        logger.info(f"[OmiAD] latents {tuple(lat.shape)}, mean {lat.mean().item():.3f}, std {lat.std().item():.3f}, "
                    f"{ipe} it/epoch")

        # epoch 0 = the teacher's own one-step prediction at t_init (the generator's initialisation)
        r0 = run_eval(gen_ema)
        logger.info(f"[OmiAD] Epoch 0 (init = teacher one-step) mAD: {r0['average']['mad']:.4f} | {r0['average']}")
        append_history('raw', 0, r0['average'])
        append_history('ema', 0, r0['average'])

        best_mad = 0.0
        g_cpu = torch.Generator().manual_seed(seed)
        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            perm = torch.randperm(N, generator=g_cpu).to(device)
            meter = {k: 0.0 for k in ('psi_dn', 'psi_adv', 'g_sid', 'g_adv', 'd_real', 'd_fake')}
            for it in range(ipe):
                x0 = lat[perm[it * train_bs:(it + 1) * train_bs]]
                B = x0.shape[0]

                # ---------------- psi (student score + discriminator) step
                with torch.no_grad():
                    xg = generate(x0, gen)
                t = torch.randint(0, t_max, (B,), device=device)
                eps = torch.randn_like(x0)
                f_psi, logit_fake = unet_forward(fake.model, diffuse(xg, t, eps), t.float(), with_logits=True)
                logit_real = unet_forward(fake.model, diffuse(x0, t, eps), t.float(), encoder_only=True)
                l_dn = F.mse_loss(f_psi, xg)
                l_adv_d = F.softplus(-logit_real).mean() + F.softplus(logit_fake).mean()
                loss_psi = l_dn + lam_d * l_adv_d
                opt_f.zero_grad(set_to_none=True)
                loss_psi.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(fake.parameters(), clip)
                opt_f.step()

                # ---------------- theta (one-step generator) step
                for p in fake.parameters():
                    p.requires_grad = False
                xg = generate(x0, gen)
                t = torch.randint(0, t_max, (B,), device=device)
                xt = diffuse(xg, t, torch.randn_like(x0))
                f_phi = unet_forward(teacher.model, xt, t.float())
                f_psi, logit_fake = unet_forward(fake.model, xt, t.float(), with_logits=True)
                with torch.no_grad():
                    w = (xg - f_phi).abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
                d = f_phi - f_psi
                l_sid = (d * ((f_phi - xg) - alpha * d) / w).mean()
                l_adv_g = F.softplus(-logit_fake).mean()
                loss_g = l_sid + lam_g * l_adv_g
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(gen.parameters(), clip)
                opt_g.step()
                for p in fake.parameters():
                    p.requires_grad = True

                with torch.no_grad():
                    for pe, p in zip(gen_ema.parameters(), gen.parameters()):
                        pe.mul_(ema_rate).add_(p, alpha=1.0 - ema_rate)

                for k, v in (('psi_dn', l_dn), ('psi_adv', l_adv_d), ('g_sid', l_sid), ('g_adv', l_adv_g),
                             ('d_real', logit_real.mean()), ('d_fake', logit_fake.mean())):
                    meter[k] += v.item() / ipe
                if not np.isfinite(loss_g.item() + loss_psi.item()):
                    raise RuntimeError(f"non-finite loss at epoch {epoch} it {it}")

            logger.info(f"[OmiAD] Epoch [{epoch}/{num_epochs}] " + ' '.join(f"{k} {v:.4f}" for k, v in meter.items())
                        + f" | {(time.time() - t0) / epoch:.1f}s/ep")

            if epoch % eval_freq == 0 or epoch == num_epochs:
                gen.eval()
                raw = run_eval(gen)
                gen.train()
                ema = run_eval(gen_ema)
                append_history('raw', epoch, raw['average'])
                append_history('ema', epoch, ema['average'])
                logger.info(f"[OmiAD] Epoch {epoch} mAD raw: {raw['average']['mad']:.4f} | ema: {ema['average']['mad']:.4f}")
                logger.info(f"[OmiAD] Epoch {epoch} ema average: {ema['average']}")
                if ema['average']['mad'] > best_mad:      # FlowAD protocol: best checkpoint on average.mad
                    best_mad = ema['average']['mad']
                    save_ckpt(best_ckpt, epoch)
                    write_results_csv(ema, os.path.join(log_dir, 'eval_results_best.csv'))
                    logger.info(f"[OmiAD] new best EMA mAD {best_mad:.4f} at epoch {epoch}")
        save_ckpt(final_ckpt, num_epochs)

    # -- final evaluation: last-epoch and best checkpoints, EMA + raw weights, each scoring variant
    variants = eval_params.get('eval_variants') or [{}]
    ckpts = [('final', args.eval_ckpt or final_ckpt)] + ([('best', best_ckpt)] if not args.eval_ckpt else [])
    for name, path in ckpts:
        if not os.path.isfile(path):
            continue
        ck = torch.load(path, map_location='cpu', weights_only=True)
        for key, tag in (('ema', ''), ('model', '_raw')):
            gen_ema.load_state_dict(ck[key])
            for var in variants:
                ep = {**eval_params, **var}
                res = run_eval(gen_ema, ep)
                vtag = f"_{ep['img_score_agg'].replace('+', '')}" if var else ''
                out = os.path.join(log_dir, f'eval_results_{name}{tag}{vtag}')
                write_results_csv(res, out + '.csv')
                logger.info(f"[OmiAD] {name.upper()}{tag}{vtag} (epoch {ck['epoch']}) mAD: {res['average']['mad']:.4f} "
                            f"| {res['average']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmiAD adversarial score distillation (re-implementation)")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_ckpt", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        params = yaml.safe_load(f)
    main(params, args)
