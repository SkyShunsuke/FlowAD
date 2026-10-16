"""AMDM reproduction driver (multi-class MVTec-AD / VisA / MPDD).

Original AMDM training kept as in the OmiAD paper (Appendix F): DDPM T=1000
linear schedule on EfficientNet-b4 features resized to 32x32 (272 ch, 224x224
input), adaptive masking (pmin 0.1, pmax 0.4, k 2), guided-diffusion U-Net,
plain Adam lr 1e-3, global batch 32, fp32. Inference: t_init=960 perturbation
+ 40-step DDIM reconstruction (no ASD stage).

Evaluation: FlowAD 7-metric mAD protocol (mirrors src/vfad/eval_recon.py),
periodic eval with best-checkpoint selection on test mAD (src/vfad/train.py
protocol, as in the other baseline reproductions).

Run from the repo root, e.g.:
  CUDA_VISIBLE_DEVICES=0,1,2,6 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --nnodes=1 --nproc-per-node=4 baselines/amdm_repro/train_amdm.py \
    --config_file baselines/amdm_repro/configs/amdm_mvtec.yaml
"""

import os
import sys
import copy
import argparse

import torch
import torch.distributed as dist
import numpy as np
import yaml
import logging

sys.path.insert(0, os.getcwd())

from src.backbones import BackboneWrapper, get_normalization_func
from src.backbones.efficientnet import build_efficient
from src.datasets import build_dataset
from src.utils.distributed import init_distributed_mode, get_rank, get_world_size
from src.utils.distributed import is_main_process as is_main
from src.utils.log import setup_logging, get_logger, AverageMeter
from src.utils.opt.optimizer import load_model_only

from baselines.amdm_repro.amdm import AMDM, evaluate_amdm, write_results_csv
from baselines.amdm_repro.guided_diffusion.unet import UNetModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="AMDM (OmiAD teacher) reproduction")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load ckpt_best.pth and run the final evaluation.")
    parser.add_argument("--eval_ckpt", type=str, default=None,
                        help="Checkpoint to evaluate with --eval_only (default: ckpt_best.pth).")
    # torchrun / distributed plumbing (same as main.py)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def build_amdm(params):
    m = params['model']
    net = UNetModel(
        image_size=m['feature_size'],
        in_channels=m['in_channels'],
        model_channels=m['model_channels'],
        out_channels=m['in_channels'],
        num_res_blocks=m['num_res_blocks'],
        attention_resolutions=tuple(m['attention_ds']),
        dropout=m.get('dropout', 0.0),
        channel_mult=tuple(m['channel_mult']),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=m.get('num_heads', 4),
        num_head_channels=m.get('num_head_channels', -1),
        use_scale_shift_norm=m.get('use_scale_shift_norm', True),
        resblock_updown=m.get('resblock_updown', False),
        use_new_attention_order=False,
    )
    d = params['diffusion']
    return AMDM(
        net,
        num_timesteps=d['num_timesteps'],
        beta_start=d['beta_start'],
        beta_end=d['beta_end'],
        p_min=d['p_min'],
        p_max=d['p_max'],
        k=d['k'],
    )


def main(params, args):
    init_distributed_mode(args)
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device('cuda:%s' % args.gpu)

    setup_logging(rank, world_size)
    logger = get_logger()
    logger.info(f"[AMDM] device: {device}, rank: {rank}, world_size: {world_size}")

    log_dir = params['logging']['log_dir']
    ckpt_logdir = os.path.join(log_dir, 'checkpoints')
    if is_main():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_logdir, exist_ok=True)
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(params, f)

    seed = params['opt']['global_seed'] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # -- data
    train_bs = params['data']['train_batch_size']
    test_bs = params['data'].get('test_batch_size', train_bs)
    train_dataset = build_dataset(train=True, **params['data'])
    test_dataset = build_dataset(train=False, **params['data'])
    num_classes = len(train_dataset.datasets) if isinstance(train_dataset, torch.utils.data.ConcatDataset) else 1
    logger.info(f"[AMDM] num_classes: {num_classes}, train: {len(train_dataset)}, test: {len(test_dataset)}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=train_bs,
        pin_memory=params['data'].get('pin_memory', True),
        num_workers=params['data'].get('num_workers', 4),
        persistent_workers=params['data'].get('persistent_workers', False),
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, sampler=test_sampler, batch_size=test_bs,
        pin_memory=params['data'].get('pin_memory', True),
        num_workers=params['data'].get('num_workers', 4),
        persistent_workers=params['data'].get('persistent_workers', False),
        drop_last=False,
    )
    ipe = len(train_loader)

    # -- frozen backbone: EfficientNet-b4 stages 1-4 -> resize 32x32 -> concat (272ch)
    bb = params['model']['backbone']
    net = build_efficient(
        model_name=bb['model_name'], pretrained=bb['pretrained'],
        outblocks=bb['outblocks'], outstrides=bb['outstrides'],
    )
    fs = params['model']['feature_size']
    fe = BackboneWrapper(net, target_size=(fs, fs))
    norm_fn = get_normalization_func(bb.get('normalization', None))
    fe.to(device).eval()
    for p in fe.parameters():
        p.requires_grad = False
    logger.info(f"[AMDM] backbone ready, latent shape (272, {fs}, {fs})")

    # -- AMDM model
    vf = build_amdm(params).to(device)
    n_params = sum(p.numel() for p in vf.parameters())
    logger.info(f"[AMDM] U-Net parameters: {n_params/1e6:.1f}M")

    opt_params = params['opt']
    num_epochs = opt_params['epochs']
    optimizer = torch.optim.Adam(vf.parameters(), lr=opt_params['lr'])

    # EMA of the weights (guided-diffusion trains with EMA 0.9999 and evaluates
    # the EMA parameters; pre-committed as the primary eval weights here)
    ema_rate = opt_params.get('ema_rate', 0.9999)
    ema_model = copy.deepcopy(vf).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    # -- optional resume (model + optimizer + EMA), for extending a finished run
    start_epoch = 0
    resume_path = (params.get('resume') or {}).get('resume_path')
    if resume_path:
        ck = torch.load(resume_path, map_location='cpu', weights_only=True)
        sd = {k.replace('module.', ''): v for k, v in ck['model'].items()}
        vf.load_state_dict(sd)
        optimizer.load_state_dict(ck['opt'])
        for g in optimizer.param_groups:   # keep the CONFIG lr, not the saved one
            g['lr'] = opt_params['lr']
        if 'ema' in ck:
            esd = {k.replace('module.', ''): v for k, v in ck['ema'].items()}
            ema_model.load_state_dict(esd)
        else:
            ema_model.load_state_dict(sd)  # cold-start EMA from resumed weights
        start_epoch = ck['epoch']
        logger.info(f"[AMDM] resumed from {resume_path} at epoch {start_epoch}")

    if args.distributed:
        vf = torch.nn.parallel.DistributedDataParallel(vf, static_graph=True)
    inner = vf.module if args.distributed else vf

    @torch.no_grad()
    def ema_update():
        for pe, p in zip(ema_model.parameters(), inner.parameters()):
            pe.mul_(ema_rate).add_(p, alpha=1.0 - ema_rate)

    eval_params = params['logging']['eval']
    eval_freq = eval_params['eval_epoch_freq']
    # RealIAD protocol: no intermediate eval — train all epochs, single final
    # eval (also: a 114k-image test pass every 50 epochs would be prohibitive)
    eval_during_training = eval_params.get('eval_during_training', True)
    img_sz = (params['data']['img_size'], params['data']['img_size'])
    best_ckpt_path = os.path.join(ckpt_logdir, 'ckpt_best.pth')
    last_ckpt_path = os.path.join(ckpt_logdir, 'ckpt_last.pth')

    def save_ckpt(path, epoch, model):
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'ema': ema_model.state_dict(),
                    'opt': optimizer.state_dict()}, path)
        logger.info(f"[AMDM] saved checkpoint at epoch {epoch} to {path}")

    if not args.eval_only:
        # -- pre-extract all training latents (as src/vfad/train.py does;
        #    transforms are deterministic, so this is exact)
        vf.train()
        from tqdm import tqdm

        def _extract_loader():
            return torch.utils.data.DataLoader(
                train_dataset, sampler=None, batch_size=test_bs,
                pin_memory=False, num_workers=params['data'].get('num_workers', 4),
                persistent_workers=False, drop_last=False,
            )

        cache_mode = params['data'].get('latent_cache', 'ram')
        if cache_mode == 'disk':
            # RealIAD-scale: the in-RAM dict costs ~1.1MB/img PER RANK. Instead
            # rank 0 extracts once into a float32 memmap that all ranks share
            # read-only (page cache keeps resident memory bounded and shared).
            import json
            cache_dir = os.path.join(log_dir, 'latent_cache')
            lat_npy = os.path.join(cache_dir, 'latents.npy')
            paths_json = os.path.join(cache_dir, 'paths.json')
            if is_main() and not (os.path.isfile(lat_npy) and os.path.isfile(paths_json)):
                os.makedirs(cache_dir, exist_ok=True)
                mm, row, path2row = None, 0, {}
                for batch in tqdm(_extract_loader()):
                    with torch.no_grad():
                        z1, _ = fe(batch["img"].to(device))
                        z1 = norm_fn(z1)
                    z1 = z1.detach().cpu().numpy().astype(np.float32)
                    if mm is None:
                        mm = np.lib.format.open_memmap(
                            lat_npy + '.tmp', mode='w+', dtype=np.float32,
                            shape=(len(train_dataset), *z1.shape[1:]))
                    for i, path in enumerate(batch["path"]):
                        mm[row] = z1[i]
                        path2row[path] = row
                        row += 1
                assert row == len(train_dataset), (row, len(train_dataset))
                mm.flush()
                del mm
                with open(paths_json + '.tmp', 'w') as f:
                    json.dump(path2row, f)
                os.replace(lat_npy + '.tmp', lat_npy)
                os.replace(paths_json + '.tmp', paths_json)
            dist.barrier()
            with open(paths_json) as f:
                path2row = json.load(f)
            lat_mm = np.load(lat_npy, mmap_mode='r')
            logger.info(f"[AMDM] latent memmap cache ready: {lat_mm.shape} at {lat_npy}")

            def get_latents(paths):
                rows = np.asarray([path2row[p] for p in paths])
                return torch.from_numpy(np.ascontiguousarray(lat_mm[rows]))
        else:
            all_latents = {}
            for batch in tqdm(_extract_loader(), disable=not is_main()):
                imgs = batch["img"]
                with torch.no_grad():
                    z1, _ = fe(imgs.to(device))
                    z1 = norm_fn(z1)
                for i, path in enumerate(batch["path"]):
                    all_latents[path] = z1[i].detach().cpu()
            logger.info(f"[AMDM] extracted latents for {len(all_latents)} training samples")

            def get_latents(paths):
                return torch.stack([all_latents[p] for p in paths], dim=0)

        best_mad = 0.0
        HIST_HEADER = 'epoch,mad,img_auroc,img_ap,img_f1max,px_auroc,px_aupro,px_ap,px_f1max\n'
        history_path = os.path.join(log_dir, 'mad_history.csv')
        history_ema_path = os.path.join(log_dir, 'mad_history_ema.csv')
        if is_main():
            for hp in (history_path, history_ema_path):
                if not os.path.exists(hp):
                    with open(hp, 'w') as f:
                        f.write(HIST_HEADER)

        def append_history(hp, epoch, avg):
            with open(hp, 'a') as f:
                f.write(f"{epoch},{avg['mad']:.6f},{avg.get('img_auroc',0):.6f},{avg.get('img_ap',0):.6f},"
                        f"{avg.get('img_f1max',0):.6f},{avg.get('px_auroc',0):.6f},{avg.get('px_aupro',0):.6f},"
                        f"{avg.get('px_ap',0):.6f},{avg.get('px_f1max',0):.6f}\n")

        for epoch in range(start_epoch + 1, num_epochs + 1):
            train_sampler.set_epoch(epoch)
            loss_meter = AverageMeter()

            for step, batch in enumerate(train_loader):
                z1 = get_latents(batch["path"]).to(device)

                loss = vf(z1)

                optimizer.zero_grad()
                loss.backward()
                # stabilizer (config-gated, off by default): the paper's Adam
                # lr 1e-3 collapsed to a constant predictor on VisA without it
                if opt_params.get('clip_grad', 0.0) > 0.0:
                    torch.nn.utils.clip_grad_norm_(vf.parameters(), opt_params['clip_grad'])
                optimizer.step()
                ema_update()

                loss_meter.step(loss.item())
                if step % params['logging']['log_step_freq'] == 0:
                    logger.info(f"[AMDM] Epoch [{epoch}/{num_epochs}] Step [{step}/{ipe}] "
                                f"Loss: {loss_meter.val:.5f} ({loss_meter.avg:.5f})")

            logger.info(f"[AMDM] Epoch [{epoch}/{num_epochs}] done. Avg loss: {loss_meter.avg:.5f}")

            if eval_during_training and (epoch % eval_freq == 0 or epoch == num_epochs):
                raw_results = evaluate_amdm(
                    vf=vf, fe=fe, norm_fn=norm_fn, dataloader=test_loader, device=device,
                    img_sz=img_sz, verbose=is_main(),
                    distributed=args.distributed, eval_params=eval_params,
                )
                ema_results = evaluate_amdm(
                    vf=ema_model, fe=fe, norm_fn=norm_fn, dataloader=test_loader, device=device,
                    img_sz=img_sz, verbose=is_main(),
                    distributed=args.distributed, eval_params=eval_params,
                )
                mad_raw = raw_results['average']['mad']
                mad_ema = ema_results['average']['mad']
                logger.info(f"[AMDM] Epoch {epoch} mAD raw: {mad_raw:.4f} | ema: {mad_ema:.4f}")
                logger.info(f"[AMDM] Epoch {epoch} ema average: {ema_results['average']}")
                if is_main():
                    append_history(history_path, epoch, raw_results['average'])
                    append_history(history_ema_path, epoch, ema_results['average'])
                    save_ckpt(last_ckpt_path, epoch, vf)
                    # primary model selection: EMA mAD (pre-committed)
                    if mad_ema > best_mad:
                        best_mad = mad_ema
                        save_ckpt(best_ckpt_path, epoch, vf)
                        write_results_csv(ema_results, os.path.join(log_dir, 'eval_results_best.csv'))
                        with open(os.path.join(log_dir, 'eval_results_best.yaml'), 'w') as f:
                            yaml.dump({'epoch': epoch, 'results': ema_results,
                                       'raw_results': raw_results}, f)
                        logger.info(f"[AMDM] new best EMA mAD {best_mad:.4f} at epoch {epoch}, checkpoint saved")

            if not eval_during_training and is_main():
                # without in-training eval there is no best-ckpt path; keep a
                # rolling last + periodic numbered snapshots (collapse forensics)
                ckpt_freq = params['logging'].get('ckpt_epoch_freq', 25)
                if ckpt_freq and epoch % ckpt_freq == 0:
                    save_ckpt(last_ckpt_path, epoch, vf)
                snap_freq = params['logging'].get('ckpt_snapshot_freq', 0)
                if snap_freq and epoch % snap_freq == 0:
                    save_ckpt(os.path.join(ckpt_logdir, f'ckpt_epoch_{epoch}.pth'), epoch, vf)
            dist.barrier()

        if is_main():
            save_ckpt(os.path.join(ckpt_logdir, f'ckpt_epoch_{num_epochs}.pth'), num_epochs, vf)
        dist.barrier()

    # -- final evaluation of the best checkpoint (EMA weights primary, raw secondary)
    eval_ckpt = args.eval_ckpt or best_ckpt_path
    if not os.path.isfile(eval_ckpt):  # no in-training eval -> no ckpt_best
        for cand in (os.path.join(ckpt_logdir, f'ckpt_epoch_{num_epochs}.pth'), last_ckpt_path):
            if os.path.isfile(cand):
                eval_ckpt = cand
                break
    assert os.path.isfile(eval_ckpt), f"checkpoint not found: {eval_ckpt}"
    ck = torch.load(eval_ckpt, map_location='cpu', weights_only=True)
    # optional logging.eval.eval_variants: [{t_init, recon_steps}, ...] (e.g. FlowAD one-step
    # plug-in, recon_steps=1); each writes eval_results_final{tag}_t{t_init}_s{steps}.*
    variants = eval_params.get('eval_variants') or [None]
    weights = [tk for tk in (('', 'ema'), ('_raw', 'model')) if tk[1] in eval_params.get('eval_weights', ['ema', 'model'])]
    for (tag, key), var in [(tk, v) for v in variants for tk in weights]:
        if key not in ck:
            continue
        sd = {k.replace('module.', ''): v for k, v in ck[key].items()}
        ema_model.load_state_dict(sd)
        ep = eval_params if var is None else {**eval_params, **var}
        if var is not None:
            tag = f"{tag}_t{ep['t_init']}_s{ep['recon_steps']}"
        eval_results = evaluate_amdm(
            vf=ema_model, fe=fe, norm_fn=norm_fn, dataloader=test_loader, device=device,
            img_sz=img_sz, verbose=is_main(),
            distributed=args.distributed, eval_params=ep,
        )
        if is_main():
            write_results_csv(eval_results, os.path.join(log_dir, f'eval_results_final{tag}.csv'))
            with open(os.path.join(log_dir, f'eval_results_final{tag}.yaml'), 'w') as f:
                yaml.dump({'epoch': ck['epoch'], 'results': eval_results}, f)
            logger.info(f"[AMDM] FINAL{tag} (epoch {ck['epoch']}) mAD: {eval_results['average']['mad']:.4f}")
            logger.info(f"[AMDM] FINAL{tag} average: {eval_results['average']}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_file, 'r') as f:
        params = yaml.safe_load(f)
    main(params, args)
