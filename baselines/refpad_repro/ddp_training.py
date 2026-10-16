"""Multi-GPU (DDP-style) replacements for ReFP-AD's train_flow / train_ebm.

User-authorized deviation from the "unmodified ReFP-AD" rule (2026-09-22):
training is parallelized across GPUs. The ReFP-AD clone itself stays
untouched — these functions have the SAME signatures as training.train_flow /
training.train_ebm and are monkey-patched into the `pipeline` module by
run_repro_enet_realiad_ddp.py.

Design — semantics preserved as closely as possible vs single-GPU:
- The GLOBAL batch is kept at cfg.flow_batch / cfg.batch_size (per-rank batch
  = global / world_size), so the number of optimizer steps per epoch, the LR
  schedules, and the EMA update count are identical to the single-GPU run.
- No torch DDP wrapper: each rank does forward/backward on the raw module and
  gradients are explicitly all-reduced (AVG) before every optimizer step.
  This keeps the modules plain (checkpoint save/load unchanged) and sidesteps
  DDP's create_graph=True limitation (the EBM gradient penalty is a double
  backward). All ranks start from identical weights (same seed_all at build)
  and apply identical updates, so parameters stay bit-synchronized.
- Per-rank RNG streams are decorrelated inside the training loops (data order
  via DistributedSampler, flow-matching t/noise, SGLD noise, replay buffers),
  AFTER the identically-seeded model construction.
- Known semantic deltas vs single-GPU (documented, unavoidable):
  * The minibatch-OT pairing in rf_loss is computed per rank over
    global_batch/world samples instead of the full batch (finer-grained
    Sinkhorn coupling). This also makes each step much cheaper (Sinkhorn is
    O(B^2)).
  * The EBM replay buffer is sharded: each rank keeps an independent buffer
    of cfg.buffer_size / world entries (total pool size and refresh rate
    match the single-GPU run).
- Flow diagnostics / health log / best-checkpoint selection run on rank 0
  only (they are observers); the chosen weights are broadcast to all ranks
  at the end. Early stopping decisions are broadcast so ranks stay in step.
"""

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# ReFP-AD modules (sys.path prepared by run_repro import in the runner)
from config import CFG  # noqa: E402
from flow import RectifiedFlow  # noqa: E402
from ebm import ReplayBuffer, precond_sgld, vanilla_sgld  # noqa: E402
from token_stats import TokenStats  # noqa: E402
from flow_diagnostics import (  # noqa: E402
    FlowHealthLog,
    pairwise_distance_correlation,
    sgld_fitness,
)

NUM_WORKERS = 6  # per rank


def _allreduce_grads(module: torch.nn.Module):
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def _broadcast_module(module: torch.nn.Module, src: int = 0):
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def _allreduce_scalar(x: float, device) -> float:
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def train_flow_ddp(
    flow: RectifiedFlow,
    tokens_cpu: torch.Tensor,
    stats: TokenStats,
    cfg: CFG,
    label: str = "Flow",
) -> Tuple[RectifiedFlow, Optional[FlowHealthLog]]:
    """Drop-in DDP replacement for training.train_flow (same signature)."""
    rank, world = dist.get_rank(), dist.get_world_size()
    r0 = rank == 0
    device = cfg.device
    assert not cfg.flow_amp, "flow_amp not supported in DDP mode (GradScaler skip-steps would desync ranks)"
    assert cfg.flow_batch % world == 0, (cfg.flow_batch, world)

    if r0:
        print(f"[-] Training {label} | max_epochs={cfg.flow_epochs}  "
              f"patience={cfg.flow_patience}")
        print(f"    diag_every={cfg.flow_diag_every}")
        print(f"    [DDP] world_size={world}  global_batch={cfg.flow_batch} "
              f"(per-rank {cfg.flow_batch // world})")

    flow.train()
    opt = torch.optim.Adam(flow.parameters(), lr=cfg.flow_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.flow_epochs)

    if tokens_cpu.shape[0] > cfg.max_fit_tokens:
        g_sub = torch.Generator()
        g_sub.manual_seed(int(cfg.seed) + 777)  # identical subset on all ranks
        idx = torch.randperm(tokens_cpu.shape[0], generator=g_sub)[: cfg.max_fit_tokens]
        tokens_cpu = tokens_cpu[idx]

    ds = TensorDataset(tokens_cpu)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                 shuffle=True, seed=int(cfg.seed), drop_last=True)
    dl = DataLoader(ds, batch_size=cfg.flow_batch // world, sampler=sampler,
                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                    persistent_workers=True)

    # rank-0-only diagnostics (identical to training.train_flow)
    health = None
    z_diag = None
    if r0:
        n_diag = min(cfg.flow_diag_samples, tokens_cpu.shape[0])
        g = torch.Generator()
        g.manual_seed(int(cfg.seed) + 12345)
        diag_idx = torch.randperm(tokens_cpu.shape[0], generator=g)[:n_diag]
        with torch.no_grad():
            z_diag = stats.normalize(tokens_cpu[diag_idx].to(device, non_blocking=True))
        health = FlowHealthLog(patience=cfg.flow_patience,
                               min_dist_corr=cfg.flow_min_dist_corr,
                               min_epochs=cfg.flow_min_epochs)

    best_state = None
    best_epoch: Optional[int] = None
    has_ema = (getattr(flow, "ema", None) is not None
               and getattr(flow, "ema_update", None) is not None)

    # decorrelate per-rank RNG (t, flow noise, OT) AFTER identical construction
    torch.manual_seed(int(cfg.seed) * 1000 + rank + 1)
    torch.cuda.manual_seed_all(int(cfg.seed) * 1000 + rank + 1)

    for ep in range(cfg.flow_epochs):
        sampler.set_epoch(ep)
        acc_rf, steps = 0.0, 0

        for (batch_cpu,) in dl:
            z0 = stats.normalize(batch_cpu.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss_rf = flow.rf_loss(z0)
            loss_rf.backward()
            _allreduce_grads(flow)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            if has_ema:
                flow.ema_update()
            acc_rf += float(loss_rf.item())
            steps += 1

        sched.step()
        avg_rf = _allreduce_scalar(acc_rf / max(1, steps), device)

        run_diag = ((ep + 1) % cfg.flow_diag_every == 0
                    or ep == 0 or ep == cfg.flow_epochs - 1)

        if not run_diag:
            if r0 and (ep + 1) % max(1, cfg.flow_epochs // 20) == 0:
                print(f"  {label} ep {ep + 1:03d} | rf_loss={avg_rf:.6f} | "
                      f"lr={sched.get_last_lr()[0]:.2e}")
            continue

        should_stop = False
        if r0:
            flow.eval()
            with torch.no_grad():
                if has_ema:
                    with flow.ema.apply(flow.v):
                        u_diag = flow.encode_euler(z_diag, use_ema=False)
                else:
                    u_diag = flow.encode_euler(z_diag, use_ema=False)
                fit = sgld_fitness(u_diag, rp_dim=128, rp_seed=int(cfg.seed) + 12345)
                dist_corr = pairwise_distance_correlation(z_diag, u_diag)
            flow.train()

            print(f"  {label} ep {ep + 1:03d} | rf_loss={avg_rf:.6f} | "
                  f"lr={sched.get_last_lr()[0]:.2e}")

            should_stop = health.log_epoch(ep + 1, avg_rf, fit, dist_corr) and (
                ep > cfg.flow_min_epochs
            )
            if fit["score"] <= health.best_score + 1e-12:
                best_state = {
                    "v": {k: v.detach().cpu().clone()
                          for k, v in flow.v.state_dict().items()},
                    "ema_v": ({k: v.detach().cpu().clone()
                               for k, v in flow.ema.ema_model.state_dict().items()}
                              if has_ema else None),
                }
                best_epoch = ep + 1

        stop_t = torch.tensor([1 if should_stop else 0], device=device)
        dist.broadcast(stop_t, src=0)
        if bool(stop_t.item()):
            break

    if r0 and best_state is not None:
        flow.v.load_state_dict({k: v.to(device) for k, v in best_state["v"].items()})
        if has_ema and best_state["ema_v"] is not None:
            flow.ema.ema_model.load_state_dict(
                {k: v.to(device) for k, v in best_state["ema_v"].items()})
        print(f"Restored best flow checkpoint "
              f"(epoch={best_epoch}, best_score={health.best_score:.4f})")

    dist.barrier()
    _broadcast_module(flow, src=0)  # covers v + ema submodules
    flow.eval()

    if r0:
        summary = health.summary()
        print(f"  {label} finished: {summary.get('total_epochs_run', '?')} epochs  "
              f"best_score={summary.get('best_score', float('nan')):.4f}  "
              f"final_ρ={summary.get('final_dist_corr', float('nan')):.4f}")

    return flow, health


def train_ebm_ddp(model, buffer, u_all, cfg, label: str = "EBM", labels=None):
    """Drop-in DDP replacement for training.train_ebm (same signature).

    `buffer` (built by pipe.train) is replaced by a fresh per-rank buffer of
    cfg.buffer_size / world entries so the total negative-pool size and its
    refresh rate match the single-GPU recipe.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    r0 = rank == 0
    device = cfg.device
    assert cfg.batch_size % world == 0, (cfg.batch_size, world)
    per_batch = cfg.batch_size // world

    if r0:
        print(f"[-] Training {label} | epochs={cfg.epochs}")
        print(f"    [DDP] world_size={world}  global_batch={cfg.batch_size} "
              f"(per-rank {per_batch})  per-rank buffer={cfg.buffer_size // world}")

    # decorrelate per-rank RNG (buffer init, SGLD noise, reinit draws)
    torch.manual_seed(int(cfg.seed) * 2000 + rank + 1)
    torch.cuda.manual_seed_all(int(cfg.seed) * 2000 + rank + 1)

    per_size = max(per_batch, cfg.buffer_size // world)
    buffer = ReplayBuffer(per_size, u_all.size(1), device, temp=cfg.noise_temp)
    if cfg.buffer_init == "data":
        ridx = torch.randint(0, u_all.size(0), (per_size,))
        buffer.x.copy_(u_all[ridx].to(device))

    ds = TensorDataset(u_all)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                 shuffle=True, seed=int(cfg.seed) + 1, drop_last=True)
    dl = DataLoader(ds, batch_size=per_batch, sampler=sampler,
                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                    persistent_workers=True)

    class_indices = {}
    if labels is not None:
        if labels.device.type != "cpu":
            labels = labels.cpu()
        for c in labels.unique():
            class_indices[c.item()] = (labels == c).nonzero(as_tuple=True)[0]
        if r0:
            print(f"    Stratified buffer init enabled ({len(class_indices)} classes).")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=list(cfg.lr_milestones), gamma=cfg.lr_decay)
    model.train()

    for ep in range(cfg.epochs):
        sampler.set_epoch(ep)
        acc_cd = acc_e2 = acc_gpos = 0.0

        it = tqdm(dl, desc=f"{label} Epoch {ep + 1}", leave=False) if r0 else dl
        for (u_pos_cpu,) in it:
            u_pos = u_pos_cpu.to(device, non_blocking=True)
            if cfg.data_noise > 0:
                u_pos = u_pos + cfg.data_noise * torch.randn_like(u_pos)

            u_neg, v_neg, idx = buffer.sample(u_pos.size(0))

            if cfg.reinit_freq > 0:
                mask = torch.rand(u_pos.size(0), device=device) < cfg.reinit_freq
                if mask.any():
                    n_reinit = int(mask.sum())
                    if cfg.buffer_init == "flow_prior":
                        temp = cfg.noise_temp if hasattr(cfg, "noise_temp") else 1.0
                        u_neg[mask] = (torch.randn(n_reinit, u_pos.size(1),
                                                   device=device) * temp)
                    else:
                        if labels is not None and len(class_indices) > 0:
                            classes = list(class_indices.keys())
                            n_per_class = math.ceil(n_reinit / len(classes))
                            new_samples = []
                            for c in classes:
                                c_idxs = class_indices[c]
                                picked = c_idxs[torch.randint(0, len(c_idxs), (n_per_class,))]
                                new_samples.append(u_all[picked])
                            u_new = torch.cat(new_samples)
                            u_new = u_new[torch.randperm(u_new.size(0))][:n_reinit].to(device)
                            u_neg[mask] = u_new
                        else:
                            ridx = torch.randint(0, u_all.size(0), (n_reinit,))
                            u_neg[mask] = u_all[ridx].to(device)
                    v_neg[mask] = 0.0

            if cfg.use_precond:
                u_neg, v_neg = precond_sgld(model, u_neg, v_neg, cfg.sgld_steps,
                                            cfg.sgld_step_size, cfg.sgld_noise,
                                            cfg.sgld_beta, cfg.sgld_eps,
                                            cfg.precond_clip)
            else:
                u_neg, v_neg = vanilla_sgld(model, u_neg, v_neg, cfg.sgld_steps,
                                            cfg.sgld_step_size, cfg.sgld_noise,
                                            cfg.sgld_beta, cfg.sgld_eps,
                                            cfg.precond_clip)

            buffer.update(idx, u_neg, v_neg)

            opt.zero_grad(set_to_none=True)
            u_pos_req = u_pos.detach().requires_grad_(True)
            e_pos = model(u_pos_req)
            e_neg = model(u_neg.detach())

            with torch.no_grad():
                e_ref = e_pos.mean().item()
            e_pos_n = e_pos - e_ref
            e_neg_n = e_neg - e_ref

            loss_cd = e_pos_n.mean() - e_neg_n.mean()
            loss_e2 = cfg.alpha_energy2 * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
            gpos = torch.autograd.grad(e_pos.sum(), u_pos_req, create_graph=True)[0]
            loss_gp = cfg.lambda_grad * gpos.pow(2).sum(dim=1).mean()

            loss = loss_cd + loss_e2 + loss_gp
            loss.backward()
            _allreduce_grads(model)
            opt.step()

            acc_cd += float(loss_cd.item())
            acc_e2 += float(loss_e2.item())
            acc_gpos += float(loss_gp.item())

        sched.step()
        n = max(1, len(dl))
        cd = _allreduce_scalar(acc_cd / n, device)
        e2 = _allreduce_scalar(acc_e2 / n, device)
        gp = _allreduce_scalar(acc_gpos / n, device)
        if r0:
            print(f"  {label} ep {ep + 1:02d} | CD={cd:.4f} | L2={e2:.4f} | "
                  f"GPOS={gp:.4f} | lr={sched.get_last_lr()[0]:.2e}")

    # identical updates should keep ranks in sync, but broadcast rank 0's
    # weights to rule out numeric drift from non-deterministic kernels
    dist.barrier()
    _broadcast_module(model, src=0)
