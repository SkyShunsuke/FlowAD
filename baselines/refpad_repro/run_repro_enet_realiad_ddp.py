"""Multi-GPU version of run_repro_enet_realiad.py (torchrun entrypoint).

User-authorized deviation (2026-09-22): ReFP-AD training is parallelized
across GPUs by monkey-patching pipeline.train_flow / pipeline.train_ebm with
the gradient-all-reduce implementations in ddp_training.py (see that file's
docstring for the exact semantics kept and changed). The ReFP-AD clone stays
unmodified. Everything else — config, no early stopping (all 250 flow
epochs), single final eval, output formats — is identical to the single-GPU
runner, whose helpers are imported here.

The final eval is also parallelized: rank r scores categories[r::world] and
writes score_maps/<cat>.npz; rank 0 then recomputes all metrics from the npz
files and writes eval_results_flowad_256.csv / eval_results_full.json.

Launch (single node):
  torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    run_repro_enet_realiad_ddp.py --data_path ... --out_dir ... --cache_dir ...

Rank 0 prints to stdout; other ranks' stdout goes to <out_dir>/rank<r>.log.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# module-level side effects: sys.path for ReFP-AD + FlowAD, RealIAD category /
# GT-style registration, pipeline.DinoBackbone = EnetBackbone
import run_repro  # noqa: E402
from run_repro_enet_realiad import (  # noqa: E402
    category_metrics,
    finalize,
    infer_category,
    load_npz,
    save_npz,
)
from config import CFG, Mode  # noqa: E402  (ReFP-AD module)
from utils import seed_all  # noqa: E402
import pipeline  # noqa: E402

from enet_backbone import IMG_SIZE  # noqa: E402
import ddp_training  # noqa: E402


def gather_and_finalize(res, nat, categories, out, rank, world):
    """All ranks contribute their per-category metric dicts; rank 0 writes."""
    gathered = [None] * world
    dist.all_gather_object(gathered, (res, nat))
    if rank == 0:
        allres, allnat = {}, {}
        for gres, gnat in gathered:
            allres.update(gres)
            allnat.update(gnat)
        missing = [c for c in categories if c not in allres]
        assert not missing, f"missing categories in merge: {missing}"
        finalize(allres, allnat, list(categories), out)


def distributed_merge(categories, maps_dir, hw, out, rank, world):
    """Recompute metrics from score_maps/*.npz, categories sharded per rank."""
    res, nat = {}, {}
    for cat in list(categories)[rank::world]:
        res[cat], nat[cat] = category_metrics(cat, load_npz(maps_dir, cat), hw)
    gather_and_finalize(res, nat, categories, out, rank, world)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=str(run_repro.FLOWAD_ROOT / "data" / "realiad_1cls"))
    ap.add_argument("--out_dir", default=str(HERE / "out" / "realiad_unified_enet_ddp"))
    ap.add_argument("--cache_dir", default=str(HERE / "cache_tokens_enet"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--merge_only", action="store_true",
                    help="rebuild CSV/JSON from saved score_maps/*.npz; the "
                         "per-category metric computation is sharded across ranks")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    r0 = rank == 0

    cfg = CFG(
        dataset="RealIAD",
        data_path=args.data_path,
        out_dir=args.out_dir,
        cache_dir=Path(args.cache_dir),
        mode=Mode.UNIFIED,
        seed=args.seed,
        device="cuda",  # resolves to this rank's device via set_device
        use_flow=True,
        buffer_init="flow_prior",  # main.py CLI default
        resize=(IMG_SIZE, IMG_SIZE),
        center_crop=(IMG_SIZE, IMG_SIZE),
    )
    cfg.dino_model = "efficientnet-b4"
    cfg.flow_min_epochs = cfg.flow_epochs  # no early stopping: train ALL epochs

    if os.environ.get("REFPAD_SMOKE") == "1":
        cfg.flow_epochs, cfg.flow_min_epochs, cfg.flow_patience = 2, 2, 1
        cfg.epochs = 1
        cfg.categories = cfg.categories[:2]
        if r0:
            print("!!! SMOKE TEST MODE: 2 categories, tiny epochs !!!")

    hw = cfg.center_crop[0]  # 256
    root = Path(cfg.data_path)
    out = Path(cfg.out_dir)
    maps_dir = out / "score_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # keep the main log clean: non-rank-0 stdout goes to a per-rank file
    if not r0:
        sys.stdout = open(out / f"rank{rank}.log", "w", buffering=1)

    if args.merge_only:
        distributed_merge(cfg.categories, maps_dir, hw, out, rank, world)
        dist.barrier()
        dist.destroy_process_group()
        return

    if world > 1:
        pipeline.train_flow = ddp_training.train_flow_ddp
        pipeline.train_ebm = ddp_training.train_ebm_ddp

    # identical seed on every rank -> identical model init (required for the
    # gradient-all-reduce scheme); per-rank RNG diverges inside the DDP loops
    seed_all(cfg.seed)
    if r0:
        cfg.save()
    print(f"### ReFP-AD repro (EfficientNet-B4, DDP world={world}) | mode=UNIFIED "
          f"dataset=RealIAD seed={cfg.seed} "
          f"flow_epochs={cfg.flow_epochs} (early stopping disabled)")

    pipe = pipeline.build_pipeline(cfg)

    ckpt_path = Path(args.ckpt) if args.ckpt else out / "unified_checkpoint.pt"
    if args.eval_only:
        pipe.load_checkpoint(ckpt_path)
    else:
        t0 = time.time()
        pipe.train()  # preprocessing duplicated per rank; training synchronized
        if r0:
            print(f"[-] Training took {(time.time() - t0) / 3600:.2f} h")
            pipe.save_checkpoint()
    dist.barrier()

    # ---------------- evaluation: categories sharded across ranks ----------------
    my_cats = list(cfg.categories)[rank::world]
    res, nat = {}, {}
    t0 = time.time()
    for cat in my_cats:
        d = infer_category(pipe, root, cat, hw)
        save_npz(maps_dir, cat, d)
        res[cat], nat[cat] = category_metrics(cat, d, hw)
        print(f"[-] Inference done: {cat} ({len(d['img_scores'])} images, "
              f"{time.time() - t0:.0f}s elapsed)", flush=True)
        del d
    dist.barrier()

    # each rank already computed its shard's metrics on its own GPU; gather
    # the small result dicts and let rank 0 write the CSV/JSON
    gather_and_finalize(res, nat, cfg.categories, out, rank, world)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
