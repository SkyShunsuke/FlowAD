# DiffusionAD — multi-class reproduction (DDP)

DiffusionAD: Norm-guided One-step Denoising Diffusion for Anomaly Detection
(Zhang et al., arXiv 2303.08730). Upstream code: github.com/HuiZhang0812/DiffusionAD
@ `8a64c03` (cloned in `baselines/DiffusionAD`; network/diffusion files vendored
unchanged in `orig/`).

Results: `RESULTS.md` (generated with `summarize.py`). Per-class CSVs: `out/<run>/final_last.csv`,
periodic evals `out/<run>/eval_ep*.csv`.

## Files

| file | role |
|---|---|
| `prepare_cache.py` | 256² uint8 caches (images, DIS foregrounds, DTD, test GT) |
| `data.py` | multi-class train set; `perlin_synthetic` copied from upstream |
| `train_ddp.py` | DDP training + sharded evaluation (FlowAD 7 metrics + mAD) |
| `pbs_train.sh` | one 8×H200 node per run: `qsub -N diffad_mvtec -l walltime=3:00:00 -v RUN=mvtec_300ep baselines/diffusionad_repro/pbs_train.sh` |
| `tools/calib_dis.py` | checks the regenerated foreground masks against the authors' |
| `configs/*.json` | per-run overrides of upstream `args1.json` |

Checkpoints: `/groups/gch51615/acg17237wy/diffusionad_repro/<run>/` (home is at quota).
Real-IAD's cache (~40 GB) is rebuilt on node-local disk at job start (~3 min).

## Kept exactly as upstream

- Reconstruction U-Net (base 128, mults 1,1,2,2,4,4, attention at 32/16/8, 4 heads, 131.7M params)
  and segmentation sub-network (28.4M), inputs 256×256 RGB in [0,1].
- Linear β schedule, T=1000; training: t_normal ~ U[0,300), t_noisier ~ U[300,1000);
  ε-MSE on normal samples only; norm-guided one-step x̂0 passed **with gradient** to the seg net.
- Loss = noise + 5·focal(α=0.5, γ=4) + SmoothL1; Adam lr 1e-4 (U-Net) / 1e-5 (seg), wd 0;
  CosineAnnealingLR(T_max=10) on the seg optimiser stepped **per iteration** (upstream quirk).
- Batch 16 per GPU (BatchNorm in the seg net sees the same batch as upstream).
- Anomaly synthesis: Perlin masks × foreground, DTD textures or 8×8 self-shuffle (objects),
  DTD only (textures), β∈[0.2,0.8], 50 % normal samples. Upstream's `texture_list`
  (carpet, zipper, leather, tile, wood, grid) decides texture handling by class name.
- Foregrounds: objects use DIS (IS-Net `isnet-general-use.pth`, DIS `Inference.py`
  preprocessing, threshold 0.5). Regenerating them this way reproduces the authors' released
  masks exactly (IoU 1.000 on bottle, pill, candle, pcb1; `tools/calib_dis.py`). Textures use
  the authors' 32 released `carpet/thresh` masks.
- Inference: t_normal=200, t_noisier=400, condition_w=1; anomaly map = seg output (no smoothing);
  image score = mean of the top-50 pixels of the map.

## Changes

1. **Multi-class**: one model per benchmark trained on the union of its classes. Like upstream,
   the train set ignores the sampler index and draws a random image, so classes are sampled in
   proportion to their size. The model has no class input.
2. **DDP** on 8 GPUs: global batch 128 (16/GPU), same learning rates (no LR scaling). With the
   drop-last sampler an epoch is ⌊N/128⌋ steps.
3. **Scope**: MVTec-AD and VisA only (the benchmarks the paper's pseudo-anomaly pipeline
   was designed for); MPDD/Real-IAD code paths remain but are not reported.
   **Budget**: 300 epochs (FlowAD's budget); upstream trains 3000 per
   class (one model per class). Upstream has no
   epoch-dependent LR schedule, so any intermediate eval is exactly what a shorter run gives.
4. **Reported weights = final epoch.** Upstream evaluates every 50 epochs on the test set and
   keeps `best` (criterion: img+px AUROC ≥ best and img AUROC ≥ best). We run the same evals and
   report that selection separately, for reference only.
5. **Data**: FlowAD splits and GT masks (256, nearest) for all four datasets (VisA: official
   1cls split; Real-IAD: realiad_jsons). Upstream eval reads masks with bilinear resize and
   `astype(int)`; we use the FlowAD masks as for all other baselines. MPDD and Real-IAD had no
   released foregrounds; they get masks from the same DIS pipeline. Upstream's VisA masks were
   stored as JPEG (compression noise at edges); ours are exact binary.
6. **Caching**: every upstream input passes through `cv2.resize(·, 256)` first; we store that
   uint8 result (bit-exact) instead of decoding full-size images each step.
7. **Numerics / RNG**: TF32 matmul/conv (upstream: fp32); per-worker imgaug seeding (upstream
   forks identical imgaug RNG states into all workers); eval noise seeded per rank.
8. **Metrics**: FlowAD 7 metrics (I-AUROC, I-AP, I-F1max, P-AUROC, P-AUPRO, P-AP, P-F1max),
   mAD = their mean; per class, then averaged over classes.
