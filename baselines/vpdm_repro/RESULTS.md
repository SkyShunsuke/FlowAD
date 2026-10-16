# VPDM (Li et al., ICML 2024) — multi-class MVTec-AD reproduction

Accelerated protocol: 300 epochs (`configs/vpdm_mvtec_300ep.yaml`, `run_mvtec_300ep.sh`),
3 seeds, final-epoch weights (no test-set model selection), FlowAD 7-metric eval.
Per-class CSVs: `out/vpdm300_mvtec_s*/final_*.csv`.

Paper (Table 1): img AUROC 98.4 / px AUROC 97.8.

## 300 epochs, mean ± std over seeds 0-2 (x100)

| variant | img AUROC | img AP | img F1 | px AUROC | px AUPRO | px AP | px F1 | mAD |
|---|---|---|---|---|---|---|---|---|
| **raw full_batch (Alg. 2, paper)** | 54.9±0.9 | 76.7±1.0 | 83.8±0.1 | 86.5±0.5 | 63.9±1.3 | 14.6±0.8 | 24.5±0.9 | 57.8±0.4 |
| raw full_semi | 50.8±0.4 | 74.1±0.8 | 83.7±0.0 | 89.3±0.5 | 73.2±0.7 | 19.4±0.8 | 30.5±0.8 | 60.1±0.3 |
| ema full_batch | 53.4±1.5 | 75.5±1.2 | 83.9±0.2 | 81.9±0.3 | 59.4±1.2 | 12.5±0.5 | 21.2±0.6 | 55.4±0.6 |
| raw t250_batch (not in paper) | 57.4±2.8 | 78.4±1.6 | 83.8±0.1 | 95.7±0.2 | 89.1±0.5 | 32.0±1.2 | 44.9±0.8 | 68.8±0.9 |
| raw t100_batch (not in paper) | 68.8±9.0 | 84.8±4.5 | 86.1±2.2 | 94.3±0.2 | 85.8±0.5 | 29.3±1.6 | 40.3±1.0 | 69.9±2.7 |
| raw yhat_batch (‖x0−ŷ‖, diagnostic) | 79.9±0.8 | 90.3±0.6 | 89.8±0.3 | 88.9±0.1 | 70.8±0.5 | 25.5±0.9 | 30.8±0.5 | 68.0±0.2 |

Table 3 ablation, N(0,I) DDPM endpoint (seed 0), Alg. 2: img 52.0 / px 82.4.

## 1000 epochs (earlier runs, `out/vpdm_mvtec_s*`), same variants

raw full_batch: img 55.4±0.9 / px 84.5±1.6 (mAD 56.7). t100: img 80.8±7.1 / px 96.4±0.1.
300 epochs matches 1000 epochs on the paper's Alg. 2 metric; only the partial-noise
(tN) variants benefit from longer training.

## Takeaway

The paper's numbers are not reproduced. The full T=1000 chain from N(ŷ, I) carries
information about the input only through ŷ (≈ one of 50 prototypes at 8x8), so
reconstructions are generic class samples and the residual is large on normal images too.
Deviations from the paper are listed in `configs/vpdm_mvtec.yaml` / `vpdm.py` docstring.

## Follow-up diagnostics (step 1) — `diag_sampling.py`, `out/*/diag_sampling.txt`

Per-pixel L2 error (standardised features, mean over map), 1000-ep s0; 300-ep s0 is the same:

| start step | err to input (normal train) | seed-to-seed spread | test img AUROC |
|---|---|---|---|
| y_hat only | 10.3 | – | 0.74 |
| 0 / 10 / 100 / 250 / 500 | 0.16 / 0.80 / 3.1 / 4.6 / 6.3 | 0.15 / 0.73 / 2.5 / 3.5 / 4.4 | 0.85 / 0.85 / 0.82 / 0.63 / 0.44 |
| 999 (Alg. 2) | 17.5 | 9.8 | 0.52 |

- The sampler is consistent (error -> 0 as start step -> 0). Full-chain reconstructions lie
  **off the data manifold**: distance to the nearest training feature map is 15.8, against 8.8
  between neighbouring training images (ŷ: 9.5).
- **Corner artifact:** in every baseline checkpoint (300 and 1000 ep), 100% of normal test images
  have their maximum error at a map corner (peak ≈40, map mean ≈15). The max-pooled image score
  therefore reads a fixed artifact. Cropping a 2-px border alone: img AUROC 0.49-0.52 -> 0.64-0.67.
  Backbone features at the corners are not unusual (norm 22 vs 17 at the centre, similar spread),
  so the artifact comes from the denoiser's generation near the border.

## Step 2: input features fed to the denoiser (NOT in paper), 300 ep, Alg. 2 inference

`configs/vpdm_mvtec_300ep_x0cond.yaml`, `run_mvtec_300ep_x0cond.sh`, `out/vpdm300x_mvtec_*`.
raw full_batch, mean±std over seeds (per-seed img/px):

| variant | img AUROC | px AUROC | px AUPRO | mAD | per seed |
|---|---|---|---|---|---|
| baseline (no input path) | 54.9±0.9 | 86.5±0.5 | 63.9±1.3 | 57.8 | 55.9/86.0 53.7/86.4 55.0/87.2 |
| jitter (noise std = 1×RMS, train only) | 62.2±14.6 | 87.4±1.5 | 70.5±4.5 | 62.1 | 51.9/86.4 51.9/86.3 **82.8/89.6** |
| mask (50% 4×4 blocks -> ŷ, train only) | 50.4±2.5 | 72.7±4.8 | 43.2±7.0 | 49.1 | 47.6/67.5 53.6/71.5 50.0/79.0 |
| clean (no corruption; collapsed at ep209, = ep208 weights) | 65.6 | 76.4 | 44.4 | 54.3 | copies anomalies (EMA px 42.6) |

- The input path anchors the chain (a reconstruction's nearest training map is its own input;
  error 17.8 -> 12.0) but does not fix detection on average.
- jitter s2's gain is a property of the model, not eval noise: eval seeds 0/2/5 give 0.82/0.83/0.84
  (s0: 0.51/0.52/0.51). s2 is the one checkpoint **without the corner artifact** (peak 15 vs 35);
  cropping the border brings jitter s0 to 0.72, level with s2's 0.73.
- Collapse-guard bug found and fixed in `train_vpdm.py`: restoring the snapshot's optimizer state
  also restored its lr, so consecutive rollbacks never halved it more than once (the clean run looped
  92 times). Earlier runs had at most one rollback in a row and are unaffected.
