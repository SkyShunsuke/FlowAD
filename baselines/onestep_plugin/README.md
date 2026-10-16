# FlowAD one-step reconstruction as a plug-in for other diffusion AD methods

FlowAD's inference (`src/vfad/eval_recon.py`): perturb the clean features once, predict x0 in a
single network evaluation (NFE), anomaly map = channel-summed squared error. Here the same
procedure replaces the native sampler of other methods, **on their own trained weights, with no
retraining**. For DDPM models (linear betas, T=1000):

    x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) eps   (+ method-specific mean shift, VPDM)
    x0_hat = model's x0 prediction at t (from eps or x0 parameterization), 1 NFE

FlowAD's operating point t_FM = 0.1 (x_t = 0.1 x1 + 0.9 eps) has the same SNR as DDPM t = 658;
t is swept because the other models were trained with uniform t (FlowAD concentrates training
around its inference t, logistic-normal t sampling).

All numbers: multi-class, FlowAD metrics (`src/utils/adeval/eval_utils.py`), class average, x100.
mAD = mean of img AUROC/AP/F1max, px AUROC/AP/F1max/AUPRO.

| method | weights | native inference | code |
|---|---|---|---|
| InvAD (CVPR'26) | official release (`~/projects/InversionAD/results/exp_dit_gigant_*_enet`) | DDIM inversion, 4 NFE (config `eval_step`) | `invad_plugin.py` |
| VPDM (ICML'24) | our reproduction (no official code/weights) | 1000-step ancestral chain from N(y_hat, I) | `../vpdm_repro` variant `oN_batch` |
| OmiAD teacher AMDM (ICML'25) | our reproduction (official weights only on Baidu Pan) | t_init=960, 40-step DDIM | `../amdm_repro` `eval_variants` |

Scoring: InvAD one-step uses FlowAD's `diff+sum` image score (InvAD has no recon score).
VPDM and AMDM keep their own scores (VPDM: L2 map, avg-pool-16 max; AMDM: max) so only the
sampler changes.

## InvAD (official DiT-G weights)

| dataset | inference | NFE | img AUROC | px AUROC | px AP | AUPRO | mAD |
|---|---|---|---|---|---|---|---|
| MVTec | inversion (native) | 4 | 98.9 | 97.3 | 45.6 | 92.5 | 83.4 |
| MVTec | one-step t=500 | 1 | 98.8 | 97.0 | 47.3 | 92.4 | 83.6 |
| MVTec | one-step t=600 | 1 | 98.6 | 97.0 | 48.0 | 92.4 | **83.7** |
| MVTec | one-step t=658 (FlowAD SNR) | 1 | 98.2 | 96.9 | 47.9 | 92.4 | 83.6 |
| VisA | inversion (native) | 4 | 97.0 | 99.1 | 37.0 | 93.1 | **79.8** |
| VisA | one-step t=500 | 1 | 95.6 | 98.8 | 39.2 | 90.7 | 79.4 |
| VisA | one-step t=658 | 1 | 94.4 | 98.6 | 37.9 | 90.1 | 78.5 |
| MPDD | inversion (native) | 4 | 95.6 | 97.9 | 36.1 | 88.9 | 78.2 |
| MPDD | one-step t=300 | 1 | 94.6 | 98.3 | 38.3 | 92.3 | **79.2** |
| MPDD | one-step t=658 | 1 | 84.5 | 97.1 | 29.7 | 89.0 | 72.3 |

Latency (MVTec, bf16, batch 32, denoiser only, per image): inversion 6.7 ms, one-step 1.7 ms;
bf16 one-step accuracy is identical to fp32 (`out/invad_mvtec_bf16`).
Full sweeps (t = 50 ... 900) and per-class CSVs: `out/invad_{mvtec,visa,mpdd}/`.

## VPDM (our reproduction)

Native = paper's Alg. 2 (`final_raw_full_batch.csv` of the same run).

| run | inference | NFE | img AUROC | px AUROC | AUPRO | mAD |
|---|---|---|---|---|---|---|
| MVTec s0/s1/s2 (paper model) | full chain | 1000 | 54.9 | 86.5 | 63.9 | 57.8 |
| MVTec s0/s1/s2 (paper model) | one-step t=100 | 1 | 73.5 | 95.4 | 87.5 | **72.2** |
| MVTec s0/s1/s2 (paper model) | one-step t=658 | 1 | 47.9 | 81.8 | 48.7 | 51.3 |
| MVTec jitter s2 (x0-cond) | full chain | 1000 | 82.8 | 89.6 | 76.6 | 71.0 |
| MVTec jitter s2 (x0-cond) | one-step t=200 | 1 | 95.5 | 96.5 | 90.4 | **81.4** |
| VisA jitter s2 | full chain | 1000 | 68.2 | 92.7 | 66.1 | 58.7 |
| VisA jitter s2 | one-step t=300 | 1 | 92.2 | 98.6 | 87.8 | **77.6** |
| MPDD jitter s2 | full chain | 1000 | 77.4 | 95.9 | 85.3 | 65.9 |
| MPDD jitter s2 | one-step t=200 | 1 | 91.3 | 97.6 | 90.8 | **74.5** |
| Real-IAD jitter s2 (30 ep) | full chain | 1000 | 64.3 | 93.7 | 75.1 | 55.2 |
| Real-IAD jitter s2 (30 ep) | one-step t=300 | 1 | 79.3 | 96.5 | 86.7 | **65.5** |

(MVTec paper-model rows: mean over 3 seeds.) Per-t logs: `../vpdm_repro/out/onestep_*.log`,
CSVs `../vpdm_repro/out/onestep_*/final_raw_o*_batch.csv`.

## Notes

- The best t is lower than FlowAD's SNR-matched t=658 for models trained with uniform t,
  and lowest for VPDM (its high-t predictions collapse onto the prototype y_hat).
- Test-set selection of t is optimistic; for a fair claim fix t per method on one dataset
  (e.g. InvAD t=500, VPDM t=200) and report it unchanged elsewhere.
- InvAD MVTec/MPDD configs use `eval_step: 4`; the paper reports S=3.

## ReFP-AD (our EfficientNet-B4 reproduction) — MVTec

ReFP-AD is not reconstruction-based: an OT rectified flow encodes each token (RK4 x 10 = 40 NFE),
then the score is the gradient norm of an EBM on the encoded token. The plug-in is the one-step
encode u = z + v(0, z) (1 NFE), `run_repro_enet.py --encode euler1`, in two settings:
`euler1` (EBM also trained on one-step latents) and `euler1_eval` (native checkpoint, one-step at
test time only). Seed 42, UNIFIED, single run each.

| inference | flow NFE | img AUROC | img AP | px AUROC | px AP | AUPRO | mAD |
|---|---|---|---|---|---|---|---|
| RK4 x 10 (native) | 40 | 90.8 | 95.7 | 95.3 | 41.9 | 88.0 | 78.7 |
| one-step, test only (`euler1_eval`) | 1 | 92.0 | 96.1 | 95.7 | 43.8 | 88.9 | 79.7 |
| one-step, train + test (`euler1`) | 1 | 92.8 | 96.6 | 95.7 | 44.8 | 89.0 | **80.2** |

Job: `../refpad_repro/pbs_onestep_mvtec.sh` (run 2026-09-26 on interactive node, ~0.2 h training
per run), outputs `../refpad_repro/out/mvtec_unified_enet_{native,euler1,euler1_eval}/`
(`eval_results_flowad_256.csv`, per-class; logs `../refpad_repro/logs/mvtec_enet_*.log`).
