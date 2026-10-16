"""VPDM — Vague Prototype-Oriented Diffusion Model for Multi-class Anomaly
Detection (Li et al., ICML 2024, PMLR v235, li24u). No official code is
public, so this is a reimplementation from the paper.

Paper-specified (Sec. 3, Alg. 1/2, Appendix D):
  - x0 = EfficientNet-b4 stage 1-4 features, 224x224 input, each resized to
    32x32 and concatenated -> (272, 32, 32).
  - VPOT: f_down = T_down(x0) (4x downsampling = low-pass), K = 50 randomly
    initialised prototypes beta in R^{K x d}, d = 8*8*272. Entropic OT between
    the batch empirical distribution P_f (uniform 1/Kf) and P_beta (uniform
    1/K) with Euclidean cost C_ij = ||f_i - beta_j||_2 (Eq. 9, 11);
    y = T x beta (Eq. 10); y_hat = T_up(y).
  - Diffusion (CARD-style, Eq. 12-15): x_T ~ N(y_hat, I),
    q(x_t | x0, y_hat) = N(sqrt(ab_t) x0 + (1 - sqrt(ab_t)) y_hat, (1 - ab_t) I),
    posterior mean gamma0 x0 + gamma1 x_t + gamma2 y_hat. T = 1000, linear
    betas 1e-4 -> 0.02.
  - Loss = L_ELBO (eps-prediction MSE) + L_OT (Eq. 17), prototypes receive
    gradients from both.
  - Inference (Alg. 2): full T-step ancestral sampling from N(y_hat, I).
    Anomaly map S = ||x0 - x0_rec||_2 over channels, bilinearly upsampled;
    image score = max of the average-pooled map (Sec. 3.4).

Unspecified in the paper (choices made here):
  - Denoiser eps_theta(x_t, y_hat, t): guided-diffusion U-Net on
    concat[x_t, y_hat] (CARD conditions eps_theta by concatenation).
  - T_down = 4x4 average pooling, T_up = bilinear interpolation.
  - Rows of T are renormalised to sum to 1 before y = T beta (Eq. 10 as written
    would scale y by 1/Kf).
  - Test-time OT: 'batch' = the same balanced Sinkhorn over the test batch
    (literal reading, depends on batch composition); 'semi' = semi-relaxed OT
    (only the per-sample marginal enforced) -> per-sample softmax(-C/eps).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_log(C, eps=1.0, n_iter=100):
    """Balanced entropic OT with uniform marginals, log-domain Sinkhorn.

    C: (n, m) cost. Returns the transport plan T (n, m), rows sum to 1/n and
    columns to 1/m. Differentiable (unrolled iterations).
    """
    n, m = C.shape
    log_a = torch.full((n,), -np.log(n), device=C.device, dtype=C.dtype)
    log_b = torch.full((m,), -np.log(m), device=C.device, dtype=C.dtype)
    logK = -C / eps
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        u = log_a - torch.logsumexp(logK + v[None, :], dim=1)
        v = log_b - torch.logsumexp(logK + u[:, None], dim=0)
    return torch.exp(u[:, None] + logK + v[None, :])


class VPOT(nn.Module):
    """Vague Prototype-Oriented Optimal Transport (Sec. 3.1)."""

    def __init__(self, channels=272, feat_size=32, down=4, num_prototypes=50,
                 eps=1.0, sinkhorn_iters=100):
        super().__init__()
        self.down = down
        self.feat_size = feat_size
        self.small = feat_size // down
        self.c = channels
        self.eps = eps
        self.sinkhorn_iters = sinkhorn_iters
        d = channels * self.small * self.small
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d))

    def downsample(self, x):
        return F.avg_pool2d(x, self.down)                      # T_down (low-pass)

    def upsample(self, y):
        return F.interpolate(y, size=(self.feat_size, self.feat_size),
                             mode='bilinear', align_corners=False)  # T_up

    def forward(self, x, mode='batch'):
        """x: (B, C, H, W) features. Returns y_hat (B, C, H, W), OT loss, plan."""
        B = x.shape[0]
        f = self.downsample(x).flatten(1)                        # (B, d)
        # Euclidean cost; clamp inside sqrt for a finite gradient at 0
        d2 = (f.pow(2).sum(1, keepdim=True) + self.prototypes.pow(2).sum(1)[None]
              - 2.0 * f @ self.prototypes.t())
        C = d2.clamp_min(1e-8).sqrt()                            # (B, K)
        if mode == 'batch':
            T = sinkhorn_log(C, eps=self.eps, n_iter=self.sinkhorn_iters)
            ot_loss = (T * C).sum() + self.eps * (T * torch.log(T + 1e-30)).sum()
            W = T / T.sum(1, keepdim=True)
        elif mode == 'semi':
            W = torch.softmax(-C / self.eps, dim=1)
            ot_loss = (W * C).sum(1).mean()
        else:
            raise ValueError(mode)
        y = W @ self.prototypes                                  # Eq. 10
        y = y.view(B, self.c, self.small, self.small)
        return self.upsample(y), ot_loss, W


class VPDM(nn.Module):
    def __init__(self, denoiser, vpot, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 endpoint='yhat', x0_cond=None):
        super().__init__()
        # x0_cond (NOT in the paper's eps_theta(x_t, y_hat, t); CARD's eps_theta
        # also receives the input x): feed the input features to the denoiser
        # as an extra channel block. mode 'none' | 'clean' | 'jitter' | 'mask';
        # the corruption is applied during training only (UniAD-style anti
        # identity-shortcut), inference sees the clean input.
        self.x0_cond = dict(x0_cond or {'mode': 'none'})
        self.model = denoiser
        self.vpot = vpot
        self.T = num_timesteps
        # 'yhat' = VPDM (Eq. 12-15); 'standard' = Table 3 ablation N(0, I) DDPM
        # conditioned on y_hat only through the network input
        self.endpoint = endpoint
        # optional (lo, hi) tensors broadcastable to x0; set from training
        # features. Ho et al.'s sampler clips x0_hat to [-1, 1]; Alg. 2 does
        # not mention it, but without it the unbounded-feature chain diverges.
        self.x0_range = None

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, 0)
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
        reg = lambda n, v: self.register_buffer(n, v.float())
        reg('betas', betas)
        reg('sqrt_ab', ab.sqrt())
        reg('sqrt_1mab', (1 - ab).sqrt())
        # CARD posterior q(x_{t-1} | x0, x_t, y_hat), Eq. 15 / Appendix A
        reg('gamma0', betas * ab_prev.sqrt() / (1 - ab))
        reg('gamma1', (1 - ab_prev) * alphas.sqrt() / (1 - ab))
        reg('gamma2', 1 + (ab.sqrt() - 1) * (alphas.sqrt() + ab_prev.sqrt()) / (1 - ab))
        reg('post_var', (1 - ab_prev) / (1 - ab) * betas)
        # standard DDPM posterior (ablation)
        reg('ddpm_c0', betas * ab_prev.sqrt() / (1 - ab))
        reg('ddpm_ct', (1 - ab_prev) * alphas.sqrt() / (1 - ab))

    def _mean_shift(self, t, y_hat):
        if self.endpoint == 'yhat':
            return (1 - self.sqrt_ab[t]).view(-1, 1, 1, 1) * y_hat
        return 0.0

    def eps_net(self, x_t, y_hat, t, cond=None):
        inp = [x_t, y_hat] if cond is None else [x_t, y_hat, cond]
        return self.model(torch.cat(inp, dim=1), t.float())

    def make_cond(self, x0, y_hat, train):
        mode = self.x0_cond.get('mode', 'none')
        if mode == 'none':
            return None
        if not train or mode == 'clean':
            return x0
        if mode == 'jitter':
            # UniAD feature jitter: gaussian noise scaled by the per-location RMS
            rms = x0.pow(2).mean(1, keepdim=True).sqrt()
            return x0 + torch.randn_like(x0) * rms * self.x0_cond.get('jitter', 1.0)
        if mode == 'mask':
            # replace random ps x ps blocks by the vague condition y_hat
            ps, B, _, H, W = self.x0_cond.get('patch', 4), *x0.shape
            keep = (torch.rand(B, 1, H // ps, W // ps, device=x0.device)
                    >= self.x0_cond.get('ratio', 0.5)).float()
            keep = F.interpolate(keep, size=(H, W), mode='nearest')
            return keep * x0 + (1 - keep) * y_hat
        raise ValueError(mode)

    def forward(self, x0):
        """Training loss (Alg. 1, Eq. 17). Returns (total, elbo_term, ot_term)."""
        B = x0.shape[0]
        y_hat, ot_loss, _ = self.vpot(x0, mode='batch')
        t = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = (self.sqrt_ab[t].view(B, 1, 1, 1) * x0 + self._mean_shift(t, y_hat)
               + self.sqrt_1mab[t].view(B, 1, 1, 1) * eps)
        loss_eps = F.mse_loss(self.eps_net(x_t, y_hat, t, self.make_cond(x0, y_hat, True)), eps)
        return loss_eps + ot_loss, loss_eps, ot_loss

    @torch.no_grad()
    def reconstruct(self, x0, ot_mode='batch', generator=None, return_yhat=False, t_start=None, one_step=False):
        """Alg. 2: full ancestral sampling from N(y_hat, I) (or N(0, I)).

        t_start (NOT in the paper, diagnostic/variant): start from
        q(x_t | x0, y_hat) at t = t_start instead of the endpoint.
        one_step (NOT in the paper, FlowAD plug-in): return the single x0 prediction
        at t_start (1 NFE, no x0_range clip) instead of running the chain.
        """
        B = x0.shape[0]
        y_hat, _, _ = self.vpot(x0, mode=ot_mode)
        cond = self.make_cond(x0, y_hat, False)
        randn = lambda: torch.randn(x0.shape, generator=generator, device=x0.device)
        if t_start is None:
            t_start = self.T - 1
            x = (y_hat if self.endpoint == 'yhat' else 0.0) + randn()
        else:
            t_start = min(t_start, self.T - 1)
            ts = torch.full((B,), t_start, device=x0.device, dtype=torch.long)
            x = self.sqrt_ab[t_start] * x0 + self._mean_shift(ts, y_hat) + self.sqrt_1mab[t_start] * randn()
        for t in range(t_start, -1, -1):
            tb = torch.full((B,), t, device=x0.device, dtype=torch.long)
            eps = self.eps_net(x, y_hat, tb, cond)
            x0_hat = (x - self._mean_shift(tb, y_hat) - self.sqrt_1mab[t] * eps) / self.sqrt_ab[t]
            if one_step:
                x = x0_hat
                break
            if self.x0_range is not None:  # DDPM-style clip of x0_hat to the data range
                x0_hat = torch.maximum(torch.minimum(x0_hat, self.x0_range[1]), self.x0_range[0])
            if t > 0:
                if self.endpoint == 'yhat':
                    mean = self.gamma0[t] * x0_hat + self.gamma1[t] * x + self.gamma2[t] * y_hat
                else:
                    mean = self.ddpm_c0[t] * x0_hat + self.ddpm_ct[t] * x
                x = mean + self.post_var[t].sqrt() * randn()
            else:
                x = x0_hat
        return (x, y_hat) if return_yhat else x

    @torch.no_grad()
    def score(self, x0, ot_mode='batch', generator=None, t_start=None, one_step=False):
        rec = self.reconstruct(x0, ot_mode=ot_mode, generator=generator, t_start=t_start, one_step=one_step)
        return (x0 - rec).pow(2).sum(1).sqrt()                    # S = ||x0 - x0_rec||_2
