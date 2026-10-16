import tqdm
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from src.flow_matching.path.scheduler import CondOTScheduler, VPScheduler, \
    CosineScheduler, EDMScheduler, LinearVPScheduler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import Solver, ODESolver

import logging
logger = logging.getLogger(__name__)

PRED_TYPES = ['data', 'noise', 'velocity']
LOSS_TYPES = ['data', 'noise', 'velocity', 'edm']

def spatial_gaussian_log_density(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, H, W)
    return Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(x).sum(dim=1)

def build_scheduler(scheduler_name: str, scheduler_params: dict) -> CondOTScheduler:
    """Build scheduler for flow matching.

    Args:
        scheduler_name (str): Name of the scheduler.
        scheduler_params (dict): Parameters for the scheduler.

    Returns:
        CondOTScheduler: Configured scheduler.
    """
    if scheduler_name == 'affine_prob':
        scheduler = CondOTScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    elif scheduler_name == 'vp':
        scheduler = VPScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    elif scheduler_name == 'cosine':
        scheduler = CosineScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    elif scheduler_name == 'linear_vp':
        scheduler = LinearVPScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    elif scheduler_name == 'edm':
        scheduler = EDMScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    else:
        raise ValueError(f'Unknown scheduler name: {scheduler_name}')
    
def build_solver(model: nn.Module, solver_name: str, solver_params: dict) -> Solver:
    """Build solver for flow matching.

    Args:
        model (nn.Module): The velocity model.
        solver_name (str): Name of the solver.
        solver_params (dict): Parameters for the solver.

    Returns:
        Solver: Configured solver.
    """
    return ODESolver(model, **solver_params)

def logit_t(bs: int, device: torch.device, mu=0., sigma=1.) -> torch.Tensor:
    # -- sample t from gaussian
    t = torch.randn(bs, device=device) * sigma + mu
    # -- apply sigmoid
    t = torch.sigmoid(t)
    return t

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_batch_t(t, B, device, dtype):
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=dtype)

    t = t.to(device=device, dtype=dtype)

    if t.ndim == 0:
        t = t.expand(B)

    if t.shape != (B,):
        raise ValueError(f"t must have shape ({B},), got {tuple(t.shape)}")

    return t


def _unfold_patches(x, patch_radius: int):
    """
    x: (B, C, H, W)
    return: (B, C * K * K, H, W)
    """
    if patch_radius == 0:
        return x

    B, C, H, W = x.shape
    k = 2 * patch_radius + 1

    patches = F.unfold(x, kernel_size=k, padding=patch_radius)
    patches = patches.view(B, C * k * k, H, W)

    return patches


def _shift_to_query_grid(x, dy: int, dx: int):
    """
    output[..., h, w] = x[..., h + dy, w + dx]

    つまり、query pixel p=(h,w) に対して、
    reference 側の p + delta を同じグリッド位置に持ってくる。

    x: (..., H, W)
    return:
        shifted: (..., H, W)
        valid_mask: (1, 1, H, W)
    """
    H, W = x.shape[-2:]
    out = torch.zeros_like(x)

    if dy >= 0:
        src_y = slice(dy, H)
        dst_y = slice(0, H - dy)
    else:
        src_y = slice(0, H + dy)
        dst_y = slice(-dy, H)

    if dx >= 0:
        src_x = slice(dx, W)
        dst_x = slice(0, W - dx)
    else:
        src_x = slice(0, W + dx)
        dst_x = slice(-dx, W)

    valid_h = H - abs(dy)
    valid_w = W - abs(dx)

    mask = torch.zeros((1, 1, H, W), dtype=torch.bool, device=x.device)

    if valid_h > 0 and valid_w > 0:
        out[..., dst_y, dst_x] = x[..., src_y, src_x]
        mask[..., dst_y, dst_x] = True

    return out, mask

class LocalPatchClosedFM(nn.Module):
    """
    Local correspondence version of closed-form FM.

    各 query pixel p に対して、訓練画像 i と近傍 offset delta の候補
    x_i[p + delta] を softmax で混合し、局所速度場を作る。

    This is not the exact closed-form FM of the original empirical image distribution.
    It is a local pixel/patch-token surrogate.
    """

    def __init__(
        self,
        train_samples: torch.Tensor,
        patch_radius: int = 2,
        search_radius: int = 2,
        train_chunk_size: int = 128,
        normalize_patch_dist: bool = True,
        bandwidth: float = 1.0,
        spatial_sigma=None,
    ):
        """
        train_samples: (N, C, H, W)

        patch_radius:
            重み計算に使うパッチ半径。
            0 -> 1x1 pixel
            1 -> 3x3 patch
            2 -> 5x5 patch

        search_radius:
            対応画素の探索範囲。
            0 -> 同座標のみ
            1 -> 3x3 近傍
            2 -> 5x5 近傍

        train_chunk_size:
            訓練サンプル方向の chunk size。
            N が大きい場合、ここを小さくする。

        normalize_patch_dist:
            True の場合、patch dimension で距離を割る。
            パッチサイズを大きくしたとき softmax が過度に鋭くなるのを防ぐ。

        bandwidth:
            softmax の温度係数。
            大きくすると重みが滑らかになる。

        spatial_sigma:
            None でなければ、offset delta に空間距離ペナルティを入れる。
            例: spatial_sigma=1.0
        """
        super().__init__()

        if train_samples.ndim != 4:
            raise ValueError("train_samples must have shape (N, C, H, W)")

        self.register_buffer("train_samples", train_samples)

        self.patch_radius = int(patch_radius)
        self.search_radius = int(search_radius)
        self.train_chunk_size = int(train_chunk_size)
        self.normalize_patch_dist = bool(normalize_patch_dist)
        self.bandwidth = float(bandwidth)
        self.spatial_sigma = spatial_sigma

    @torch.no_grad()
    def get_optimal_v(self, xt: torch.Tensor, t, eps: float = 1e-6):
        """
        xt: (B, C, H, W)
        t: scalar or (B,)

        return:
            v: (B, C, H, W)
        """
        B, C, H, W = xt.shape

        if self.train_samples.shape[1:] != (C, H, W):
            raise ValueError(
                f"train_samples shape {tuple(self.train_samples.shape[1:])} "
                f"does not match xt shape {(C, H, W)}"
            )

        t = _as_batch_t(t, B, xt.device, xt.dtype)

        one_minus_t = torch.clamp(1.0 - t, min=eps)
        denom = 2.0 * (one_minus_t * self.bandwidth) ** 2  # (B,)

        # query patch: phi_p(x_t)
        q_patch = _unfold_patches(xt, self.patch_radius)  # (B, D, H, W)
        D = q_patch.shape[1]
        q_norm2 = (q_patch ** 2).sum(dim=1)                # (B, H, W)

        # streaming logsumexp accumulation
        m = torch.full(
            (B, H, W),
            -float("inf"),
            device=xt.device,
            dtype=xt.dtype,
        )
        z = torch.zeros((B, H, W), device=xt.device, dtype=xt.dtype)
        num = torch.zeros((B, C, H, W), device=xt.device, dtype=xt.dtype)

        offsets = [
            (dy, dx)
            for dy in range(-self.search_radius, self.search_radius + 1)
            for dx in range(-self.search_radius, self.search_radius + 1)
        ]

        N = self.train_samples.shape[0]

        for start in range(0, N, self.train_chunk_size):
            ref = self.train_samples[start:start + self.train_chunk_size]
            ref = ref.to(device=xt.device, dtype=xt.dtype)  # (n, C, H, W)

            # phi_q(x_i)
            ref_patch_base = _unfold_patches(ref, self.patch_radius)  # (n, D, H, W)

            for dy, dx in offsets:
                # query grid p に対して、reference の p + delta を対応させる
                ref_patch, valid = _shift_to_query_grid(ref_patch_base, dy, dx)
                ref_val, _ = _shift_to_query_grid(ref, dy, dx)

                # distance:
                # || phi_p(x_t) - t * phi_{p+delta}(x_i) ||^2
                ref_norm2 = (ref_patch ** 2).sum(dim=1)  # (n, H, W)

                dot = torch.einsum(
                    "bdhw,ndhw->bnhw",
                    q_patch,
                    ref_patch,
                )  # (B, n, H, W)

                d2 = (
                    q_norm2[:, None, :, :]
                    - 2.0 * t[:, None, None, None] * dot
                    + (t[:, None, None, None] ** 2) * ref_norm2[None, :, :, :]
                )  # (B, n, H, W)

                if self.normalize_patch_dist:
                    d2 = d2 / D

                logits = -d2 / denom[:, None, None, None]  # (B, n, H, W)

                # offset が大きくて画像外を参照する候補を除外
                logits = logits.masked_fill(~valid, -float("inf"))

                # optional spatial prior:
                # 遠い offset をやや不利にする
                if self.spatial_sigma is not None:
                    spatial_cost = (dy * dy + dx * dx) / (
                        2.0 * float(self.spatial_sigma) ** 2
                    )
                    logits = logits - spatial_cost

                # block-wise stable softmax accumulation
                block_m = logits.max(dim=1).values  # (B, H, W)

                block_exp = torch.exp(logits - block_m[:, None, :, :])
                block_exp = torch.where(
                    torch.isfinite(logits),
                    block_exp,
                    torch.zeros_like(block_exp),
                )

                block_z = block_exp.sum(dim=1)  # (B, H, W)

                # sum exp(logit) * x_i[p + delta]
                block_num = torch.einsum(
                    "bnhw,nchw->bchw",
                    block_exp,
                    ref_val,
                )  # (B, C, H, W)

                new_m = torch.maximum(m, block_m)

                old_scale = torch.exp(m - new_m)
                old_scale = torch.where(
                    torch.isfinite(m),
                    old_scale,
                    torch.zeros_like(old_scale),
                )

                block_scale = torch.exp(block_m - new_m)
                block_scale = torch.where(
                    torch.isfinite(block_m),
                    block_scale,
                    torch.zeros_like(block_scale),
                )

                z = z * old_scale + block_z * block_scale
                num = (
                    num * old_scale[:, None, :, :]
                    + block_num * block_scale[:, None, :, :]
                )

                m = new_m

        # local conditional expectation of matched target pixel
        x_bar = num / torch.clamp(z[:, None, :, :], min=eps)

        # local velocity
        v = (x_bar - xt) / one_minus_t[:, None, None, None]

        return v

    @torch.no_grad()
    def sample(self, x0, steps: int, start_t: float = 0.0, **model_kwargs):
        timesteps = torch.linspace(
            start_t,
            1.0,
            steps + 1,
            device=x0.device,
            dtype=x0.dtype,
        )

        step_size = (1.0 - start_t) / steps

        x = x0

        for i in range(steps):
            t = timesteps[i].expand(x.shape[0])
            x = x + self.get_optimal_v(x, t) * step_size

        return x

class SmoothedClosedFM(nn.Module):
    def __init__(self, train_samples, gamma=0.01, num_noise=100):
        super(SmoothedClosedFM, self).__init__()
        self.train_samples = train_samples  # (N, C, H, W)
        self.gamma = gamma
        self.num_noise = num_noise
    
    def compute_w(self, xt, t):
        temp = 2 * (1 - t) ** 2
        x_query = xt.unsqueeze(0)  # (1, B, C, H, W)
        x_ref = self.train_samples.unsqueeze(1)  # (N, 1, C, H, W)
        # all t must be same
        assert torch.all(t == t[0]), "All t values must be the same for computing weights." 
        l2_dist = torch.sum((x_query - t[0] * x_ref) ** 2, dim=(2, 3, 4))  # (N, B)
        # apply softmax to get weights
        w = torch.softmax(-l2_dist / temp, dim=0).transpose(0, 1)  # (B, N)
        return w  # (B, N)
    
    @torch.no_grad()
    def get_optimal_v(self, xt, t):
        w = self.compute_w(xt, t) # (B, N)
        x_ref = self.train_samples.unsqueeze(0)  # (1, N, C, H, W)
        x_query = xt.unsqueeze(1)  # (B, 1, C, H, W)
        
        t = t[0]
        w = w.view(w.shape[0], w.shape[1], 1, 1, 1)  # (B, N, 1, 1, 1)
        v_opt = torch.sum(w * (x_ref - x_query)/(1-t), dim=1)  # (B, C, H, W)
        return v_opt  # (B, C, H, W)

    @torch.no_grad()
    def sample(self, x0, steps: int, start_t: float=0.0, **model_kwargs):
        # - define timesteps
        timesteps = torch.linspace(start_t, 1., steps + 1).to(x0.device)
        step_size = (1.0 - start_t) / steps
        
        x_in = x0
        for i in range(steps):
            # print(f"Sampling step {i+1}/{steps}...")
            t_in = timesteps[i]
            t_in = t_in.expand(x_in.shape[0])
            
            # smoothing by adding noise into x_in
            v_all = [self.get_optimal_v(x_in, t_in)]
            for _ in range(self.num_noise):
                noise = torch.randn_like(x_in) * self.gamma
                v_all.append(self.get_optimal_v(x_in + noise, t_in))
            v_avg = torch.mean(torch.stack(v_all, dim=0), dim=0)
            
            x_in = x_in + v_avg * step_size
        samples = x_in
        return samples

class LocalizedClosedFM(nn.Module):
    def __init__(
        self,
        train_samples,
        query_batch_size=4096,
        ref_batch_size=4096,
        store_dtype=None,
        bandwidth_min=0.,
    ):
        super().__init__()

        assert train_samples.ndim == 4
        N, C, H, W = train_samples.shape

        # Important:
        # (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
        train_flat = (
            train_samples
            .detach()
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, C)
        )

        if store_dtype is not None:
            train_flat = train_flat.to(store_dtype)

        # model.to(device) で一緒に移動する
        self.register_buffer("train_samples", train_flat, persistent=False)

        self.query_batch_size = query_batch_size
        self.ref_batch_size = ref_batch_size
        
        self.bandwidth_min = bandwidth_min
        
        self.path = build_scheduler("affine_prob", {})
        
    def perturb(self, x1, x0, t):
        """Perturb the data points for training.
        Args:
            x1 (torch.Tensor): The ending points.
            x0 (torch.Tensor): The starting points.
            t (torch.Tensor): The time steps.
        Returns:
            torch.Tensor: The perturbed points.
        """
        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        return path_sample.x_t


    @torch.no_grad()
    def get_optimal_v(self, xt, t):
        """
        Computes

            v(x_t, t) = ( E_w[x_ref] - x_t ) / (1 - t)

        where

            w_i ∝ exp( -||x_t - t x_ref_i||^2 / (2(1-t)^2) )

        without materializing w of shape (B*H*W, N*H*W).
        """

        assert xt.ndim == 4
        B, C, H, W = xt.shape
        device = xt.device

        assert torch.all(t == t[0]), \
            "All t values must be the same for computing weights."

        t0 = t.flatten()[0].to(device=device, dtype=torch.float32)
        one_minus_t = 1.0 - t0

        if one_minus_t <= 0:
            raise ValueError("t must be strictly smaller than 1.")

        temp = 2.0 * (one_minus_t.square() + self.bandwidth_min ** 2)

        # (B, C, H, W) -> (B*H*W, C)
        x_query_all = (
            xt
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, C)
        )

        x_ref_all = self.train_samples
        if x_ref_all.device != device:
            raise RuntimeError(
                "train_samples is on a different device. "
                "Call model.to(xt.device) before sampling."
            )

        Q = x_query_all.shape[0]
        R = x_ref_all.shape[0]

        out = torch.empty((Q, C), device=device, dtype=xt.dtype)

        for q_start in range(0, Q, self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, Q)

            q = x_query_all[q_start:q_end].to(torch.float32)
            q_size = q.shape[0]

            # Online log-sum-exp state
            # m: running max logit
            # z: running sum exp(logit - m)
            # s: running weighted sum of x_ref
            m = torch.full(
                (q_size, 1),
                -torch.inf,
                device=device,
                dtype=torch.float32,
            )
            z = torch.zeros(
                (q_size, 1),
                device=device,
                dtype=torch.float32,
            )
            s = torch.zeros(
                (q_size, C),
                device=device,
                dtype=torch.float32,
            )

            q_norm = (q * q).sum(dim=1, keepdim=True)

            for r_start in range(0, R, self.ref_batch_size):
                r_end = min(r_start + self.ref_batch_size, R)

                r = x_ref_all[r_start:r_end].to(torch.float32)

                r_norm = (r * r).sum(dim=1).unsqueeze(0)

                # score = -||q - t r||^2 / temp
                #
                # ||q - t r||^2
                # = ||q||^2 - 2t q r^T + t^2 ||r||^2
                #
                # score
                # = (-||q||^2 + 2t q r^T - t^2 ||r||^2) / temp
                scores = (
                    2.0 * t0 * (q @ r.T)
                    - q_norm
                    - t0.square() * r_norm
                ) / temp

                chunk_max = scores.max(dim=1, keepdim=True).values
                m_new = torch.maximum(m, chunk_max)

                old_scale = torch.exp(m - m_new)

                # scores becomes exp(scores - m_new)
                scores.sub_(m_new)
                scores.exp_()

                z = z * old_scale + scores.sum(dim=1, keepdim=True)
                s = s * old_scale + scores @ r

                m = m_new

                # optional: release references earlier
                del r, r_norm, scores, chunk_max, m_new, old_scale

            mean_ref = s / z.clamp_min(1e-30)

            v = (mean_ref - q) / one_minus_t
            out[q_start:q_end] = v.to(dtype=xt.dtype)

            del q, q_norm, m, z, s, mean_ref, v

        v_opt = (
            out
            .view(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return v_opt

    @torch.no_grad()
    def sample(
        self,
        x0,
        steps: int,
        start_t: float = 0.0,
        return_intermediate: bool = False,
        **model_kwargs,
    ):
        timesteps = torch.linspace(
            start_t,
            1.0,
            steps + 1,
            device=x0.device,
            dtype=x0.dtype,
        )

        step_size = (1.0 - start_t) / steps

        x_in = x0
        v_traj = []

        for i in range(steps):
            t_in = timesteps[i].expand(x_in.shape[0])
            opt_v = self.get_optimal_v(x_in, t_in)

            x_in = x_in + opt_v * step_size

            if return_intermediate:
                v_traj.append(opt_v)

        if return_intermediate:
            return x_in, v_traj

        return x_in
  
# class LocalizedClosedFM(nn.Module):
#     def __init__(self, train_samples, dist_batch_size=128):
#         super(LocalizedClosedFM, self).__init__()
#         N, C, H, W = train_samples.shape
#         self.train_samples = train_samples.reshape(N, H, W, C).view(-1, C)  # (N, C, H, W) -> (N*H*W, C)
#         self.dist_batch_size = dist_batch_size
    
#     def compute_w(self, xt, t):
#         temp = 2 * (1 - t) ** 2
#         B, C, H, W = xt.shape
#         x_query = xt.reshape(B, H, W, C).view(-1, C).unsqueeze(1)  # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, 1, C)
#         x_ref = self.train_samples.unsqueeze(0) # (1, N*H*W, C) 
#         # all t must be same
#         assert torch.all(t == t[0]), "All t values must be the same for computing weights." 
#         x_ref_split = torch.split(x_ref, self.dist_batch_size, dim=1)  # split reference into chunks
#         l2_dist_list = []
#         for x_ref_chunk in x_ref_split:
#             l2_dist_chunk = torch.sum((x_query - t[0] * x_ref_chunk) ** 2, dim=2)  # (B*H*W, chunk_size)
#             l2_dist_list.append(l2_dist_chunk)
#         l2_dist = torch.cat(l2_dist_list, dim=1)  # (B*H*W, N*H*W)
#         # l2_dist = torch.sum((x_query - t[0] * x_ref) ** 2, dim=2)  # (B*H*W, N*H*W)
#         # apply softmax to get weights
#         w = torch.softmax(-l2_dist / temp, dim=-1)
#         return w  # (B*H*W, N*H*W)
    
#     @torch.no_grad()
#     def get_optimal_v(self, xt, t):
#         w = self.compute_w(xt, t) # (B*H*W, N*H*W)
        
#         B, C, H, W = xt.shape
#         x_ref = self.train_samples.unsqueeze(0)  # (1, N*H*W, C)
#         x_query = xt.reshape(B, H, W, C).view(-1, C).unsqueeze(1)  # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, 1, C)
        
#         t = t[0]
        
#         w = w.unsqueeze(-1)  # (B*H*W, N*H*W, 1)
#         x_ref_split = torch.split(x_ref, self.dist_batch_size, dim=1)  # split reference into chunks
#         v_opt_list = []
#         for x_ref_chunk in x_ref_split:
#             v_opt_chunk = torch.sum(w * (x_ref_chunk - x_query)/(1-t), dim=1)  # (B*H*W, C)
#             v_opt_list.append(v_opt_chunk)
#         v_opt = torch.sum(v_opt_list, dim=0)  # (B*H*W, C)
        
#         # v_opt = torch.sum(w * (x_ref - x_query)/(1-t), dim=1)  # (B*H*W, C)
#         v_opt = v_opt.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
#         return v_opt  # (B, C, H, W)

#     @torch.no_grad()
#     def sample(self, x0, steps: int, start_t: float=0.0, return_intermediate: bool=False, **model_kwargs):
#         # - define timesteps
#         timesteps = torch.linspace(start_t, 1., steps + 1).to(x0.device)
#         step_size = (1.0 - start_t) / steps
        
#         x_in = x0
#         v_traj = []
#         for i in range(steps):
#             # print(f"Sampling step {i+1}/{steps}...")
#             t_in = timesteps[i]
#             t_in = t_in.expand(x_in.shape[0])
#             opt_v = self.get_optimal_v(x_in, t_in)
            
#             x_in = x_in + opt_v * step_size
#             v_traj.append(opt_v)
#         samples = x_in
#         if return_intermediate:
#             return samples, v_traj
#         return samples

class ClosedFM(nn.Module):
    def __init__(self, train_samples):
        super(ClosedFM, self).__init__()
        self.train_samples = train_samples  # (N, C, H, W)
    
    def compute_w(self, xt, t):
        temp = 2 * (1 - t) ** 2
        x_query = xt.unsqueeze(0)  # (1, B, C, H, W)
        x_ref = self.train_samples.unsqueeze(1)  # (N, 1, C, H, W)
        # all t must be same
        assert torch.all(t == t[0]), "All t values must be the same for computing weights." 
        l2_dist = torch.sum((x_query - t[0] * x_ref) ** 2, dim=(2, 3, 4))  # (N, B)
        # apply softmax to get weights
        w = torch.softmax(-l2_dist / temp, dim=0).transpose(0, 1)  # (B, N)
        return w  # (B, N)
    
    @torch.no_grad()
    def get_optimal_v(self, xt, t):
        w = self.compute_w(xt, t) # (B, N)
        x_ref = self.train_samples.unsqueeze(0)  # (1, N, C, H, W)
        x_query = xt.unsqueeze(1)  # (B, 1, C, H, W)
        
        t = t[0]
        w = w.view(w.shape[0], w.shape[1], 1, 1, 1)  # (B, N, 1, 1, 1)
        v_opt = torch.sum(w * (x_ref - x_query)/(1-t), dim=1)  # (B, C, H, W)
        return v_opt  # (B, C, H, W)

    @torch.no_grad()
    def sample(self, x0, steps: int, start_t: float=0.0, return_intermediate: bool=False, **model_kwargs):
        # - define timesteps
        timesteps = torch.linspace(start_t, 1., steps + 1).to(x0.device)
        step_size = (1.0 - start_t) / steps
        
        x_in = x0
        v_traj = []
        for i in range(steps):
            # print(f"Sampling step {i+1}/{steps}...")
            t_in = timesteps[i]
            t_in = t_in.expand(x_in.shape[0])
            opt_v = self.get_optimal_v(x_in, t_in)
            x_in = x_in + opt_v * step_size
            v_traj.append(opt_v)
        samples = x_in
        if return_intermediate:
            return samples, v_traj
        return samples
    
class SmoothedClosedFM(LocalizedClosedFM):
    """Optimal velocity field for the Gaussian-smoothed empirical distribution
    p_sigma = (1/N) sum_i N(x_i, sigma^2 I).

    sigma=0 reduces exactly to LocalizedClosedFM.
    """

    def __init__(self, train_samples, sigma=0.1, **kwargs):
        kwargs.pop("bandwidth_min", None)   # 役割が sigma に置き換わる
        super().__init__(train_samples, **kwargs)
        self.sigma = float(sigma)

    @torch.no_grad()
    def get_optimal_v(self, xt, t):
        B, C, H, W = xt.shape
        device = xt.device

        assert torch.all(t == t[0])
        t0 = t.flatten()[0].to(device=device, dtype=torch.float32)
        one_minus_t = 1.0 - t0

        sigma2 = self.sigma ** 2
        gamma2 = one_minus_t.square() + t0.square() * sigma2   # γ_t²
        if gamma2 <= 0:
            raise ValueError("gamma_t^2 must be positive (increase sigma or use t < 1).")

        temp = 2.0 * gamma2                                    # 変更点 (1)
        c_t = (t0 * sigma2 - one_minus_t) / gamma2             # 変更点 (2) の係数

        x_query_all = xt.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x_ref_all = self.train_samples
        Q, R = x_query_all.shape[0], x_ref_all.shape[0]
        out = torch.empty((Q, C), device=device, dtype=xt.dtype)

        for q_start in range(0, Q, self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, Q)
            q = x_query_all[q_start:q_end].to(torch.float32)
            q_size = q.shape[0]

            m = torch.full((q_size, 1), -torch.inf, device=device, dtype=torch.float32)
            z = torch.zeros((q_size, 1), device=device, dtype=torch.float32)
            s = torch.zeros((q_size, C), device=device, dtype=torch.float32)
            q_norm = (q * q).sum(dim=1, keepdim=True)

            for r_start in range(0, R, self.ref_batch_size):
                r = x_ref_all[r_start:min(r_start + self.ref_batch_size, R)].to(torch.float32)
                r_norm = (r * r).sum(dim=1).unsqueeze(0)

                scores = (2.0 * t0 * (q @ r.T) - q_norm - t0.square() * r_norm) / temp

                chunk_max = scores.max(dim=1, keepdim=True).values
                m_new = torch.maximum(m, chunk_max)
                old_scale = torch.exp(m - m_new)
                scores.sub_(m_new); scores.exp_()

                z = z * old_scale + scores.sum(dim=1, keepdim=True)
                s = s * old_scale + scores @ r
                m = m_new
                del r, r_norm, scores, chunk_max, m_new, old_scale

            mean_ref = s / z.clamp_min(1e-30)                  # x̄

            # v = x̄ + c_t (x_t - t x̄)      ← 変更点 (2)
            v = mean_ref + c_t * (q - t0 * mean_ref)
            out[q_start:q_end] = v.to(dtype=xt.dtype)
            del q, q_norm, m, z, s, mean_ref, v

        return out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    
class WrappedModel(nn.Module):
    """A wrapper for the velocity model to handle additional conditioning information."""
    def __init__(self, model: nn.Module, path, pred_type: str, cfg_interval=[0.1, 1.0], cfg_scale=1.0, eps=0.05):
        super(WrappedModel, self).__init__()
        self.model = model
        self.path = path
        self.pred_type = pred_type
        self.cfg_interval = cfg_interval
        self.cfg_scale = cfg_scale
        self.eps = eps

    @staticmethod
    def _batch_time(t: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=device, dtype=dtype)
        else:
            t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"Expected scalar time or shape ({batch_size},), got {tuple(t.shape)}."
            )
        return t
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None, **extras) -> torch.Tensor:
        t = self._batch_time(t, x.shape[0], x.device, x.dtype)
        assert self.cfg_scale == 1.0 or y is not None, "Classifier-free guidance requires conditioning information y."
        
        if self.cfg_scale != 1.0:
            extras.update({'cfg_scale': self.cfg_scale, 'cfg_interval': self.cfg_interval})
            model_out = self.model.forward_with_cfg(x, t, y=y, **extras)
        else:
            model_out = self.model(x, t, y=y, **extras)
        
        if self.pred_type == 'data':
            out = self.data_to_velocity(x, t, model_out)
        elif self.pred_type == 'noise':
            out = self.noise_to_velocity(x, t, model_out)
        else:  # velocity
            out = model_out
        return out

    def noise_to_velocity(self, x_t: torch.Tensor, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Convert noise prediction to velocity field.
        Args:
            x_t (torch.Tensor): The intermediate points along the path.
            t (torch.Tensor): The time steps.
            out (torch.Tensor): The noise prediction from the model.
        Returns:
            torch.Tensor: The velocity field at x_t.
        """
        return self.path.epsilon_to_velocity(epsilon=out, x_t=x_t, t=t)
    
    def data_to_velocity(self, x_t: torch.Tensor, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Convert data prediction to velocity field.
        Args:
            x_t (torch.Tensor): The intermediate points along the path.
            t (torch.Tensor): The time steps.
            out (torch.Tensor): The data prediction from the model.
        Returns:
            torch.Tensor: The velocity field at x_t.
        """
        return self.path.target_to_velocity(x_1=out, x_t=x_t, t=t)

class VelocityField(nn.Module):
    def __init__(self, model: nn.Module, input_sz: tuple, scheduler_name: str,  solver_name: str, \
        pred_type:str='velocity', loss_type:str='velocity', loss_fn:str='mse', train_steps: int = -1, 
        t_scheduler_train: str = 'linear', t_scheduler_infer: str = 'linear', t_mu: float=0.0, t_sigma: float=1.0, \
        cfg_interval: list = [0.1, 1.0], cfg_scale: float = 1.0, div_eps: float=0.05, scheduler_params: dict=None, solver_params: dict = None
    ):
        """Velocity field module for flow matching.
        Args:
            model (nn.Module): The neural network model representing the velocity field.
            input_sz (tuple): The size of the input data.
            scheduler_name (str): The name of the scheduler to use.
            solver_name (str): The name of the solver to use.
            pred_type (str, optional): Type of prediction ('data', 'noise', 'velocity'). Defaults to 'velocity'.
            loss_type (str, optional): Type of loss ('data', 'noise', 'velocity'). Defaults to 'velocity'.
            train_steps (int, optional): Number of training steps. Defaults to -1.
            t_scheduler_train (str, optional): Name of the t scheduler for training. Defaults to 'linear'.
            t_scheduler_infer (str, optional): Name of the t scheduler for inference. Defaults to 'linear'.
            t_mu (float, optional): Mean of the Gaussian distribution for sampling t (the fixed t itself when t_scheduler_train='constant'). Defaults to 0.0.
            t_sigma (float, optional): Standard deviation of the Gaussian distribution for sampling t. Defaults to 1.0.
            cfg_interval (list, optional): Time interval which use classifier-free guidance. Defaults to [0.1, 1.0].
            cfg_scale (int): Cfg scale value. 
            div_eps (float, optional): Epsilon value for numerical stability in velocity calculation. Defaults to 0.05.
            scheduler_params (dict, optional): Additional parameters for the scheduler. Defaults to None.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
        Usage: 
            - initialization
            vf = VelocityField(model, input_sz, scheduler_name, solver_name, scheduler_params, solver_params)
            - training
            loss = vf(x1, y)
            - sampling
            x0 = torch.randn(batch_size, *input_sz).to(device)
            x1 = vf.sample(x0, y, steps=10)
            - inversion
            x0 = vf.invert(x1, y, steps=10)
            - density estimation
            log_prob = vf.log_prob(x1, y, steps=10, solver='euler', exact=False, hte_acc=10)
        Assumptions:
            - The model takes input of shape (batch_size, *input_sz) and returns output of the same shape.
            - The scheduler and solver are implemented elsewhere and are compatible with this module.
        """
        assert pred_type in PRED_TYPES, f"pred_type must be one of {PRED_TYPES}"
        assert loss_type in LOSS_TYPES, f"loss_type must be one of {LOSS_TYPES}"
        super(VelocityField, self).__init__()

        self.model = model
        self.input_sz = input_sz
        self.pred_type = pred_type
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        self.cfg_interval = cfg_interval
        self.cfg_scale = cfg_scale
        self.div_eps = div_eps
        self.t_scheduler_train = t_scheduler_train
        self.t_scheduler_infer = t_scheduler_infer
        self.t_mu = t_mu
        self.t_sigma = t_sigma
        self.train_steps = train_steps
        self.scheduler_name = scheduler_name
        self.path = build_scheduler(scheduler_name, scheduler_params or {})
        self.is_edm = isinstance(self.path.scheduler, EDMScheduler)
        self.solver = build_solver(model, solver_name, solver_params or {})

    @staticmethod
    def _expand_batch_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            value = value.expand(reference.shape[0])
        if value.ndim != 1 or value.shape[0] != reference.shape[0]:
            raise ValueError(
                f"Expected scalar or batch vector of length {reference.shape[0]}, "
                f"got {tuple(value.shape)}."
            )
        return value.reshape(value.shape[0], *([1] * (reference.ndim - 1)))

    def _backbone_dtype(self, fallback: torch.dtype) -> torch.dtype:
        try:
            dtype = next(self.model.parameters()).dtype
        except StopIteration:
            dtype = fallback
        # EDM's solver carries states in float64, while the network normally
        # remains in float32/bfloat16.
        return torch.float32 if dtype == torch.float64 else dtype

    def _call_model(self, x, t, y=None, *, use_cfg: bool = False, **model_kwargs):
        if use_cfg and self.cfg_scale != 1.0:
            if y is None:
                raise ValueError("Classifier-free guidance requires conditioning labels.")
            if not hasattr(self.model, "forward_with_cfg"):
                raise AttributeError("The wrapped model does not implement forward_with_cfg().")
            extras = dict(model_kwargs)
            extras.update({'cfg_scale': self.cfg_scale, 'cfg_interval': self.cfg_interval})
            return self.model.forward_with_cfg(x, t, y=y, **extras)
        return self.model(x, t, y=y, **model_kwargs)

    def edm_denoise(self, x, sigma, y=None, *, use_cfg: bool = False, **model_kwargs):
        r"""Apply EDM input/output/noise preconditioning and return ``D_theta``."""
        if not self.is_edm:
            raise RuntimeError("edm_denoise() requires scheduler_name='edm'.")

        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        sigma = sigma.to(device=x.device, dtype=x.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(x.shape[0])
        sigma_x = self._expand_batch_scalar(sigma, x)
        sigma_safe = sigma.clamp_min(torch.finfo(sigma.dtype).tiny)

        sigma_data = self.path.scheduler.sigma_data
        c_skip = sigma_data ** 2 / (sigma_x.square() + sigma_data ** 2)
        c_out = sigma_x * sigma_data / torch.sqrt(sigma_x.square() + sigma_data ** 2)
        c_in = torch.rsqrt(sigma_x.square() + sigma_data ** 2)
        c_noise = sigma_safe.log() / 4.0

        model_dtype = self._backbone_dtype(x.dtype)
        f_x = self._call_model(
            (c_in * x).to(model_dtype),
            c_noise.to(model_dtype),
            y=y,
            use_cfg=use_cfg,
            **model_kwargs,
        )
        return c_skip * x + c_out * f_x.to(x.dtype)
        
    def compute_loss(self, x0, x1, path_sample, y=None, **model_kwargs):
        """Compute a representation-consistent affine-path or EDM loss."""
        xt = path_sample.x_t
        t = path_sample.t
        v = path_sample.dx_t

        if self.loss_type == 'edm':
            if not self.is_edm:
                raise ValueError("loss_type='edm' requires scheduler_name='edm'.")
            denoised = self.edm_denoise(xt, t, y=y, **model_kwargs)
            sigma = self._expand_batch_scalar(t.to(xt.dtype), xt)
            sigma_data = self.path.scheduler.sigma_data
            weight = (sigma.square() + sigma_data ** 2) / (sigma * sigma_data).square()
            return (weight * (denoised - x1).square()).mean()

        out = self._call_model(xt, t, y=y, **model_kwargs)

        if self.loss_type == 'velocity':
            target = v
            if self.pred_type == 'velocity':
                pred = out
            elif self.pred_type == 'data':
                pred = self.path.target_to_velocity(x_1=out, x_t=xt, t=t)
            elif self.pred_type == 'noise':
                pred = self.path.epsilon_to_velocity(epsilon=out, x_t=xt, t=t)
        elif self.loss_type == 'data':
            target = x1
            if self.pred_type == 'velocity':
                pred = self.path.velocity_to_target(velocity=out, x_t=xt, t=t)
            elif self.pred_type == 'data':
                pred = out
            elif self.pred_type == 'noise':
                pred = self.path.epsilon_to_target(epsilon=out, x_t=xt, t=t)
        elif self.loss_type == 'noise':
            target = x0
            if self.pred_type == 'velocity':
                pred = self.path.velocity_to_epsilon(velocity=out, x_t=xt, t=t)
            elif self.pred_type == 'data':
                pred = self.path.target_to_epsilon(x_1=out, x_t=xt, t=t)
            elif self.pred_type == 'noise':
                pred = out
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        if self.loss_fn == 'mse':
            return (pred - target).square().mean()
        raise NotImplementedError(f"Loss function {self.loss_fn} not implemented.")

    def forward(self, x1, y=None, **model_kwargs):
        """Sample a path point and evaluate the configured training objective."""
        bs = x1.shape[0]
        device = x1.device
        x0 = torch.randn_like(x1)

        if self.is_edm:
            # EDM samples noise levels directly from log sigma ~ N(P_mean, P_std^2).
            sigma_dtype = x1.dtype if x1.dtype in (torch.float32, torch.float64) else torch.float32
            t = self.path.scheduler.sample_train_sigma(
                bs, device=device, dtype=sigma_dtype
            )
        elif self.t_scheduler_train == 'linear':
            if self.train_steps > 0:
                # A discrete uniform grid in [0, 1), avoiding the singular data endpoint.
                indices = torch.randint(self.train_steps, (bs,), device=device)
                t = indices.to(x1.dtype) / self.train_steps
            else:
                t = torch.rand(bs, device=device, dtype=x1.dtype)
        elif self.t_scheduler_train == 'logistic':
            t = logit_t(bs, device, self.t_mu, self.t_sigma).to(x1.dtype)
        elif self.t_scheduler_train == 'constant':
            # Fixed-timestep ablation: every sample is trained at t = t_mu (raw t, not a logit).
            t = torch.full((bs,), self.t_mu, device=device, dtype=x1.dtype)
        elif self.t_scheduler_train in {'ddpm', 'discrete_uniform'}:
            n_steps = self.train_steps if self.train_steps > 0 else 1000
            s = torch.randint(1, n_steps + 1, (bs,), device=device)
            t = 1.0 - s.to(x1.dtype) / n_steps
        else:
            raise NotImplementedError(
                f"t_scheduler_name {self.t_scheduler_train} not implemented."
            )

        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        return self.compute_loss(x0, x1, path_sample, y=y, **model_kwargs)

    @torch.no_grad()
    def _integrate_edm_sigma_path(
        self,
        x_init,
        sigma_steps,
        y=None,
        return_intermediate: bool = False,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        output_dtype: torch.dtype = None,
        **model_kwargs,
    ):
        r"""Integrate EDM's probability-flow ODE along a supplied sigma grid.

        ``x_init`` must already represent the state at ``sigma_steps[0]``.  In
        contrast to :meth:`sample_edm`, this helper never rescales its input,
        which also makes it suitable for partial denoising/reconstruction from
        an observed state ``x_sigma = x_data + sigma * epsilon``.
        """
        if not self.is_edm:
            raise RuntimeError(
                "_integrate_edm_sigma_path() requires scheduler_name='edm'."
            )
        if sigma_steps.ndim != 1:
            raise ValueError(
                f"sigma_steps must be one-dimensional, got {tuple(sigma_steps.shape)}."
            )
        if sigma_steps.numel() < 2:
            raise ValueError("sigma_steps must contain at least a start and an end value.")
        if not torch.isfinite(sigma_steps).all():
            raise ValueError("sigma_steps contains NaN or infinity.")
        if sigma_steps[0] < 0 or sigma_steps[-1] < 0:
            raise ValueError("EDM noise levels must be non-negative.")
        if not torch.all(sigma_steps[:-1] > sigma_steps[1:]):
            raise ValueError("sigma_steps must be strictly decreasing.")
        if S_churn < 0:
            raise ValueError(f"S_churn must be non-negative, got {S_churn}.")
        if S_noise < 0:
            raise ValueError(f"S_noise must be non-negative, got {S_noise}.")

        # EDM evaluates the network in its native precision but carries the ODE
        # state in float64 to reduce accumulation error.
        public_dtype = x_init.dtype if output_dtype is None else output_dtype
        sigma_steps = sigma_steps.to(device=x_init.device, dtype=torch.float64)
        x_next = x_init.to(torch.float64)
        trajectory = [x_next.to(public_dtype)] if return_intermediate else None
        n_updates = sigma_steps.numel() - 1

        for sigma_cur, sigma_next in zip(sigma_steps[:-1], sigma_steps[1:]):
            x_cur = x_next

            use_churn = bool(((sigma_cur >= S_min) & (sigma_cur <= S_max)).item())
            gamma = (
                min(S_churn / n_updates, 2.0 ** 0.5 - 1.0)
                if use_churn
                else 0.0
            )
            sigma_hat = sigma_cur * (1.0 + gamma)

            if gamma > 0.0:
                noise_scale = torch.sqrt(
                    (sigma_hat.square() - sigma_cur.square()).clamp_min(0)
                )
                x_hat = (
                    x_cur
                    + noise_scale * S_noise * torch.randn_like(x_cur)
                )
            else:
                # Avoid consuming RNG state during deterministic reconstruction.
                x_hat = x_cur

            sigma_hat_batch = sigma_hat.expand(x_hat.shape[0])
            denoised = self.edm_denoise(
                x_hat,
                sigma_hat_batch,
                y=y,
                use_cfg=True,
                **model_kwargs,
            ).to(torch.float64)
            d_cur = (x_hat - denoised) / sigma_hat
            x_euler = x_hat + (sigma_next - sigma_hat) * d_cur

            # Heun correction is valid only at a positive next noise level.
            # The terminal sigma=0 update remains Euler, as in EDM Algorithm 2.
            if bool((sigma_next > 0).item()):
                sigma_next_batch = sigma_next.expand(x_euler.shape[0])
                denoised_next = self.edm_denoise(
                    x_euler,
                    sigma_next_batch,
                    y=y,
                    use_cfg=True,
                    **model_kwargs,
                ).to(torch.float64)
                d_prime = (x_euler - denoised_next) / sigma_next
                x_next = x_hat + (sigma_next - sigma_hat) * (
                    0.5 * d_cur + 0.5 * d_prime
                )
            else:
                x_next = x_euler

            if return_intermediate:
                trajectory.append(x_next.to(public_dtype))

        result = x_next.to(public_dtype)
        return (result, trajectory) if return_intermediate else result

    def _edm_partial_sigma_grid(
        self,
        sigma_start,
        steps: int,
        *,
        device,
        sigma_min: float = None,
        rho: float = None,
    ) -> torch.Tensor:
        """Build a descending Karras grid from an arbitrary sigma to zero."""
        if not self.is_edm:
            raise RuntimeError(
                "_edm_partial_sigma_grid() requires scheduler_name='edm'."
            )
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")

        sigma_start_tensor = torch.as_tensor(
            sigma_start, device=device, dtype=torch.float64
        )
        if sigma_start_tensor.numel() != 1:
            raise ValueError(
                "sigma_start must be scalar; all samples in a denoising batch "
                "must use the same starting noise level."
            )
        sigma_start_value = float(sigma_start_tensor.item())
        if not torch.isfinite(sigma_start_tensor).item():
            raise ValueError("sigma_start must be finite.")
        if sigma_start_value < 0:
            raise ValueError(
                f"sigma_start must be non-negative, got {sigma_start_value}."
            )
        if sigma_start_value == 0:
            # The caller handles this identity case without evaluating the model.
            return torch.zeros(1, device=device, dtype=torch.float64)

        scheduler = self.path.scheduler
        sigma_min = scheduler.sigma_min if sigma_min is None else float(sigma_min)
        rho = scheduler.rho if rho is None else float(rho)
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}.")
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}.")

        # A single update is exactly D_theta(x_sigma, sigma) when churn is off.
        # The same fallback is numerically preferable when sigma_start is already
        # below the configured terminal positive noise level.
        if steps == 1 or sigma_start_value <= sigma_min:
            return torch.tensor(
                [sigma_start_value, 0.0], device=device, dtype=torch.float64
            )

        return scheduler.time_grid(
            steps,
            device=device,
            dtype=torch.float64,
            sigma_min=sigma_min,
            sigma_max=sigma_start_value,
            rho=rho,
            append_zero=True,
        )

    @torch.no_grad()
    def denoise_edm(
        self,
        x_sigma,
        sigma_start,
        y=None,
        steps: int = 18,
        return_intermediate: bool = False,
        sigma_min: float = None,
        rho: float = None,
        S_churn: float = 0.0,
        S_min: float = None,
        S_max: float = None,
        S_noise: float = None,
        **model_kwargs,
    ):
        r"""Denoise a state from an arbitrary EDM noise level to sigma=0.

        The input is assumed to follow

        .. math:: x_{\sigma_s} = x_\mathrm{data} + \sigma_s\epsilon.

        Unlike :meth:`sample_edm`, ``x_sigma`` is already scaled/noised and is
        therefore passed to the solver unchanged.  Deterministic reconstruction
        is the default: ``S_churn=0`` injects no additional noise along the path.
        """
        if not self.is_edm:
            raise RuntimeError("denoise_edm() requires scheduler_name='edm'.")

        sigma_steps = self._edm_partial_sigma_grid(
            sigma_start,
            steps,
            device=x_sigma.device,
            sigma_min=sigma_min,
            rho=rho,
        )
        if sigma_steps.numel() == 1:
            result = x_sigma.clone()
            if return_intermediate:
                return result, [result]
            return result

        scheduler = self.path.scheduler
        S_min = scheduler.S_min if S_min is None else float(S_min)
        S_max = scheduler.S_max if S_max is None else float(S_max)
        S_noise = scheduler.S_noise if S_noise is None else float(S_noise)
        return self._integrate_edm_sigma_path(
            x_sigma,
            sigma_steps,
            y=y,
            return_intermediate=return_intermediate,
            S_churn=float(S_churn),
            S_min=S_min,
            S_max=S_max,
            S_noise=S_noise,
            **model_kwargs,
        )

    @torch.no_grad()
    def reconstruct_edm(
        self,
        x_data,
        sigma_start,
        y=None,
        steps: int = 18,
        noise=None,
        generator=None,
        return_intermediate: bool = False,
        **denoise_kwargs,
    ):
        r"""Add noise at ``sigma_start`` and denoise the observation to zero.

        Supplying ``noise`` makes the reconstruction exactly repeatable.  When
        omitted, one independent standard-normal tensor is sampled per element.
        With ``return_intermediate=True``, the first trajectory element is the
        constructed noisy state ``x_data + sigma_start * noise``.
        """
        if not self.is_edm:
            raise RuntimeError("reconstruct_edm() requires scheduler_name='edm'.")

        sigma = torch.as_tensor(
            sigma_start, device=x_data.device, dtype=x_data.dtype
        )
        if sigma.numel() != 1:
            raise ValueError("sigma_start must be scalar.")
        if not torch.isfinite(sigma).item() or sigma.item() < 0:
            raise ValueError(
                f"sigma_start must be finite and non-negative, got {sigma.item()}."
            )

        if noise is None:
            noise = torch.randn(
                x_data.shape,
                device=x_data.device,
                dtype=x_data.dtype,
                generator=generator,
            )
        else:
            if noise.shape != x_data.shape:
                raise ValueError(
                    f"noise must have shape {tuple(x_data.shape)}, "
                    f"got {tuple(noise.shape)}."
                )
            noise = noise.to(device=x_data.device, dtype=x_data.dtype)

        x_sigma = x_data + sigma * noise
        return self.denoise_edm(
            x_sigma,
            sigma_start=sigma,
            y=y,
            steps=steps,
            return_intermediate=return_intermediate,
            **denoise_kwargs,
        )

    @torch.no_grad()
    def sample_edm(
        self,
        latents,
        y=None,
        steps: int = 18,
        return_intermediate: bool = False,
        sigma_min: float = None,
        sigma_max: float = None,
        rho: float = None,
        S_churn: float = None,
        S_min: float = None,
        S_max: float = None,
        S_noise: float = None,
        **model_kwargs,
    ):
        """EDM Algorithm 2 (Euler predictor plus second-order Heun correction)."""
        if not self.is_edm:
            raise RuntimeError("sample_edm() requires scheduler_name='edm'.")

        scheduler = self.path.scheduler
        sigma_min = scheduler.sigma_min if sigma_min is None else float(sigma_min)
        sigma_max = scheduler.sigma_max if sigma_max is None else float(sigma_max)
        rho = scheduler.rho if rho is None else float(rho)
        S_churn = scheduler.S_churn if S_churn is None else float(S_churn)
        S_min = scheduler.S_min if S_min is None else float(S_min)
        S_max = scheduler.S_max if S_max is None else float(S_max)
        S_noise = scheduler.S_noise if S_noise is None else float(S_noise)

        sigma_steps = scheduler.time_grid(
            steps,
            device=latents.device,
            dtype=torch.float64,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            append_zero=True,
        )

        x_init = latents.to(torch.float64) * sigma_steps[0]
        return self._integrate_edm_sigma_path(
            x_init,
            sigma_steps,
            y=y,
            return_intermediate=return_intermediate,
            S_churn=S_churn,
            S_min=S_min,
            S_max=S_max,
            S_noise=S_noise,
            output_dtype=latents.dtype,
            **model_kwargs,
        )

    def sample(self, x0, y, steps: int, return_intermediate: bool=False, solver_name: str='euler', solver_params: dict=None, start_t: float=0.0, **model_kwargs):
        """Sample from the velocity field using the specified solver.
        Args:
            x0 (torch.Tensor): The starting points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            return_intermediate (bool, optional): Whether to return intermediate results. Defaults to False.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            start_t (float, optional): The starting time for sampling. Defaults to 0.0.
        Returns:
            torch.Tensor: The sampled points.
        Usage: 
            x0 = torch.randn(batch_size, *input_sz).to(device)
            x1 = vf.sample(x0, y, steps=10)  # (batch_size, *input_sz)
            x1_inter = vf.sample(x0, y, steps=10, return_intermediate=True)  # List of intermediate results
        """
        
        if self.is_edm:
            if start_t != 0.0:
                raise ValueError(
                    "For EDM, sample() expects standard-normal latents and start_t=0. "
                    "Use sample_edm() directly to override sigma_min/sigma_max."
                )
            edm_params = dict(solver_params or {})
            return self.sample_edm(
                x0,
                y=y,
                steps=steps,
                return_intermediate=return_intermediate,
                **edm_params,
                **model_kwargs,
            )

        # - build solver
        model = WrappedModel(
            self.model,
            self.path,
            self.pred_type,
            cfg_interval=self.cfg_interval,
            cfg_scale=self.cfg_scale,
            eps=self.div_eps,
        )
        # solver = build_solver(model, solver_name, solver_params or {})

        # Fixed-step explicit solvers.  The original implementation accepted
        # ``solver_name`` but always executed Euler; supporting midpoint/Heun
        # is useful for the SI and Score-SDE probability-flow ODE cases while
        # preserving the existing return value (effective velocity per step).
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        method = solver_name.lower()
        if method not in {"euler", "midpoint", "heun", "heun2"}:
            raise ValueError(
                f"Unsupported fixed-step solver '{solver_name}'. "
                "Choose one of: euler, midpoint, heun."
            )

        timesteps = torch.linspace(
            start_t, 1.0, steps + 1, device=x0.device, dtype=torch.float32
        )
        v_traj = []
        x_in = x0
        for i in range(steps):
            t_cur_scalar = timesteps[i]
            t_next_scalar = timesteps[i + 1]
            h = (t_next_scalar - t_cur_scalar).to(x_in.dtype)
            t_cur = t_cur_scalar.expand(x_in.shape[0])

            if method == "euler":
                v_eff = model(x_in, t_cur, y=y, **model_kwargs)
            elif method == "midpoint":
                k1 = model(x_in, t_cur, y=y, **model_kwargs)
                t_mid = (0.5 * (t_cur_scalar + t_next_scalar)).expand(x_in.shape[0])
                k2 = model(x_in + 0.5 * h * k1, t_mid, y=y, **model_kwargs)
                v_eff = k2
            else:  # explicit trapezoidal rule / second-order Heun
                k1 = model(x_in, t_cur, y=y, **model_kwargs)
                t_next = t_next_scalar.expand(x_in.shape[0])
                k2 = model(x_in + h * k1, t_next, y=y, **model_kwargs)
                v_eff = 0.5 * (k1 + k2)

            x_in = x_in + h * v_eff
            v_traj.append(v_eff)

        if return_intermediate:
            return x_in, v_traj
        return x_in
    
    def sample_ddim(self, x0, y, steps: int, start_t: float=0.0, **model_kwargs):
        x = x0.clone()
        timesteps = torch.linspace(start_t, 1., steps + 1).to(x0.device)
        ones = torch.ones_like(x0) 
        for i in range(steps):
            t_cur = timesteps[i] * ones
            t_cur_model = timesteps[i].expand(x.shape[0])
            t_nxt = timesteps[i + 1] * ones
            eps = self.model(x, t_cur_model, y=y, **model_kwargs)
            out_nxt = self.path.scheduler(t_nxt)
            x1_hat = self.path.epsilon_to_target(epsilon=eps, x_t=x, t=t_cur)
            # eta = 0 
            x = out_nxt.alpha_t * x1_hat + out_nxt.sigma_t * eps
        return x
            

    def perturb(self, x1, x0, t):
        """Perturb the data points for training.
        Args:
            x1 (torch.Tensor): The ending points.
            x0 (torch.Tensor): The starting points.
            t (torch.Tensor): The time steps.
        Returns:
            torch.Tensor: The perturbed points.
        """
        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        return path_sample.x_t

    def invert(self, x1, y, steps: int, solver_name: str='euler', solver_params: dict=None, return_intermediate: bool=False, stop_time: float=0.0):
        """Invert the velocity field using the specified solver.
        Args:
            x1 (torch.Tensor): The ending points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            return_intermediate (bool, optional): Whether to return intermediate results. Defaults to False.
            stop_time (float, optional): The stopping time for inversion. Defaults to 0.0.
        Returns:
            torch.Tensor: The inverted points.
        Usage: 
            x1 = torch.randn(batch_size, *input_sz).to(device)
            x0 = vf.invert(x1, y, steps=10)  # (batch_size, *input_sz)
        """
        # - build solver
        model = WrappedModel(
            self.model,
            self.path,
            self.pred_type,
            cfg_interval=self.cfg_interval,
            cfg_scale=self.cfg_scale,
            eps=self.div_eps,
        )
        solver = build_solver(model, solver_name, solver_params or {})

        # - define timesteps
        timesteps = torch.tensor([1., stop_time]).to(x1.device)
        step_size = 1.0 / steps
        
        # - invert with solver
        samples = solver.sample(
            time_grid=timesteps,
            x_init=x1,
            method=solver_name,
            step_size=step_size,
            return_intermediate=True,
            y=y,
        )
        if return_intermediate:
            return samples  # List of intermediate results
        else:
            if isinstance(samples, list):
                return samples[-1]  # Final result
            else:
                return samples
    
    def density(self, x1, y, steps: int, solver_name: str='euler', solver_params: dict=None, exact_divergence: bool=False, hutchinson_samples: int=10, hutchinson_bs: int=20):
        """Estimate the log-probability of the data points using the velocity field.
        Args:
            x1 (torch.Tensor): The data points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            exact_divergence (bool, optional): Whether to use exact computation. Defaults to False.
            hutchinson_samples (int, optional): Accuracy parameter for Hutchinson's trace estimator. Defaults to 10.
        Returns:
            torch.Tensor: The estimated log-probabilities.
        Usage: 
            log_prob = vf.log_prob(x1, y, steps=10)  # (batch_size,)
        """
        # -- build solver
        model = WrappedModel(
            self.model,
            self.path,
            self.pred_type,
            cfg_interval=self.cfg_interval,
            cfg_scale=self.cfg_scale,
            eps=self.div_eps,
        )
        solver = build_solver(model, solver_name, solver_params or {})
        
        # -- define prior log-probability
        gaussian_log_density = spatial_gaussian_log_density
        if not exact_divergence:
            # -- do hutchinson trace estimator in a si
            hutchinson_bs = min(hutchinson_bs, hutchinson_samples)
            hutchinson_step = hutchinson_samples // hutchinson_bs
            
            log_p_acc = 0
            for i in tqdm.tqdm(range(hutchinson_step)):
                Hb = hutchinson_bs
                B, _, h, w = x1.shape
                x1_rep = x1.repeat_interleave(Hb, dim=0)
                y_rep = y.repeat_interleave(Hb, dim=0) if y is not None else None
                _, log_p_rep = solver.compute_likelihood(x_1=x1_rep, method=solver_name, step_size=1.0/steps, exact_divergence=exact_divergence, log_p0=gaussian_log_density, y=y_rep)
                log_p = log_p_rep.view(B, Hb, h, w).mean(dim=1)  
                log_p_acc += log_p
            log_p = log_p_acc / hutchinson_step  # unbiased estimator
        else:
            _, log_p = solver.compute_likelihood(x_1=x1, method=solver_name, step_size=1.0/steps, exact_divergence=exact_divergence, log_p0=gaussian_log_density, y=y)
        return log_p
    
        
        
        
        