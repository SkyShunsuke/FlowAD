import os
import torch
import numpy as np  
import torch.nn as nn

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