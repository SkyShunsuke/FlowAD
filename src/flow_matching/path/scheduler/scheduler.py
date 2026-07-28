# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union, Optional

import torch

from torch import Tensor


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...


class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""Scheduler for convex paths.

        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from :math:`\kappa_t`.

        Args:
            kappa (Tensor): :math:`\kappa`, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)

        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """CondOT Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa


class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial Scheduler."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."

        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return torch.pow(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__()

    def __call__(self, t: Tensor) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = -(1 - t) * (B - b) - b

        return SchedulerOutput(
            alpha_t=torch.exp(-0.5 * T),
            sigma_t=torch.sqrt(1 - torch.exp(-T)),
            d_alpha_t=-0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T)),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        T = -torch.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = 1 - ((-b + torch.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
        return t


class LinearVPScheduler(Scheduler):
    """Linear Variance Preserving Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=(1 - t**2) ** 0.5,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-t / (1 - t**2) ** 0.5,
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return torch.sqrt(snr**2 / (1 + snr**2))


class CosineScheduler(Scheduler):
    """Cosine Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        pi = torch.pi
        return SchedulerOutput(
            alpha_t=torch.sin(pi / 2 * t),
            sigma_t=torch.cos(pi / 2 * t),
            d_alpha_t=pi / 2 * torch.cos(pi / 2 * t),
            d_sigma_t=-pi / 2 * torch.sin(pi / 2 * t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return 2.0 * torch.atan(snr) / torch.pi

class DiscreteVPScheduler(Scheduler):
    def __init__(
        self,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        betas: Optional[Tensor] = None,
    ) -> None:
        if betas is None:
            betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)
        betas = betas.detach().to(torch.float64).flatten()
        assert torch.all((betas > 0) & (betas < 1)), "betas must lie in (0, 1)."

        self.n_steps = int(betas.numel())
        self.betas = betas
        # Λ_k = -log ᾱ_k (k = 0..N, Λ_0 = 0). 区間 [k, k+1) の傾き dΛ_k = -log(1-β_{k+1}) > 0
        self._dlambda = -torch.log1p(-betas)  # (N,)
        self._lambdas = torch.cat(
            [torch.zeros(1, dtype=torch.float64), torch.cumsum(self._dlambda, dim=0)]
        )  # (N+1,)

    # ------------------------------------------------------------------ utilities
    @property
    def alphas_cumprod(self) -> Tensor:
        """ᾱ_k (k = 0..N, ᾱ_0 = 1) の表, float64, shape (N+1,)."""
        return torch.exp(-self._lambdas)

    def time_grid(
        self, n_sampling_steps: int, device=None, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """DDIM 用の昇順 FM 時刻グリッド (長さ n_sampling_steps+1, 表の行に厳密に乗る).

        n_sampling_steps が n_steps を割り切ると等間隔の部分列になる.
        """
        s = torch.linspace(self.n_steps, 0, n_sampling_steps + 1).round()
        return (1.0 - s / self.n_steps).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ Scheduler API
    def __call__(self, t: Tensor) -> SchedulerOutput:
        lambdas = self._lambdas.to(t.device)
        dlambda = self._dlambda.to(t.device)
        n = self.n_steps

        s = (1.0 - t.to(torch.float64)) * n            # DDPM 時刻 (連続), 0..N
        k = s.floor().clamp(0, n - 1).long()           # 区間 [k, k+1)
        lam = lambdas[k] + (s - k) * dlambda[k]        # Λ(s): 区分線形
        dlam_dt = -n * dlambda[k]                      # dΛ/dt = (dΛ/ds)(ds/dt), ds/dt = -N

        exp_neg = torch.exp(-lam)
        alpha_t = torch.exp(-0.5 * lam)
        sigma_t = torch.sqrt(1.0 - exp_neg)
        return SchedulerOutput(
            alpha_t=alpha_t.to(t.dtype),
            sigma_t=sigma_t.to(t.dtype),
            d_alpha_t=(-0.5 * dlam_dt * alpha_t).to(t.dtype),
            # 注意: t=1 (s=0) では σ_t=0 のため d_sigma_t は発散する (連続 VP と同じ特異性).
            # ε-予測学習や DDIM は微分を使わないので影響しない.
            d_sigma_t=(0.5 * dlam_dt * exp_neg / sigma_t).to(t.dtype),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        """snr = α_t/σ_t から t を返す (公式 flow_matching の Scheduler API 互換).

        e^{-Λ} = snr²/(1+snr²) ⇔ Λ = log(1 + snr⁻²) を区分線形 Λ(s) の逆写像で s に戻す.
        """
        lam = torch.log1p(snr.to(torch.float64) ** (-2))
        lam = lam.clamp(0.0, float(self._lambdas[-1]))
        lambdas = self._lambdas.to(snr.device)
        dlambda = self._dlambda.to(snr.device)
        k = (torch.searchsorted(lambdas, lam, right=True) - 1).clamp(0, self.n_steps - 1)
        frac = (lam - lambdas[k]) / dlambda[k]
        return (1.0 - (k + frac) / self.n_steps).to(snr.dtype)