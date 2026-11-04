"""Ledoit-Wolf shrinkage towards a constant-correlation target (report, section 3)."""
from __future__ import annotations

import numpy as np


def ledoit_wolf_constant_corr(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Shrink the sample covariance of returns `R` (T x n).

    Returns (Sigma_shrunk, S_sample, lambda_star) with
    Sigma_shrunk = lambda* . F + (1 - lambda*) . S, where F keeps the sample
    variances and uses the average pairwise correlation off the diagonal.
    """
    T_obs, n = R.shape
    S = np.cov(R, rowvar=False)
    std = np.sqrt(np.diag(S))

    corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)
    rho_bar = corr.sum() / (n * (n - 1))

    F = rho_bar * np.outer(std, std)
    np.fill_diagonal(F, np.diag(S))

    num = ((S ** 2 + np.outer(np.diag(S), np.diag(S))) / T_obs).sum()
    den = float(((S - F) ** 2).sum())
    lam = float(np.clip(num / den, 0.0, 1.0)) if den > 1e-12 else 0.0

    return lam * F + (1 - lam) * S, S, lam
