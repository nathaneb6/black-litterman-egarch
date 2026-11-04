"""Black-Litterman with Ledoit-Wolf prior covariance and EGARCH-M views (report, section 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .egarch_m import egarch_m_view
from .optimizers import max_sharpe


def implied_returns(Sigma: np.ndarray, delta: float, w_mkt: np.ndarray) -> np.ndarray:
    """Equilibrium (CAPM-implied) excess returns: Pi = delta . Sigma . w_mkt."""
    return delta * Sigma @ w_mkt


def posterior(Pi: np.ndarray, Sigma: np.ndarray, Q: np.ndarray, omega: np.ndarray,
              tau: float, P: np.ndarray | None = None, eps: float = 1e-10):
    """Bayesian update.

    mu_BL    = [(tau Σ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(tau Σ)⁻¹ Π + Pᵀ Ω⁻¹ Q]
    Sigma_BL = Σ + [(tau Σ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹
    """
    n = len(Pi)
    P = np.eye(n) if P is None else P
    I = np.eye(n)
    tS_inv = np.linalg.inv(tau * Sigma + I * eps)
    O_inv = np.linalg.inv(np.diag(omega) + np.eye(len(Q)) * eps)
    M = np.linalg.inv(tS_inv + P.T @ O_inv @ P + I * eps)
    mu_bl = M @ (tS_inv @ Pi + P.T @ O_inv @ Q)
    return mu_bl, Sigma + M


def egarch_views(train: pd.DataFrame, cache: dict | None = None,
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One absolute EGARCH-M view per asset. Views do not depend on delta, so
    they can be cached by (asset, last training date) when sweeping delta."""
    n = train.shape[1]
    Q, omega = np.zeros(n), np.zeros(n)
    key_date = train.index[-1]
    for i, col in enumerate(train.columns):
        key = (col, key_date)
        if cache is not None and key in cache:
            Q[i], omega[i] = cache[key]
        else:
            Q[i], omega[i] = egarch_m_view(train[col].values, verbose=verbose)
            if cache is not None:
                cache[key] = (Q[i], omega[i])
    return Q, omega


def black_litterman_weights(train: pd.DataFrame, Sigma_lw: np.ndarray, cfg: Config,
                            rng: np.random.Generator | None = None,
                            view_cache: dict | None = None):
    """Full BL step used at each rebalancing date.

    1. prior Pi from an equal-weight market portfolio
    2. EGARCH-M views Q, Omega
    3. Q bounded to Pi +/- one annual volatility (filters extreme views)
    4. tau auto-calibrated so that Omega / (tau Σ) ≈ 1, Omega floored at tau Σ_ii
    5. max-Sharpe on (mu_BL, Sigma_BL)
    """
    n, D = train.shape[1], cfg.trading_days
    w_mkt = np.ones(n) / n
    Pi = implied_returns(Sigma_lw, cfg.delta, w_mkt)

    Q, omega = egarch_views(train, cache=view_cache)

    vol_ann = np.sqrt(np.diag(Sigma_lw) * D)
    Q = np.clip(Q * D, Pi * D - vol_ann, Pi * D + vol_ann) / D

    omega_valid = omega[omega > 1e-8]
    sigma_med = float(np.median(np.diag(Sigma_lw)))
    if len(omega_valid) > 0 and sigma_med > 1e-12:
        tau = float(np.clip(np.median(omega_valid) / sigma_med, 0.01, 5.0))
    else:
        tau = cfg.tau
    omega = np.maximum(omega, tau * np.diag(Sigma_lw))

    mu_bl, Sigma_bl = posterior(Pi, Sigma_lw, Q, omega, tau)
    w = max_sharpe(mu_bl, Sigma_bl, cfg.rf / D, cfg.max_weight, rng=rng)

    info = {"Pi_ann_%": Pi * D * 100, "Q_ann_%": Q * D * 100,
            "muBL_ann_%": mu_bl * D * 100, "tau": tau}
    return w, mu_bl, info
