"""Portfolio optimisers (long-only, fully invested, capped weights)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def max_sharpe(mu: np.ndarray, Sigma: np.ndarray, rf_daily: float, max_weight: float,
               rng: np.random.Generator | None = None, n_restarts: int = 5) -> np.ndarray:
    """Numerical max-Sharpe portfolio (SLSQP, equal-weight + random Dirichlet starts)."""
    rng = rng or np.random.default_rng()
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        vol = float(np.sqrt(max(w @ Sigma @ w, 1e-14)))
        return -(float(w @ mu) - rf_daily) / vol

    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    best_w, best_f = w0.copy(), np.inf
    for start in [w0] + [rng.dirichlet(np.ones(n)) for _ in range(n_restarts)]:
        try:
            res = minimize(neg_sharpe, start, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"ftol": 1e-12})
            if res.success and res.fun < best_f:
                best_f, best_w = res.fun, res.x.copy()
        except Exception:
            pass

    best_w = np.clip(best_w, 0, max_weight)
    return best_w / best_w.sum() if best_w.sum() > 1e-6 else w0


def markowitz_tangency(mu: np.ndarray, Sigma: np.ndarray, rf_daily: float,
                       max_weight: float) -> np.ndarray:
    """Classical Markowitz baseline: analytic tangency portfolio w ∝ Σ⁻¹(μ - rf).

    Long-only and weight caps are imposed afterwards by clipping + renormalising,
    which is what makes this baseline "bang-bang" (report, section 6.1).
    """
    n = len(mu)
    w = np.linalg.inv(Sigma + np.eye(n) * 1e-10) @ (mu - rf_daily)
    w = w / w.sum() if w.sum() != 0 else np.ones(n) / n
    w = np.clip(w, 0, max_weight)
    return w / w.sum() if w.sum() > 1e-6 else np.ones(n) / n
