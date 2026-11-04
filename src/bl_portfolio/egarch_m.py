"""EGARCH-M view generation (report, section 4).

Mean equation      : r_t = mu + lambda * sigma_t^2 + eps_t
Variance equation  : EGARCH(1,1) with leverage term, Student-t innovations

The two equations are estimated alternately until (mu, lambda) converge:
1. fit an EGARCH(1,1) on the current residuals  -> sigma_t^2
2. OLS of r_t on [1, sigma_t^2]                   -> (mu, lambda)
3. new residuals = r_t - (mu + lambda * sigma_t^2)

The one-step-ahead forecasts feed Black-Litterman:
    Q_i     = mu + lambda * sigma_{T+1}^2      (view)
    Omega_ii = sigma_{T+1}^2                    (view uncertainty)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _imports():
    try:
        from arch import arch_model
        import statsmodels.api as sm
    except ImportError as e:  # pragma: no cover
        raise ImportError("EGARCH-M requires `pip install arch statsmodels`") from e
    return arch_model, sm


def egarch_m_view(r: np.ndarray, tol: float = 1e-5, max_iter: int = 30,
                  verbose: bool = False) -> tuple[float, float]:
    """Daily (Q_i, sigma2_{T+1}) for one asset from its daily log returns `r`.

    Includes the safeguards used in the backtest: empirical fallback if the fit
    fails or is absurd, Q clipped to +/-4 sigma, and a floor on the variance.
    """
    arch_model, sm = _imports()

    var_emp = float(np.var(r))
    mu_emp = float(np.mean(r))
    if len(r) < 100:
        return mu_emp, var_emp

    r_scaled = pd.Series(r * 100)          # arch is better conditioned on % returns
    resid = r_scaled.copy()

    diff, it = 1.0, 0
    mu_prev, lam_prev = 0.0, 0.0
    sigma2_next, Q_i = var_emp, mu_emp

    try:
        res = None
        while diff > tol and it < max_iter:
            am = arch_model(resid, vol="EGARCH", p=1, o=1, q=1, dist="t", mean="Zero")
            res = am.fit(disp="off", show_warning=False)
            var_t = res.conditional_volatility ** 2

            ols = sm.OLS(r_scaled, sm.add_constant(var_t)).fit()
            mu_est = float(ols.params.iloc[0])
            lam_est = float(ols.params.iloc[1])

            diff = abs(mu_est - mu_prev) + abs(lam_est - lam_prev)
            resid = r_scaled - (mu_est + lam_est * var_t)
            mu_prev, lam_prev = mu_est, lam_est
            it += 1

        if verbose:
            status = "converged" if diff <= tol else "max_iter reached"
            print(f"      EGARCH-M: {status} after {it} iteration(s)")

        if res is not None:
            var_next = float(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
            sigma2_raw = var_next / 10_000
            Q_raw = (mu_est + lam_est * var_next) / 100

            if np.isfinite(sigma2_raw) and 0 < sigma2_raw < var_emp * 50:
                sigma2_next = sigma2_raw

            risk_premium = abs(lam_est * var_next / 100)
            if np.isfinite(Q_raw) and risk_premium < max(10 * abs(mu_emp), 5e-4):
                Q_i = Q_raw
    except Exception:
        if verbose:
            print("      EGARCH-M: fit failed -> empirical fallback")

    sigma_d = np.sqrt(max(sigma2_next, 1e-10))
    Q_i = float(np.clip(Q_i, -4 * sigma_d, 4 * sigma_d))
    omega_min = max(var_emp / 10.0, 1e-8)
    return Q_i, float(max(sigma2_next, omega_min))


def egarch_m_parameters(prices: pd.Series) -> dict:
    """Full-sample EGARCH-M parameters for one asset (exploratory table).

    Single-pass estimation: EGARCH(1,1)-t with constant mean, then OLS for the
    risk premium lambda. Used by `scripts/egarch_params.py`.
    """
    arch_model, sm = _imports()
    r = (np.log(prices / prices.shift(1)) * 100).dropna()

    res = arch_model(r, vol="EGARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    var_t = res.conditional_volatility ** 2
    ols = sm.OLS(r, sm.add_constant(var_t)).fit()
    mu, lam = ols.params.iloc[0], ols.params.iloc[1]

    var_next = res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
    return {
        "Q (r_T+1)": (mu + lam * var_next) / 100,
        "sigma_T+1": np.sqrt(var_next) / 100,
        "Omega (sigma2_T+1)": var_next / 10_000,
        "mu": mu,
        "lambda": lam,
        "omega": res.params["omega"],
        "alpha": res.params["alpha[1]"],
        "gamma": res.params["gamma[1]"],
        "beta": res.params["beta[1]"],
        "nu": res.params["nu"],
    }
