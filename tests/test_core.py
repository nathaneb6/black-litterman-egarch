"""Sanity checks that do not require `arch` (run with `pytest`)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bl_portfolio import (implied_returns, ledoit_wolf_constant_corr, markowitz_tangency,  # noqa: E402
                          max_sharpe, posterior, synthetic_prices, total_return_log_returns)


def _returns():
    return total_return_log_returns(synthetic_prices(n_days=1200), 0.03, 252)


def test_ledoit_wolf_is_valid_covariance():
    R = _returns().values
    Sigma, S, lam = ledoit_wolf_constant_corr(R)
    assert 0.0 <= lam <= 1.0
    assert np.allclose(Sigma, Sigma.T)
    assert np.all(np.linalg.eigvalsh(Sigma) > 0)
    assert np.allclose(np.diag(Sigma), np.diag(S))          # target keeps sample variances


def test_posterior_limits():
    Sigma = np.cov(_returns().values, rowvar=False)
    n = len(Sigma)
    Pi = implied_returns(Sigma, 2.5, np.ones(n) / n)
    Q = Pi + 0.001
    mu_vague, _ = posterior(Pi, Sigma, Q, omega=np.full(n, 1e6), tau=0.05)
    mu_sure, _ = posterior(Pi, Sigma, Q, omega=np.full(n, 1e-12), tau=0.05)
    assert np.allclose(mu_vague, Pi, atol=1e-6)   # no confidence in views -> prior
    assert np.allclose(mu_sure, Q, atol=1e-6)     # certain views -> views


def test_optimisers_respect_constraints():
    R = _returns()
    mu, S = R.mean().values, np.cov(R.values, rowvar=False)
    w_bl = max_sharpe(mu, S, 0.03 / 252, 0.30, rng=np.random.default_rng(0))
    w_mkz = markowitz_tangency(mu, S, 0.03 / 252, 0.30)
    for w in (w_bl, w_mkz):
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= -1e-9)
    assert np.all(w_bl <= 0.30 + 1e-6)
    # note: the tangency baseline clips then renormalises, so its cap is only approximate
