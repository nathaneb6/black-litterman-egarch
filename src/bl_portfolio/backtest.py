"""Rolling-window backtest: Markowitz vs Black-Litterman vs naive 1/N."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .black_litterman import black_litterman_weights
from .config import Config
from .covariance import ledoit_wolf_constant_corr
from .optimizers import markowitz_tangency


def backtest(returns: pd.DataFrame, cfg: Config, start: str | None = None,
             end: str | None = None, view_cache: dict | None = None, verbose: bool = True):
    """Walk forward through `returns` and rebalance every `cfg.rebalance_freq` days.

    At each rebalancing date t the models are fitted on returns[t-W : t] only
    (no look-ahead), and the weights are held over returns[t : t+freq].
    Transaction costs are ignored.

    Returns (daily_returns: dict[str, Series], weights: dict[str, DataFrame]).
    """
    rng = np.random.default_rng(cfg.seed)
    T, n = returns.shape
    W, F = cfg.est_window, cfg.rebalance_freq
    rf_d = cfg.rf / cfg.trading_days
    w_naive = np.ones(n) / n

    dates = []
    for t in range(W, T, F):
        d = returns.index[t]
        if start and d < pd.to_datetime(start):
            continue
        if end and d > pd.to_datetime(end):
            break
        dates.append(t)
    if not dates:
        raise ValueError(f"No rebalancing date between {start} and {end}")
    if verbose:
        print(f"    {len(dates)} rebalancings | window {W} d | every {F} d | delta = {cfg.delta}")

    out = {"Markowitz": [], "Black-Litterman": [], "Naive 1/N": []}
    idx, w_hist, reb_dates, n_fallback = [], {"Markowitz": [], "Black-Litterman": []}, [], 0

    for t in dates:
        train = returns.iloc[t - W:t]
        test = returns.iloc[t:min(t + F, T)]

        Sigma_lw, S, _ = ledoit_wolf_constant_corr(train.values)
        w_mkz = markowitz_tangency(train.mean().values, S, rf_d, cfg.max_weight)
        try:
            w_bl, _, _ = black_litterman_weights(train, Sigma_lw, cfg, rng=rng, view_cache=view_cache)
        except Exception as e:
            n_fallback += 1
            if verbose:
                print(f"    BL failed on {test.index[0].date()} ({e}) -> 1/N")
            w_bl = w_naive.copy()

        w_hist["Markowitz"].append(w_mkz)
        w_hist["Black-Litterman"].append(w_bl)
        reb_dates.append(test.index[0])

        X = test.values
        out["Markowitz"].extend(X @ w_mkz)
        out["Black-Litterman"].extend(X @ w_bl)
        out["Naive 1/N"].extend(X @ w_naive)
        idx.extend(test.index)

    if n_fallback and verbose:
        print(f"    WARNING: BL fell back to 1/N on {n_fallback} rebalancing(s)")

    daily = {k: pd.Series(v, index=idx) for k, v in out.items()}
    weights = {k: pd.DataFrame(v, index=reb_dates, columns=returns.columns) for k, v in w_hist.items()}
    return daily, weights
