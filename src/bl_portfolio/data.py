"""Loading Investing.com price exports and building total-return log returns."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .config import UNIVERSE


def _to_float(x):
    """Parse Investing.com number formats ('1,23', '1.2K', '3.4M', '0.5%')."""
    if isinstance(x, str):
        x = x.replace("\xa0", "").replace(" ", "").replace(",", ".").replace("%", "")
        if "K" in x:
            return float(x.replace("K", "")) * 1_000
        if "M" in x:
            return float(x.replace("M", "")) * 1_000_000
    return float(x)


def load_prices(folder: str, universe: list[str] | None = UNIVERSE) -> pd.DataFrame:
    """Load one `<Asset name>.txt` CSV per stock and return aligned closing prices.

    Expected columns (Investing.com export): Date, Dernier (or Close), ...
    """
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt price file found in: {folder}")

    prices = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if universe is not None and name not in universe:
            continue
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        date_col = next(c for c in df.columns if "date" in c.lower())
        price_col = next(c for c in df.columns if "dernier" in c.lower() or "close" in c.lower())
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        prices[name] = df[price_col].apply(_to_float)

    out = pd.DataFrame(prices).ffill().dropna()
    if universe is not None:
        missing = [a for a in universe if a not in out.columns]
        if missing:
            print(f"WARNING: missing price files for {missing}")
    return out[sorted(out.columns)]


def total_return_log_returns(prices: pd.DataFrame, div_yield: float, trading_days: int) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1}) + div_yield / trading_days   (report, eq. 14)."""
    return np.log(prices / prices.shift(1)).dropna() + div_yield / trading_days


def synthetic_prices(n_days: int = 4900, start: str = "2007-01-02", seed: int = 42,
                     names: list[str] = UNIVERSE) -> pd.DataFrame:
    """Correlated GBM prices with the same shape as the real data (demo / tests only)."""
    rng = np.random.default_rng(seed)
    n = len(names)
    vols = rng.uniform(0.18, 0.40, n) / np.sqrt(252)
    drifts = rng.uniform(0.02, 0.12, n) / 252
    corr = 0.35 * np.ones((n, n)) + 0.65 * np.eye(n)
    cov = np.outer(vols, vols) * corr
    rets = rng.multivariate_normal(drifts, cov, size=n_days)
    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=names)
