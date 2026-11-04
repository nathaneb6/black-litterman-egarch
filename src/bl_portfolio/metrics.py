"""Performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(r: pd.Series, rf: float = 0.03, trading_days: int = 252) -> dict:
    """Annualised return, volatility, Sharpe ratio and maximum drawdown of daily returns."""
    ret = r.mean() * trading_days
    vol = r.std() * np.sqrt(trading_days)
    wealth = (1 + r).cumprod()
    mdd = ((wealth - wealth.cummax()) / wealth.cummax()).min()
    return {
        "Ann. return (%)": round(ret * 100, 2),
        "Ann. volatility (%)": round(vol * 100, 2),
        "Sharpe": round((ret - rf) / vol if vol > 0 else 0.0, 3),
        "Max drawdown (%)": round(mdd * 100, 2),
    }


def metrics_table(daily: dict[str, pd.Series], rf: float = 0.03, trading_days: int = 252) -> pd.DataFrame:
    return pd.DataFrame({k: performance_metrics(v, rf, trading_days) for k, v in daily.items()}).T
