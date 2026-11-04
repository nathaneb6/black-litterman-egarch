"""Run the backtest for one or several risk-aversion values and market regimes.

Examples
--------
    python scripts/run_backtest.py --data-dir data/                       # delta = 18, all regimes
    python scripts/run_backtest.py --data-dir data/ --delta 1 10 18 100 -5.5 --scenarios 4_full_history
    python scripts/run_backtest.py --synthetic                            # no data needed (demo)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from bl_portfolio import (SCENARIOS, Config, backtest, load_prices, metrics_table,  # noqa: E402
                          synthetic_prices, total_return_log_returns)
from bl_portfolio.plotting import save_all  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="folder with one '<Asset>.txt' Investing.com export per stock")
    src.add_argument("--synthetic", action="store_true", help="use simulated prices (demo)")
    p.add_argument("--delta", type=float, nargs="+", default=[Config.delta])
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--out", default="results/runs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    base = Config(seed=args.seed)
    prices = synthetic_prices() if args.synthetic else load_prices(args.data_dir)
    returns = total_return_log_returns(prices, base.div_yield, base.trading_days)
    print(f"{returns.shape[1]} assets | {returns.index[0].date()} -> {returns.index[-1].date()}")

    view_cache: dict = {}   # EGARCH-M views do not depend on delta -> fit once, reuse
    rows = []
    for delta in args.delta:
        cfg = replace(base, delta=delta)
        for sc in args.scenarios:
            start, end = SCENARIOS[sc]
            print(f"\n> delta = {delta:g} | {sc} ({start} -> {end})")
            daily, weights = backtest(returns, cfg, start, end, view_cache=view_cache)
            table = metrics_table(daily, cfg.rf, cfg.trading_days)
            print(table.to_string())

            save_all(daily, weights, table, f"{sc} | delta = {delta:g}",
                     os.path.join(args.out, f"delta_{delta:g}", sc))
            rows.append(table.assign(delta=delta, scenario=sc))

    os.makedirs(args.out, exist_ok=True)
    summary = pd.concat(rows).rename_axis("strategy").reset_index()
    summary = summary[["scenario", "delta", "strategy"] + [c for c in summary.columns
                                                           if c not in ("scenario", "delta", "strategy")]]
    path = os.path.join(args.out, "summary.csv")
    summary.to_csv(path, index=False)
    print(f"\nSummary written to {path}")


if __name__ == "__main__":
    main()
