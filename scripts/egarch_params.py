"""Full-sample EGARCH-M parameters and one-step-ahead views for every stock.

    python scripts/egarch_params.py --data-dir data/ --out results/egarch_m_parameters.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from bl_portfolio import load_prices  # noqa: E402
from bl_portfolio.egarch_m import egarch_m_parameters  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="results/egarch_m_parameters.csv")
    args = p.parse_args()

    prices = load_prices(args.data_dir)
    rows = {}
    for name in prices.columns:
        try:
            rows[name] = egarch_m_parameters(prices[name])
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAILED] {name}: {e}")

    table = pd.DataFrame(rows).T.rename_axis("asset")
    pd.set_option("display.width", 200)
    print(table.round(5))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
