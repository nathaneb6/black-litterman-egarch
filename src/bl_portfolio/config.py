"""Global parameters, asset universe and backtest scenarios."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """All tunable parameters of the pipeline (values used in the report)."""

    delta: float = 18.0          # risk-aversion coefficient (best positive value in the report)
    rf: float = 0.03             # annual risk-free rate
    div_yield: float = 0.03      # constant annual dividend yield (total-return adjustment)
    tau: float = 0.05            # fallback prior uncertainty if tau cannot be auto-calibrated
    trading_days: int = 252
    max_weight: float = 0.30     # long-only, max 30 % per asset
    est_window: int = 1000       # rolling training window (~4 years of trading days)
    rebalance_freq: int = 63     # quarterly rebalancing
    seed: int | None = 0         # seed for the optimiser's random restarts (None = unseeded)


# 10 CAC 40 stocks chosen for sector diversification (report, section 5.2)
UNIVERSE = [
    "Air Liquide", "Airbus", "AXA",
    "LVMH", "Sanofi", "TotalEnergies",
    "BNP Paribas", "Schneider Electric", "Danone", "Capgemini",
]

# Market regimes analysed in the report (section 6)
SCENARIOS = {
    "1_pre_crisis":   ("2017-01-01", "2019-12-31"),
    "2_covid_crash":  ("2019-01-01", "2021-03-31"),
    "3_post_covid":   ("2021-03-01", "2024-03-31"),
    "4_full_history": ("2016-01-01", "2025-12-31"),
}
