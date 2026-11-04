"""Figures: cumulative performance, risk/return and weight history."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BG = "#0F1923"
COLORS = {"Markowitz": "#E63946", "Black-Litterman": "#2196F3", "Naive 1/N": "#9E9E9E"}


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.1, color="white")
    for s in ax.spines.values():
        s.set_color("#33404d")


def plot_performance(daily: dict[str, pd.Series], table: pd.DataFrame, title: str, path: str):
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :]); _style(ax1)
    for name, r in daily.items():
        ax1.plot((1 + r).cumprod(), color=COLORS[name], lw=2, label=name,
                 linestyle="--" if name == "Naive 1/N" else "-")
    ax1.set_title(f"Cumulative total return | {title}", color="white")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0]); _style(ax2)
    for name, row in table.iterrows():
        ax2.scatter(row["Ann. volatility (%)"], row["Ann. return (%)"], s=200, color=COLORS[name], label=name)
    ax2.set_title("Risk vs return", color="white")
    ax2.set_xlabel("Volatility (%)", color="white"); ax2.set_ylabel("Return (%)", color="white")

    ax3 = fig.add_subplot(gs[1, 1]); ax3.axis("off")
    tbl = ax3.table(cellText=table.reset_index().values,
                    colLabels=["Strategy", "Ret. (%)", "Vol. (%)", "Sharpe", "Max DD (%)"], loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)

    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def plot_weights(weights: dict[str, pd.DataFrame], path: str):
    fig, axes = plt.subplots(len(weights), 1, figsize=(14, 5 * len(weights)), facecolor=BG)
    for ax, (name, w) in zip(np.atleast_1d(axes), weights.items()):
        _style(ax)
        ax.stackplot(w.index, (w.values * 100).T, labels=w.columns, alpha=0.85)
        ax.set_title(f"Weights (%) - {name}", color="white")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def save_all(daily, weights, table, title, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plot_performance(daily, table, title, os.path.join(out_dir, "performance.png"))
    plot_weights(weights, os.path.join(out_dir, "weights.png"))
