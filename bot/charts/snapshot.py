"""Matplotlib chart snapshot generation for trade journal."""
from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from bot.indicators.momentum import rsi
from bot.indicators.trend import ema, macd
from bot.indicators.volatility import bollinger_bands

logger = logging.getLogger(__name__)

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "grid.color": "#333",
})


def _plot_candlesticks(ax, df: pd.DataFrame) -> None:
    up = df[df["close"] >= df["open"]]
    down = df[df["close"] < df["open"]]
    width = 0.0025  # for datetime index
    ax.bar(up.index, up["close"] - up["open"], width, bottom=up["open"], color="#26a69a", alpha=0.9)
    ax.bar(up.index, up["high"] - up["close"], width * 0.3, bottom=up["close"], color="#26a69a", alpha=0.9)
    ax.bar(up.index, up["low"] - up["open"], width * 0.3, bottom=up["open"], color="#26a69a", alpha=0.9)
    ax.bar(down.index, down["close"] - down["open"], width, bottom=down["open"], color="#ef5350", alpha=0.9)
    ax.bar(down.index, down["high"] - down["open"], width * 0.3, bottom=down["open"], color="#ef5350", alpha=0.9)
    ax.bar(down.index, down["low"] - down["close"], width * 0.3, bottom=down["close"], color="#ef5350", alpha=0.9)


def generate_entry_chart(
    df: pd.DataFrame,
    entry_price: float,
    symbol: str,
    trade_id: int,
    output_dir: str = "data/charts",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)

    ax_price, ax_rsi, ax_vol = axes

    _plot_candlesticks(ax_price, df)
    ax_price.axhline(entry_price, color="#ffd700", linestyle="--", linewidth=1, label=f"Entry €{entry_price:,.2f}")

    # Overlays
    if len(df) >= 20:
        ax_price.plot(df.index, ema(df["close"], 20), color="#42a5f5", linewidth=0.8, label="EMA20")
    if len(df) >= 50:
        ax_price.plot(df.index, ema(df["close"], 50), color="#ab47bc", linewidth=0.8, label="EMA50")
    if len(df) >= 20:
        bb = bollinger_bands(df["close"])
        ax_price.fill_between(df.index, bb["lower"], bb["upper"], alpha=0.1, color="#42a5f5")

    ax_price.set_title(f"{symbol} Entry — Trade #{trade_id}", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)

    # RSI
    if len(df) >= 14:
        r = rsi(df["close"])
        ax_rsi.plot(df.index, r, color="#ffa726", linewidth=0.8)
        ax_rsi.axhline(70, color="#ef5350", linestyle=":", linewidth=0.5)
        ax_rsi.axhline(30, color="#26a69a", linestyle=":", linewidth=0.5)
        ax_rsi.set_ylabel("RSI", fontsize=9)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.3)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    ax_vol.bar(df.index, df["volume"], width=0.0025, color=colors, alpha=0.7)
    ax_vol.set_ylabel("Volume", fontsize=9)
    ax_vol.grid(True, alpha=0.3)

    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()

    filename = f"{symbol.replace('-', '')}_{trade_id}_entry.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_exit_chart(
    df: pd.DataFrame,
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    take_profit: float,
    symbol: str,
    trade_id: int,
    output_dir: str = "data/charts",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)
    ax_price, ax_rsi, ax_vol = axes

    _plot_candlesticks(ax_price, df)
    ax_price.axhline(entry_price, color="#ffd700", linestyle="--", linewidth=1, label=f"Entry €{entry_price:,.2f}")
    ax_price.axhline(exit_price, color="#00e676", linestyle="--", linewidth=1, label=f"Exit €{exit_price:,.2f}")
    ax_price.axhline(stop_loss, color="#ef5350", linestyle=":", linewidth=0.8, label=f"SL €{stop_loss:,.2f}")
    ax_price.axhline(take_profit, color="#26a69a", linestyle=":", linewidth=0.8, label=f"TP €{take_profit:,.2f}")

    if len(df) >= 20:
        ax_price.plot(df.index, ema(df["close"], 20), color="#42a5f5", linewidth=0.8, label="EMA20")
    if len(df) >= 50:
        ax_price.plot(df.index, ema(df["close"], 50), color="#ab47bc", linewidth=0.8, label="EMA50")

    ax_price.set_title(f"{symbol} Exit — Trade #{trade_id}", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)

    if len(df) >= 14:
        r = rsi(df["close"])
        ax_rsi.plot(df.index, r, color="#ffa726", linewidth=0.8)
        ax_rsi.axhline(70, color="#ef5350", linestyle=":", linewidth=0.5)
        ax_rsi.axhline(30, color="#26a69a", linestyle=":", linewidth=0.5)
        ax_rsi.set_ylabel("RSI", fontsize=9)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.3)

    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    ax_vol.bar(df.index, df["volume"], width=0.0025, color=colors, alpha=0.7)
    ax_vol.set_ylabel("Volume", fontsize=9)
    ax_vol.grid(True, alpha=0.3)

    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()

    filename = f"{symbol.replace('-', '')}_{trade_id}_exit.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
