"""Liquidity detection: Equal Highs/Lows, Session Levels, and Sweep detection.

Liquidity exists where stop-losses cluster - at equal highs, equal lows,
and session extreme levels. A liquidity sweep occurs when price wicks past
a level but closes back inside.
"""

from enum import Enum

import numpy as np
import pandas as pd

from concepts.fractals import detect_swings


class LiquidityStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SWEPT = "SWEPT"


def detect_equal_levels(
    df: pd.DataFrame,
    swing_length: int = 5,
    range_percent: float = 0.001,
    min_touches: int = 2,
) -> pd.DataFrame:
    """Detect equal highs and equal lows (liquidity pools).

    Multiple swing highs/lows at approximately the same level indicate
    resting liquidity.

    Args:
        df: OHLC DataFrame.
        swing_length: For swing detection.
        range_percent: How close levels must be to count as "equal" (as % of price).
        min_touches: Minimum number of touches to form liquidity.

    Returns:
        DataFrame with one row per liquidity level:
        - direction: +1 (buy-side liquidity above), -1 (sell-side below)
        - level: price level
        - count: number of touches
        - indices: list of swing indices that touch this level
        - status: LiquidityStatus.ACTIVE
    """
    swings = detect_swings(df, swing_length=swing_length)

    levels = []

    # Process swing highs
    sh_mask = swings["swing_high"]
    sh_prices = swings.loc[sh_mask, "swing_high_price"].values
    sh_indices = swings.index[sh_mask].tolist()

    _cluster_levels(sh_prices, sh_indices, 1, range_percent, min_touches, levels)

    # Process swing lows
    sl_mask = swings["swing_low"]
    sl_prices = swings.loc[sl_mask, "swing_low_price"].values
    sl_indices = swings.index[sl_mask].tolist()

    _cluster_levels(sl_prices, sl_indices, -1, range_percent, min_touches, levels)

    if not levels:
        return pd.DataFrame(columns=["direction", "level", "count", "indices", "status"])

    return pd.DataFrame(levels)


def _cluster_levels(
    prices: np.ndarray,
    indices: list[int],
    direction: int,
    range_percent: float,
    min_touches: int,
    output: list[dict],
) -> None:
    """Cluster nearby price levels into liquidity zones."""
    if len(prices) < min_touches:
        return

    used = set()
    for i in range(len(prices)):
        if i in used:
            continue
        level = prices[i]
        threshold = level * range_percent
        cluster_prices = [level]
        cluster_indices = [indices[i]]
        used.add(i)

        for j in range(i + 1, len(prices)):
            if j in used:
                continue
            if abs(prices[j] - level) <= threshold:
                cluster_prices.append(prices[j])
                cluster_indices.append(indices[j])
                used.add(j)

        if len(cluster_prices) >= min_touches:
            output.append({
                "direction": direction,
                "level": float(np.mean(cluster_prices)),
                "count": len(cluster_prices),
                "indices": cluster_indices,
                "status": LiquidityStatus.ACTIVE,
            })


def detect_session_levels(
    df: pd.DataFrame,
    level_type: str = "daily",
) -> pd.DataFrame:
    """Detect previous session high/low levels.

    Args:
        df: OHLC DataFrame with 'time' column.
        level_type: "daily", "weekly", or "monthly".

    Returns:
        DataFrame with high/low levels per session period.
    """
    if "time" not in df.columns:
        return pd.DataFrame(columns=["period_start", "high", "low"])

    data = df.set_index("time")

    freq_map = {"daily": "D", "weekly": "W", "monthly": "ME"}
    freq = freq_map.get(level_type, "D")

    grouped = data.resample(freq).agg({"high": "max", "low": "min"}).dropna()
    grouped = grouped.reset_index()
    grouped.columns = ["period_start", "high", "low"]

    return grouped


def detect_sweep(
    candle_high: float,
    candle_low: float,
    candle_close: float,
    level: float,
    direction: int,
) -> bool:
    """Check if current candle sweeps a liquidity level.

    A sweep = wick past the level, close doesn't break it.

    Args:
        direction: +1 for buy-side (above), -1 for sell-side (below).
    """
    if direction == 1:
        # Buy-side: wick above level, close below
        return candle_high > level and candle_close <= level
    else:
        # Sell-side: wick below level, close above
        return candle_low < level and candle_close >= level
