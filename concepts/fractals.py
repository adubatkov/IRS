"""Swing High/Low (Fractal) detection.

Detects swing highs and lows using a configurable window size (swing_length).
A swing high at index i means high[i] is the maximum in the window
[i - swing_length, i + swing_length]. Similarly for swing lows.

Confirmation is delayed by swing_length bars (no look-ahead bias in backtest).
"""

from enum import Enum

import numpy as np
import pandas as pd


class SwingStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SWEPT = "SWEPT"
    BROKEN = "BROKEN"


def detect_swings(
    df: pd.DataFrame,
    swing_length: int = 5,
) -> pd.DataFrame:
    """Detect swing highs and lows in OHLC data.

    Uses vectorized rolling window comparison. A swing high at index i is
    confirmed when high[i] is the max of highs in [i-swing_length, i+swing_length].
    Similarly for swing lows.

    Args:
        df: DataFrame with 'high' and 'low' columns.
        swing_length: Number of candles on each side to compare.

    Returns:
        DataFrame with columns:
        - swing_high: bool, True at swing high points
        - swing_low: bool, True at swing low points
        - swing_high_price: float, price at swing high (NaN otherwise)
        - swing_low_price: float, price at swing low (NaN otherwise)
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    window = 2 * swing_length + 1

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    # Vectorized: use rolling max/min
    high_series = pd.Series(highs)
    low_series = pd.Series(lows)

    # Rolling max of highs and rolling min of lows over the full window
    rolling_max = high_series.rolling(window=window, center=True, min_periods=window).max()
    rolling_min = low_series.rolling(window=window, center=True, min_periods=window).min()

    # Swing high: the center candle's high equals the rolling max
    # AND it's strictly higher than its immediate neighbors
    swing_high_mask = (
        (high_series == rolling_max)
        & (high_series > high_series.shift(1))
        & (high_series > high_series.shift(-1))
    )

    # Swing low: the center candle's low equals the rolling min
    # AND it's strictly lower than its immediate neighbors
    swing_low_mask = (
        (low_series == rolling_min)
        & (low_series < low_series.shift(1))
        & (low_series < low_series.shift(-1))
    )

    swing_high = swing_high_mask.fillna(False).values
    swing_low = swing_low_mask.fillna(False).values

    # Resolve conflicts: a candle cannot be both swing high and swing low.
    # Keep the one with greater relative extremity.
    overlap = swing_high & swing_low
    if overlap.any():
        high_range = highs - rolling_max.values
        low_range = rolling_min.values - lows
        # Where high is more extreme, keep swing_high; otherwise keep swing_low
        prefer_high = np.abs(high_range) >= np.abs(low_range)
        swing_low[overlap & prefer_high] = False
        swing_high[overlap & ~prefer_high] = False

    result = pd.DataFrame({
        "swing_high": swing_high,
        "swing_low": swing_low,
        "swing_high_price": np.where(swing_high, highs, np.nan),
        "swing_low_price": np.where(swing_low, lows, np.nan),
    }, index=df.index)

    return result


def get_swing_points(
    df: pd.DataFrame,
    swings: pd.DataFrame,
) -> pd.DataFrame:
    """Extract a list of swing points with their metadata.

    Returns a DataFrame with one row per swing point, containing:
    - index: original index in the OHLC DataFrame
    - time: timestamp
    - direction: +1 for swing high, -1 for swing low
    - level: price level
    - status: SwingStatus.ACTIVE
    """
    points = []

    sh_idx = swings.index[swings["swing_high"]]
    sl_idx = swings.index[swings["swing_low"]]

    for idx in sh_idx:
        row = {"orig_index": idx, "direction": 1, "level": swings.loc[idx, "swing_high_price"],
               "status": SwingStatus.ACTIVE}
        if "time" in df.columns:
            row["time"] = df.loc[idx, "time"]
        points.append(row)

    for idx in sl_idx:
        row = {"orig_index": idx, "direction": -1, "level": swings.loc[idx, "swing_low_price"],
               "status": SwingStatus.ACTIVE}
        if "time" in df.columns:
            row["time"] = df.loc[idx, "time"]
        points.append(row)

    if not points:
        return pd.DataFrame(columns=["orig_index", "time", "direction", "level", "status"])

    result = pd.DataFrame(points).sort_values("orig_index").reset_index(drop=True)
    return result


def update_swing_status(
    swing_points: pd.DataFrame,
    current_high: float,
    current_low: float,
) -> pd.DataFrame:
    """Update swing point statuses based on current price action.

    - SWEPT: price wick went past swing level but didn't close past it
    - BROKEN: price closed past swing level (handled by structure module)

    For simplicity, this function marks swings as SWEPT when:
    - Swing high: current_high > swing_high_level
    - Swing low: current_low < swing_low_level
    """
    sp = swing_points.copy()

    active_mask = sp["status"] == SwingStatus.ACTIVE

    # Swing highs swept when price goes above
    sh_swept = active_mask & (sp["direction"] == 1) & (current_high > sp["level"])
    sp.loc[sh_swept, "status"] = SwingStatus.SWEPT

    # Swing lows swept when price goes below
    sl_swept = active_mask & (sp["direction"] == -1) & (current_low < sp["level"])
    sp.loc[sl_swept, "status"] = SwingStatus.SWEPT

    return sp
