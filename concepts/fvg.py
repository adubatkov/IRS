"""Fair Value Gap (FVG) detection and lifecycle management.

FVG is a 3-candle pattern where price left an "untouched" zone.
Bullish FVG: low[i] > high[i-2] (gap up)
Bearish FVG: high[i] < low[i-2] (gap down)

Also handles Inverted FVGs (IFVGs) - FVGs that price closed through,
now acting as opposite support/resistance.
"""

from enum import Enum

import pandas as pd


class FVGStatus(str, Enum):
    FRESH = "FRESH"
    TESTED = "TESTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FULLY_FILLED = "FULLY_FILLED"
    MITIGATED = "MITIGATED"
    INVERTED = "INVERTED"


def detect_fvg(
    df: pd.DataFrame,
    min_gap_pct: float = 0.0005,
    join_consecutive: bool = True,
) -> pd.DataFrame:
    """Detect Fair Value Gaps in OHLC data.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        min_gap_pct: Minimum gap size as percentage of price.
        join_consecutive: If True, merge adjacent FVGs of same direction.

    Returns:
        DataFrame with one row per FVG:
        - direction: +1 bullish, -1 bearish
        - top: upper boundary of gap
        - bottom: lower boundary of gap
        - midpoint: (top + bottom) / 2
        - creation_index: index of the third candle
        - status: FVGStatus.FRESH
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    fvgs = []

    for i in range(2, n):
        # Bullish FVG: low of candle 3 > high of candle 1
        if lows[i] > highs[i - 2]:
            gap_top = lows[i]
            gap_bottom = highs[i - 2]
            gap_size = gap_top - gap_bottom
            if gap_size > min_gap_pct * closes[i]:
                fvgs.append({
                    "direction": 1,
                    "top": gap_top,
                    "bottom": gap_bottom,
                    "midpoint": (gap_top + gap_bottom) / 2,
                    "creation_index": df.index[i],
                    "status": FVGStatus.FRESH,
                })

        # Bearish FVG: high of candle 3 < low of candle 1
        if highs[i] < lows[i - 2]:
            gap_top = lows[i - 2]
            gap_bottom = highs[i]
            gap_size = gap_top - gap_bottom
            if gap_size > min_gap_pct * closes[i]:
                fvgs.append({
                    "direction": -1,
                    "top": gap_top,
                    "bottom": gap_bottom,
                    "midpoint": (gap_top + gap_bottom) / 2,
                    "creation_index": df.index[i],
                    "status": FVGStatus.FRESH,
                })

    if not fvgs:
        return pd.DataFrame(
            columns=["direction", "top", "bottom", "midpoint", "creation_index", "status"]
        )

    result = pd.DataFrame(fvgs)

    if join_consecutive and len(result) > 1:
        result = _join_consecutive_fvgs(result)

    return result


def _join_consecutive_fvgs(fvgs: pd.DataFrame) -> pd.DataFrame:
    """Merge adjacent FVGs of the same direction into one."""
    if len(fvgs) <= 1:
        return fvgs

    merged = []
    current = fvgs.iloc[0].to_dict()

    for i in range(1, len(fvgs)):
        row = fvgs.iloc[i]
        # Same direction and overlapping or adjacent
        if (row["direction"] == current["direction"]
                and _zones_overlap(current["bottom"], current["top"],
                                   row["bottom"], row["top"])):
            # Merge: extend the zone
            current["top"] = max(current["top"], row["top"])
            current["bottom"] = min(current["bottom"], row["bottom"])
            current["midpoint"] = (current["top"] + current["bottom"]) / 2
            current["creation_index"] = row["creation_index"]  # Use latest
        else:
            merged.append(current)
            current = row.to_dict()

    merged.append(current)
    return pd.DataFrame(merged)


def _zones_overlap(bot1: float, top1: float, bot2: float, top2: float) -> bool:
    """Check if two price zones overlap or are adjacent."""
    return bot1 <= top2 and bot2 <= top1


def update_fvg_status(
    fvgs: pd.DataFrame,
    candle_high: float,
    candle_low: float,
    candle_close: float,
    mitigation_mode: str = "close",
) -> pd.DataFrame:
    """Update FVG statuses based on new price action.

    Args:
        fvgs: DataFrame of FVGs with 'direction', 'top', 'bottom', 'status'.
        candle_high: High of the current candle.
        candle_low: Low of the current candle.
        candle_close: Close of the current candle.
        mitigation_mode: "wick", "close", "ce", or "full".
    """
    result = fvgs.copy()

    for idx in result.index:
        status = result.loc[idx, "status"]
        if status in (FVGStatus.MITIGATED, FVGStatus.INVERTED):
            continue

        direction = result.loc[idx, "direction"]
        top = result.loc[idx, "top"]
        bottom = result.loc[idx, "bottom"]
        midpoint = result.loc[idx, "midpoint"]

        if direction == 1:  # Bullish FVG (zone below price, acts as support)
            # Check if price came into the zone from above
            touched = candle_low <= top

            if not touched:
                continue

            if mitigation_mode == "wick":
                if candle_low <= bottom:
                    result.loc[idx, "status"] = FVGStatus.FULLY_FILLED
                elif candle_low <= midpoint:
                    result.loc[idx, "status"] = FVGStatus.PARTIALLY_FILLED
                else:
                    result.loc[idx, "status"] = FVGStatus.TESTED

            elif mitigation_mode == "close":
                if candle_close < bottom:
                    result.loc[idx, "status"] = FVGStatus.INVERTED
                elif candle_close <= midpoint:
                    result.loc[idx, "status"] = FVGStatus.PARTIALLY_FILLED
                elif touched:
                    result.loc[idx, "status"] = FVGStatus.TESTED

            elif mitigation_mode == "ce":
                if candle_low <= midpoint:
                    result.loc[idx, "status"] = FVGStatus.MITIGATED
                elif touched:
                    result.loc[idx, "status"] = FVGStatus.TESTED

            elif mitigation_mode == "full":
                if candle_close < bottom:
                    result.loc[idx, "status"] = FVGStatus.INVERTED
                elif candle_low <= bottom:
                    result.loc[idx, "status"] = FVGStatus.FULLY_FILLED
                elif touched:
                    result.loc[idx, "status"] = FVGStatus.TESTED

        else:  # Bearish FVG (zone above price, acts as resistance)
            touched = candle_high >= bottom

            if not touched:
                continue

            if mitigation_mode == "wick":
                if candle_high >= top:
                    result.loc[idx, "status"] = FVGStatus.FULLY_FILLED
                elif candle_high >= midpoint:
                    result.loc[idx, "status"] = FVGStatus.PARTIALLY_FILLED
                else:
                    result.loc[idx, "status"] = FVGStatus.TESTED

            elif mitigation_mode == "close":
                if candle_close > top:
                    result.loc[idx, "status"] = FVGStatus.INVERTED
                elif candle_close >= midpoint:
                    result.loc[idx, "status"] = FVGStatus.PARTIALLY_FILLED
                elif touched:
                    result.loc[idx, "status"] = FVGStatus.TESTED

            elif mitigation_mode == "ce":
                if candle_high >= midpoint:
                    result.loc[idx, "status"] = FVGStatus.MITIGATED
                elif touched:
                    result.loc[idx, "status"] = FVGStatus.TESTED

            elif mitigation_mode == "full":
                if candle_close > top:
                    result.loc[idx, "status"] = FVGStatus.INVERTED
                elif candle_high >= top:
                    result.loc[idx, "status"] = FVGStatus.FULLY_FILLED
                elif touched:
                    result.loc[idx, "status"] = FVGStatus.TESTED

    return result
