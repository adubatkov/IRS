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
        - start_index: index of the first candle in the pattern
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
                    "start_index": df.index[i - 2],
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
                    "start_index": df.index[i - 2],
                    "creation_index": df.index[i],
                    "status": FVGStatus.FRESH,
                })

    if not fvgs:
        return pd.DataFrame(
            columns=["direction", "top", "bottom", "midpoint",
                     "start_index", "creation_index", "status"]
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
            # Merge: extend the zone, keep earliest start, latest creation
            current["top"] = max(current["top"], row["top"])
            current["bottom"] = min(current["bottom"], row["bottom"])
            current["midpoint"] = (current["top"] + current["bottom"]) / 2
            current["start_index"] = min(current["start_index"], row["start_index"])
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


def track_fvg_lifecycle(
    df: pd.DataFrame,
    fvgs: pd.DataFrame,
    mitigation_mode: str = "close",
    max_age_bars: int = 192,
) -> list[dict]:
    """Track the full lifecycle of each FVG through subsequent price action.

    For each FVG, iterates bar-by-bar from creation forward, tracking:
    - How deeply price penetrates the zone (fill_level)
    - Status transitions (FRESH -> TESTED -> PARTIALLY_FILLED -> FULLY_FILLED/INVERTED)
    - When the FVG ends (mitigation, inversion, or max age expiry)

    Args:
        df: OHLC DataFrame (same one used to detect FVGs).
        fvgs: DataFrame from detect_fvg().
        mitigation_mode: "close" (candle close determines inversion) or "wick".
        max_age_bars: Maximum bars before FVG expires.

    Returns:
        List of dicts, one per FVG, with:
        - fvg_idx: row index in the fvgs DataFrame
        - start_index: first candle of the FVG pattern
        - end_index: bar where FVG ended (mitigation/inversion/expiry)
        - status: final status at end
        - fill_level: deepest price penetration into the zone
        - inversion_index: bar index where IFVG was created (or None)
        - direction, top, bottom, midpoint: copied from original FVG
    """
    if len(fvgs) == 0 or len(df) == 0:
        return []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    results = []

    for fvg_row_idx in range(len(fvgs)):
        fvg = fvgs.iloc[fvg_row_idx]
        direction = fvg["direction"]
        top = fvg["top"]
        bottom = fvg["bottom"]
        midpoint = fvg["midpoint"]
        creation_idx = fvg["creation_index"]
        start_idx = fvg.get("start_index", creation_idx)

        # Find positional start for iteration (bar after creation)
        try:
            creation_pos = df.index.get_loc(creation_idx)
        except KeyError:
            continue
        if not isinstance(creation_pos, int):
            continue

        status = FVGStatus.FRESH
        fill_level = None  # Deepest penetration
        end_pos = min(creation_pos + max_age_bars, len(df) - 1)
        end_index = df.index[end_pos]
        inversion_index = None

        for pos in range(creation_pos + 1, min(creation_pos + max_age_bars + 1, len(df))):
            c_high = highs[pos]
            c_low = lows[pos]
            c_close = closes[pos]

            if direction == 1:  # Bullish FVG — support zone below
                if c_low <= top:  # Price entered the zone
                    if fill_level is None or c_low < fill_level:
                        fill_level = c_low

                    if mitigation_mode == "close" and c_close < bottom:
                        status = FVGStatus.INVERTED
                        end_index = df.index[pos]
                        inversion_index = df.index[pos]
                        break
                    elif mitigation_mode == "wick" and c_low < bottom:
                        status = FVGStatus.FULLY_FILLED
                        end_index = df.index[pos]
                        break

                    if c_low <= midpoint:
                        if status in (FVGStatus.FRESH, FVGStatus.TESTED):
                            status = FVGStatus.PARTIALLY_FILLED
                    elif status == FVGStatus.FRESH:
                        status = FVGStatus.TESTED

            else:  # Bearish FVG — resistance zone above
                if c_high >= bottom:  # Price entered the zone
                    if fill_level is None or c_high > fill_level:
                        fill_level = c_high

                    if mitigation_mode == "close" and c_close > top:
                        status = FVGStatus.INVERTED
                        end_index = df.index[pos]
                        inversion_index = df.index[pos]
                        break
                    elif mitigation_mode == "wick" and c_high > top:
                        status = FVGStatus.FULLY_FILLED
                        end_index = df.index[pos]
                        break

                    if c_high >= midpoint:
                        if status in (FVGStatus.FRESH, FVGStatus.TESTED):
                            status = FVGStatus.PARTIALLY_FILLED
                    elif status == FVGStatus.FRESH:
                        status = FVGStatus.TESTED

        results.append({
            "fvg_idx": fvg_row_idx,
            "direction": direction,
            "top": top,
            "bottom": bottom,
            "midpoint": midpoint,
            "start_index": start_idx,
            "creation_index": creation_idx,
            "end_index": end_index,
            "status": status,
            "fill_level": fill_level,
            "inversion_index": inversion_index,
        })

    return results
