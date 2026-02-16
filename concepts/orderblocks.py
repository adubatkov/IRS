"""Order Block (OB) detection and lifecycle management.

An Order Block is the last opposing candle before a significant displacement
(BOS/CHoCH). It represents an institutional order zone.

Bullish OB: last bearish candle before bullish BOS/displacement
Bearish OB: last bullish candle before bearish BOS/displacement
"""

from enum import Enum

import pandas as pd


class OBStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TESTED = "TESTED"
    MITIGATED = "MITIGATED"
    BROKEN = "BROKEN"


def detect_orderblocks(
    df: pd.DataFrame,
    structure_events: pd.DataFrame,
    max_age_candles: int = 500,
    close_mitigation: bool = True,
) -> pd.DataFrame:
    """Detect Order Blocks based on structure breaks (BOS/CHoCH).

    For each structure event, find the last opposing candle before the break.

    Args:
        df: OHLC DataFrame.
        structure_events: DataFrame from detect_bos_choch with 'direction', 'swing_index'.
        max_age_candles: OBs older than this are expired.
        close_mitigation: If True, OB mitigation requires candle close (not just wick).

    Returns:
        DataFrame with one row per OB:
        - direction: +1 bullish OB (demand zone), -1 bearish OB (supply zone)
        - top: upper boundary (high of OB candle)
        - bottom: lower boundary (low of OB candle)
        - ob_index: index of the OB candle
        - trigger_index: index of the structure break that created it
        - status: OBStatus.ACTIVE
    """
    if len(structure_events) == 0:
        return pd.DataFrame(
            columns=["direction", "top", "bottom", "ob_index", "trigger_index", "status"]
        )

    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    obs = []

    for _, event in structure_events.iterrows():
        direction = event["direction"]
        swing_idx = event["swing_index"]
        broken_idx = event["broken_index"]

        # Search backward from the swing for the last opposing candle
        search_start = swing_idx
        if search_start < 0 or search_start >= len(df):
            continue

        if direction == 1:
            # Bullish break: find last bearish candle before it
            for j in range(search_start, max(search_start - 50, -1), -1):
                if j < 0 or j >= len(df):
                    continue
                if closes[j] < opens[j]:  # Bearish candle
                    obs.append({
                        "direction": 1,
                        "top": highs[j],
                        "bottom": lows[j],
                        "ob_index": df.index[j],
                        "trigger_index": broken_idx,
                        "status": OBStatus.ACTIVE,
                    })
                    break
        else:
            # Bearish break: find last bullish candle before it
            for j in range(search_start, max(search_start - 50, -1), -1):
                if j < 0 or j >= len(df):
                    continue
                if closes[j] > opens[j]:  # Bullish candle
                    obs.append({
                        "direction": -1,
                        "top": highs[j],
                        "bottom": lows[j],
                        "ob_index": df.index[j],
                        "trigger_index": broken_idx,
                        "status": OBStatus.ACTIVE,
                    })
                    break

    if not obs:
        return pd.DataFrame(
            columns=["direction", "top", "bottom", "ob_index", "trigger_index", "status"]
        )

    return pd.DataFrame(obs)


def update_ob_status(
    obs: pd.DataFrame,
    candle_high: float,
    candle_low: float,
    candle_close: float,
    close_mitigation: bool = True,
) -> pd.DataFrame:
    """Update Order Block statuses based on new price action."""
    result = obs.copy()

    for idx in result.index:
        status = result.loc[idx, "status"]
        if status in (OBStatus.MITIGATED, OBStatus.BROKEN):
            continue

        direction = result.loc[idx, "direction"]
        top = result.loc[idx, "top"]
        bottom = result.loc[idx, "bottom"]

        if direction == 1:  # Bullish OB (demand zone below price)
            if close_mitigation:
                if candle_close < bottom:
                    result.loc[idx, "status"] = OBStatus.BROKEN
                elif candle_low <= top:
                    result.loc[idx, "status"] = OBStatus.TESTED
            else:
                if candle_low < bottom:
                    result.loc[idx, "status"] = OBStatus.BROKEN
                elif candle_low <= top:
                    result.loc[idx, "status"] = OBStatus.TESTED
        else:  # Bearish OB (supply zone above price)
            if close_mitigation:
                if candle_close > top:
                    result.loc[idx, "status"] = OBStatus.BROKEN
                elif candle_high >= bottom:
                    result.loc[idx, "status"] = OBStatus.TESTED
            else:
                if candle_high > top:
                    result.loc[idx, "status"] = OBStatus.BROKEN
                elif candle_high >= bottom:
                    result.loc[idx, "status"] = OBStatus.TESTED

    return result
