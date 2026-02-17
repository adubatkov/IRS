"""HTF bias determination from structure events and price action."""

import pandas as pd
from strategy.types import Bias
from concepts.structure import StructureType


def determine_bias(
    candles: pd.DataFrame,
    structure_events: pd.DataFrame,
    lookback: int = 10,
) -> Bias:
    """Determine directional bias from HTF candles and structure.

    Algorithm:
    1. Look at the last ``lookback`` structure events.
    2. Count bullish vs bearish events.
    3. Weight BOS (reversal) events more than cBOS (continuation).
    4. If predominantly bullish -> BULLISH.
    5. If predominantly bearish -> BEARISH.
    6. If mixed or no events -> UNDEFINED.

    Scoring:
    - BOS events count as 2 (they indicate trend reversal).
    - cBOS events count as 1 (they confirm existing trend).
    - Sum bullish score vs bearish score.
    - If ratio > 0.6 in one direction -> that direction.
    - Otherwise -> UNDEFINED.

    Args:
        candles: HTF OHLC DataFrame (for reference, not used in v1).
        structure_events: BOS/cBOS events from ``detect_structure()``.
        lookback: Number of recent events to consider.

    Returns:
        Bias enum value.
    """
    if structure_events.empty:
        return Bias.UNDEFINED

    recent = structure_events.tail(lookback)

    bullish_score = 0.0
    bearish_score = 0.0

    for _, event in recent.iterrows():
        weight = 2.0 if event["type"] == StructureType.BOS else 1.0
        if event["direction"] == 1:
            bullish_score += weight
        elif event["direction"] == -1:
            bearish_score += weight

    total = bullish_score + bearish_score
    if total == 0:
        return Bias.UNDEFINED

    bullish_ratio = bullish_score / total
    bearish_ratio = bearish_score / total

    if bullish_ratio > 0.6:
        return Bias.BULLISH
    if bearish_ratio > 0.6:
        return Bias.BEARISH
    return Bias.UNDEFINED


def determine_bias_at(
    candles: pd.DataFrame,
    structure_events: pd.DataFrame,
    timestamp: pd.Timestamp,
    lookback: int = 10,
) -> Bias:
    """Time-filtered version of determine_bias.

    Only uses structure events with ``broken_index`` whose corresponding
    candle time <= *timestamp*. Uses the *candles* DataFrame to map indices
    to times.

    Args:
        candles: HTF OHLC DataFrame with a ``time`` column.
        structure_events: BOS/cBOS events from ``detect_structure()``.
        timestamp: Only consider events at or before this time.
        lookback: Number of recent (filtered) events to consider.

    Returns:
        Bias enum value.
    """
    if structure_events.empty:
        return Bias.UNDEFINED

    # Build a mapping from candle index to its time.
    if "time" in candles.columns:
        time_series = candles["time"]
    else:
        # Fallback: if the DataFrame index itself holds timestamps
        time_series = pd.Series(candles.index, index=candles.index)

    # Filter structure events whose broken_index maps to a time <= timestamp
    mask = []
    for _, event in structure_events.iterrows():
        broken_idx = event["broken_index"]
        if broken_idx in time_series.index:
            event_time = time_series.loc[broken_idx]
            mask.append(event_time <= timestamp)
        else:
            # If the index is not in candles, skip this event
            mask.append(False)

    filtered = structure_events[mask]
    return determine_bias(candles, filtered, lookback=lookback)


def get_trend_from_structure(
    structure_events: pd.DataFrame,
    n_recent: int = 3,
) -> Bias:
    """Derive trend direction from the most recent N structure events.

    Simpler than ``determine_bias``: just looks at direction of last N
    events. If last N events are all or mostly the same direction, return
    that direction.

    The majority rule: if more than half of the last *n_recent* events
    share a direction, that direction wins. Otherwise UNDEFINED.

    Args:
        structure_events: BOS/cBOS events from ``detect_structure()``.
        n_recent: Number of most-recent events to examine.

    Returns:
        Bias enum value.
    """
    if structure_events.empty:
        return Bias.UNDEFINED

    recent = structure_events.tail(n_recent)
    directions = recent["direction"].values

    bullish_count = int((directions == 1).sum())
    bearish_count = int((directions == -1).sum())
    total = len(directions)

    if total == 0:
        return Bias.UNDEFINED

    if bullish_count > total / 2:
        return Bias.BULLISH
    if bearish_count > total / 2:
        return Bias.BEARISH
    return Bias.UNDEFINED
