"""Market Structure detection: BOS, cBOS, and CISD.

BOS (Break of Structure): Price breaks a swing level against the current trend (reversal signal).
cBOS (Continuation BOS): Price breaks a swing level in the same direction as trend (continuation).
CISD (Change in State of Delivery): Early momentum shift via candle open breaks.
"""

from enum import Enum

import pandas as pd

from concepts.fractals import detect_swings, get_swing_points


class StructureType(str, Enum):
    BOS = "BOS"
    CBOS = "CBOS"


class Trend(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    UNDEFINED = "UNDEFINED"


def detect_structure(
    df: pd.DataFrame,
    swing_length: int = 5,
    close_break: bool = True,
) -> pd.DataFrame:
    """Detect BOS and cBOS events from OHLC data.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.
        swing_length: Swing detection parameter.
        close_break: If True, use candle close for break detection (stricter).
                     If False, use high/low (wick) for break detection.

    Returns:
        DataFrame with one row per structure event:
        - type: BOS (against trend) or CBOS (with trend)
        - direction: +1 bullish, -1 bearish
        - broken_level: the swing level that was broken
        - broken_index: index of the candle that broke it
        - swing_index: original index of the swing that was broken
    """
    swings = detect_swings(df, swing_length=swing_length)
    points = get_swing_points(df, swings)

    if len(points) < 2:
        return pd.DataFrame(
            columns=["type", "direction", "broken_level", "broken_index", "swing_index"]
        )

    events = []
    trend = Trend.UNDEFINED

    # Track the most recent unbroken swing high and swing low
    last_swing_high = None  # (orig_index, level)
    last_swing_low = None   # (orig_index, level)

    # Collect all swing points into a lookup
    swing_by_index = {}
    for _, pt in points.iterrows():
        swing_by_index[pt["orig_index"]] = pt

    # Process bar by bar
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for i in range(len(df)):
        orig_idx = df.index[i]

        # Register any swing that was confirmed at this index
        # (swing at position p is confirmed at p + swing_length)
        confirmed_idx = orig_idx - swing_length
        if confirmed_idx in swing_by_index:
            pt = swing_by_index[confirmed_idx]
            if pt["direction"] == 1:
                last_swing_high = (confirmed_idx, pt["level"])
            else:
                last_swing_low = (confirmed_idx, pt["level"])

        # Check for breaks
        if close_break:
            break_up = closes[i]
            break_down = closes[i]
        else:
            break_up = highs[i]
            break_down = lows[i]

        # Check break above swing high
        if last_swing_high is not None:
            sh_idx, sh_level = last_swing_high
            if break_up > sh_level:
                if trend == Trend.BULLISH or trend == Trend.UNDEFINED:
                    event_type = StructureType.CBOS
                else:
                    event_type = StructureType.BOS
                events.append({
                    "type": event_type,
                    "direction": 1,
                    "broken_level": sh_level,
                    "broken_index": orig_idx,
                    "swing_index": sh_idx,
                })
                trend = Trend.BULLISH
                last_swing_high = None  # Consumed

        # Check break below swing low
        if last_swing_low is not None:
            sl_idx, sl_level = last_swing_low
            if break_down < sl_level:
                if trend == Trend.BEARISH or trend == Trend.UNDEFINED:
                    event_type = StructureType.CBOS
                else:
                    event_type = StructureType.BOS
                events.append({
                    "type": event_type,
                    "direction": -1,
                    "broken_level": sl_level,
                    "broken_index": orig_idx,
                    "swing_index": sl_idx,
                })
                trend = Trend.BEARISH
                last_swing_low = None  # Consumed

    if not events:
        return pd.DataFrame(
            columns=["type", "direction", "broken_level", "broken_index", "swing_index"]
        )

    return pd.DataFrame(events)


def detect_cisd(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Detect Change in State of Delivery (CISD) events.

    CISD detects when price closes beyond the opening price of a previous
    directional sequence, signaling an early momentum shift.

    Args:
        df: DataFrame with 'open', 'close' columns.

    Returns:
        DataFrame with one row per CISD event:
        - direction: +1 bullish (closes above bearish sequence start), -1 bearish
        - level: the opening price that was broken
        - trigger_index: index of the candle that confirmed CISD
        - origin_index: index of the first candle of the opposing sequence
    """
    opens = df["open"].values
    closes = df["close"].values
    n = len(df)

    events = []

    # Track the start of current directional sequence
    seq_start_idx = 0
    seq_start_open = opens[0]
    seq_direction = 0  # 0=undefined, 1=bullish candles, -1=bearish candles

    for i in range(1, n):
        candle_dir = 1 if closes[i] > opens[i] else (-1 if closes[i] < opens[i] else 0)

        if candle_dir == 0:
            continue

        if seq_direction == 0:
            seq_direction = candle_dir
            seq_start_idx = i
            seq_start_open = opens[i]
            continue

        if candle_dir == seq_direction:
            # Same direction, sequence continues
            continue

        # Direction changed - check for CISD
        orig_idx = df.index[seq_start_idx]

        if seq_direction == -1 and closes[i] > seq_start_open:
            # Was bearish sequence, now bullish candle closes above sequence start open
            events.append({
                "direction": 1,
                "level": seq_start_open,
                "trigger_index": df.index[i],
                "origin_index": orig_idx,
            })
        elif seq_direction == 1 and closes[i] < seq_start_open:
            # Was bullish sequence, now bearish candle closes below sequence start open
            events.append({
                "direction": -1,
                "level": seq_start_open,
                "trigger_index": df.index[i],
                "origin_index": orig_idx,
            })

        # Start new sequence
        seq_direction = candle_dir
        seq_start_idx = i
        seq_start_open = opens[i]

    if not events:
        return pd.DataFrame(columns=["direction", "level", "trigger_index", "origin_index"])

    return pd.DataFrame(events)
