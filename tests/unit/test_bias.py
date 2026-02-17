"""Tests for HTF bias determination from structure events."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.structure import StructureType
from context.bias import determine_bias, determine_bias_at, get_trend_from_structure
from strategy.types import Bias


def _make_structure_events(entries: list[tuple[str, int]]) -> pd.DataFrame:
    """Build a synthetic structure-events DataFrame.

    Args:
        entries: List of (type_str, direction) tuples.
            type_str is "BOS" or "CBOS", direction is +1 or -1.

    Returns:
        DataFrame matching ``detect_structure()`` output format.
    """
    rows = []
    for i, (stype, direction) in enumerate(entries):
        rows.append({
            "type": StructureType(stype),
            "direction": direction,
            "broken_level": 100.0 + i,
            "broken_index": i,
            "swing_index": max(0, i - 1),
        })
    return pd.DataFrame(rows)


def _make_candles(n: int) -> pd.DataFrame:
    """Create a minimal candle DataFrame with a ``time`` column.

    Indices are 0..n-1, times spaced 1 hour apart.
    """
    times = pd.date_range("2024-01-01 09:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
    })


class TestDetermineBias:
    def test_bullish_bias_from_bullish_structure(self):
        """Mostly bullish events should yield BULLISH bias."""
        events = _make_structure_events([
            ("BOS", 1),
            ("CBOS", 1),
            ("CBOS", 1),
            ("CBOS", 1),
            ("CBOS", -1),
        ])
        candles = _make_candles(10)
        bias = determine_bias(candles, events)
        assert bias == Bias.BULLISH

    def test_bearish_bias_from_bearish_structure(self):
        """Mostly bearish events should yield BEARISH bias."""
        events = _make_structure_events([
            ("BOS", -1),
            ("CBOS", -1),
            ("CBOS", -1),
            ("CBOS", -1),
            ("CBOS", 1),
        ])
        candles = _make_candles(10)
        bias = determine_bias(candles, events)
        assert bias == Bias.BEARISH

    def test_undefined_when_mixed(self):
        """Equal bullish/bearish events should yield UNDEFINED."""
        events = _make_structure_events([
            ("CBOS", 1),
            ("CBOS", -1),
            ("CBOS", 1),
            ("CBOS", -1),
        ])
        candles = _make_candles(10)
        bias = determine_bias(candles, events)
        assert bias == Bias.UNDEFINED

    def test_undefined_when_no_events(self):
        """Empty structure events should yield UNDEFINED."""
        events = pd.DataFrame(
            columns=["type", "direction", "broken_level", "broken_index", "swing_index"]
        )
        candles = _make_candles(10)
        bias = determine_bias(candles, events)
        assert bias == Bias.UNDEFINED

    def test_bos_weighted_more_than_cbos(self):
        """BOS events (weight 2) should outweigh cBOS (weight 1).

        Scenario: 1 bullish BOS (score 2) vs 2 bearish cBOS (score 2).
        Total bullish=2, bearish=2 => ratio 0.5 each => UNDEFINED.

        But 1 bullish BOS (score 2) vs 1 bearish cBOS (score 1)
        => bullish ratio = 2/3 = 0.667 > 0.6 => BULLISH.
        """
        events = _make_structure_events([
            ("BOS", 1),    # weight 2
            ("CBOS", -1),  # weight 1
        ])
        candles = _make_candles(10)
        bias = determine_bias(candles, events)
        assert bias == Bias.BULLISH

    def test_lookback_filters_old_events(self):
        """Only the last ``lookback`` events should be considered.

        Create 6 bearish events followed by 4 bullish events.
        With lookback=4, only the 4 bullish events are seen => BULLISH.
        """
        entries = [("CBOS", -1)] * 6 + [("CBOS", 1)] * 4
        events = _make_structure_events(entries)
        candles = _make_candles(20)
        bias = determine_bias(candles, events, lookback=4)
        assert bias == Bias.BULLISH


class TestDetermineBiasAt:
    def test_determine_bias_at_filters_by_time(self):
        """Events after the given timestamp should be excluded.

        Create 3 bullish events at indices 0,1,2 and 3 bearish at 3,4,5.
        Candle times: 09:00, 10:00, ..., 14:00.
        With timestamp=11:00, only indices 0,1,2 are included (times
        09:00, 10:00, 11:00).  All 3 are bullish => BULLISH.
        """
        events = _make_structure_events([
            ("CBOS", 1),   # idx 0 -> 09:00
            ("CBOS", 1),   # idx 1 -> 10:00
            ("CBOS", 1),   # idx 2 -> 11:00
            ("CBOS", -1),  # idx 3 -> 12:00  (excluded)
            ("CBOS", -1),  # idx 4 -> 13:00  (excluded)
            ("CBOS", -1),  # idx 5 -> 14:00  (excluded)
        ])
        candles = _make_candles(6)
        timestamp = pd.Timestamp("2024-01-01 11:00", tz="UTC")
        bias = determine_bias_at(candles, events, timestamp)
        assert bias == Bias.BULLISH

    def test_determine_bias_at_empty_when_all_future(self):
        """If all events are after the timestamp, result is UNDEFINED."""
        events = _make_structure_events([
            ("CBOS", 1),   # idx 0 -> 09:00
            ("CBOS", 1),   # idx 1 -> 10:00
        ])
        candles = _make_candles(5)
        # timestamp before any event
        timestamp = pd.Timestamp("2024-01-01 08:00", tz="UTC")
        bias = determine_bias_at(candles, events, timestamp)
        assert bias == Bias.UNDEFINED


class TestGetTrendFromStructure:
    def test_all_bullish(self):
        """All recent events bullish -> BULLISH."""
        events = _make_structure_events([
            ("CBOS", 1),
            ("CBOS", 1),
            ("CBOS", 1),
        ])
        assert get_trend_from_structure(events, n_recent=3) == Bias.BULLISH

    def test_all_bearish(self):
        """All recent events bearish -> BEARISH."""
        events = _make_structure_events([
            ("CBOS", -1),
            ("CBOS", -1),
            ("CBOS", -1),
        ])
        assert get_trend_from_structure(events, n_recent=3) == Bias.BEARISH

    def test_majority_bullish(self):
        """2 out of 3 bullish -> BULLISH."""
        events = _make_structure_events([
            ("CBOS", -1),
            ("CBOS", 1),
            ("CBOS", 1),
        ])
        assert get_trend_from_structure(events, n_recent=3) == Bias.BULLISH

    def test_even_split_undefined(self):
        """Equal split -> UNDEFINED."""
        events = _make_structure_events([
            ("CBOS", 1),
            ("CBOS", -1),
        ])
        assert get_trend_from_structure(events, n_recent=2) == Bias.UNDEFINED

    def test_empty_undefined(self):
        """Empty structure -> UNDEFINED."""
        events = pd.DataFrame(
            columns=["type", "direction", "broken_level", "broken_index", "swing_index"]
        )
        assert get_trend_from_structure(events) == Bias.UNDEFINED

    def test_n_recent_limits_window(self):
        """Only the last n_recent events are examined.

        5 bearish then 3 bullish. With n_recent=3, only the 3 bullish
        events are considered => BULLISH.
        """
        entries = [("CBOS", -1)] * 5 + [("CBOS", 1)] * 3
        events = _make_structure_events(entries)
        assert get_trend_from_structure(events, n_recent=3) == Bias.BULLISH
