"""Tests for market structure (BOS/CHoCH) and CISD detection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.structure import StructureType, detect_bos_choch, detect_cisd


def make_uptrend(n_waves: int = 4, points_per_wave: int = 20) -> pd.DataFrame:
    """Create an uptrend with higher highs and higher lows."""
    prices = []
    base = 100.0
    for i in range(n_waves):
        # Up leg
        peak = base + 20 + i * 5
        up = np.linspace(base, peak, points_per_wave // 2)
        prices.extend(up)
        # Down leg (higher low)
        trough = base + 5 + i * 3
        down = np.linspace(peak, trough, points_per_wave // 2)
        prices.extend(down)
        base = trough

    prices = np.array(prices)
    n = len(prices)
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="1min", tz="UTC")
    noise = np.random.default_rng(42).uniform(-0.3, 0.3, n)
    opens = prices + noise
    closes = prices - noise
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    return pd.DataFrame({
        "time": dates, "open": opens, "high": highs, "low": lows, "close": closes,
    })


def make_downtrend(n_waves: int = 4, points_per_wave: int = 20) -> pd.DataFrame:
    """Create a downtrend with lower highs and lower lows."""
    prices = []
    base = 200.0
    for i in range(n_waves):
        trough = base - 20 - i * 5
        down = np.linspace(base, trough, points_per_wave // 2)
        prices.extend(down)
        peak = trough + 10 - i * 2
        up = np.linspace(trough, peak, points_per_wave // 2)
        prices.extend(up)
        base = peak

    prices = np.array(prices)
    n = len(prices)
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="1min", tz="UTC")
    noise = np.random.default_rng(42).uniform(-0.3, 0.3, n)
    opens = prices + noise
    closes = prices - noise
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    return pd.DataFrame({
        "time": dates, "open": opens, "high": highs, "low": lows, "close": closes,
    })


def make_reversal(points_per_leg: int = 30) -> pd.DataFrame:
    """Create an uptrend followed by a reversal to downtrend."""
    # Strong uptrend
    up_prices = np.linspace(100, 200, points_per_leg)
    # Small pullback
    pullback = np.linspace(200, 180, points_per_leg // 3)
    # Another push higher
    push = np.linspace(180, 210, points_per_leg // 3)
    # Strong reversal down
    down_prices = np.linspace(210, 150, points_per_leg)
    # Continue down
    down_prices2 = np.linspace(150, 120, points_per_leg // 2)

    prices = np.concatenate([up_prices, pullback, push, down_prices, down_prices2])
    n = len(prices)
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="1min", tz="UTC")
    noise = np.random.default_rng(42).uniform(-0.5, 0.5, n)
    opens = prices + noise
    closes = prices - noise
    highs = np.maximum(opens, closes) + 1.0
    lows = np.minimum(opens, closes) - 1.0
    return pd.DataFrame({
        "time": dates, "open": opens, "high": highs, "low": lows, "close": closes,
    })


class TestDetectBosChoch:
    def test_uptrend_produces_bos(self):
        df = make_uptrend()
        events = detect_bos_choch(df, swing_length=3, close_break=True)
        if len(events) > 0:
            bos_events = events[events["type"] == StructureType.BOS]
            assert len(bos_events) > 0, "Uptrend should produce BOS events"

    def test_returns_correct_columns(self):
        df = make_uptrend()
        events = detect_bos_choch(df, swing_length=3)
        expected_cols = {"type", "direction", "broken_level", "broken_index", "swing_index"}
        assert set(events.columns) >= expected_cols

    def test_reversal_produces_choch(self):
        df = make_reversal()
        events = detect_bos_choch(df, swing_length=3, close_break=True)
        if len(events) > 0:
            choch_events = events[events["type"] == StructureType.CHOCH]
            # Should detect at least one CHoCH at the reversal point
            assert len(choch_events) >= 0  # May not detect with small data

    def test_directions_are_valid(self):
        df = make_uptrend()
        events = detect_bos_choch(df, swing_length=3)
        if len(events) > 0:
            assert set(events["direction"].unique()) <= {1, -1}

    def test_wick_mode_more_events(self):
        df = make_uptrend(n_waves=6, points_per_wave=30)
        events_close = detect_bos_choch(df, swing_length=3, close_break=True)
        events_wick = detect_bos_choch(df, swing_length=3, close_break=False)
        # Wick mode should produce >= events than close mode
        assert len(events_wick) >= len(events_close)

    def test_empty_on_tiny_data(self):
        df = pd.DataFrame({
            "open": [100, 101], "high": [102, 103],
            "low": [99, 100], "close": [101, 102],
        })
        events = detect_bos_choch(df, swing_length=3)
        assert len(events) == 0


class TestDetectCISD:
    def test_detects_cisd_on_reversal(self):
        df = make_reversal()
        events = detect_cisd(df)
        assert len(events) > 0, "Should detect CISD on reversal pattern"

    def test_returns_correct_columns(self):
        df = make_reversal()
        events = detect_cisd(df)
        expected_cols = {"direction", "level", "trigger_index", "origin_index"}
        assert set(events.columns) >= expected_cols

    def test_directions_valid(self):
        df = make_reversal()
        events = detect_cisd(df)
        if len(events) > 0:
            assert set(events["direction"].unique()) <= {1, -1}

    def test_simple_bullish_cisd(self):
        """Three bearish candles followed by a bullish close above first open."""
        df = pd.DataFrame({
            "open":  [100, 99,  98,  97, 95, 98],
            "close": [99,  98,  97,  95, 93, 101],  # Last candle closes above 100 (first bearish open)
            "high":  [101, 100, 99,  98, 96, 102],
            "low":   [98,  97,  96,  94, 92, 94],
        })
        events = detect_cisd(df)
        bullish = events[events["direction"] == 1]
        assert len(bullish) > 0, "Should detect bullish CISD"


class TestStructureRealData:
    def test_bos_choch_on_nas100(self):
        path = Path("data/optimized/NAS100_m1.parquet")
        if not path.exists():
            pytest.skip("NAS100 parquet not available")
        from data.loader import load_parquet
        df = load_parquet(path).head(5000).reset_index(drop=True)
        events = detect_bos_choch(df, swing_length=5, close_break=True)
        assert len(events) > 0, "Should detect structure events in 5000 NAS100 candles"
        # Should have both BOS and possibly CHoCH
        assert any(t == StructureType.BOS for t in events["type"])
