"""Tests for fractal (swing high/low) detection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.fractals import SwingStatus, detect_swings, get_swing_points, update_swing_status


def make_zigzag(peaks: list[float], troughs: list[float], points_between: int = 10) -> pd.DataFrame:
    """Create a zigzag OHLC pattern with known peaks and troughs.

    Alternates between going up (to peak) and down (to trough).
    """
    prices = []
    # Start at first trough
    current = troughs[0]
    for i in range(len(peaks)):
        # Go up to peak
        up = np.linspace(current, peaks[i], points_between, endpoint=False)
        prices.extend(up)
        current = peaks[i]
        # Go down to trough (if available)
        if i < len(troughs) - 1:
            current_trough = troughs[i + 1]
        else:
            current_trough = troughs[-1]
        down = np.linspace(current, current_trough, points_between, endpoint=False)
        prices.extend(down)
        current = current_trough

    prices = np.array(prices)
    n = len(prices)
    noise = np.random.default_rng(42).uniform(-0.5, 0.5, n)

    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="1min", tz="UTC")
    opens = prices + noise
    closes = prices - noise
    highs = np.maximum(opens, closes) + abs(noise)
    lows = np.minimum(opens, closes) - abs(noise)

    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })


class TestDetectSwings:
    def test_output_shape(self):
        df = make_zigzag([110, 115, 120], [100, 95, 90])
        swings = detect_swings(df, swing_length=3)
        assert len(swings) == len(df)
        assert "swing_high" in swings.columns
        assert "swing_low" in swings.columns

    def test_detects_swing_highs_in_zigzag(self):
        df = make_zigzag([200, 210, 220], [100, 105, 110], points_between=15)
        swings = detect_swings(df, swing_length=5)
        n_highs = swings["swing_high"].sum()
        assert n_highs >= 2, f"Expected at least 2 swing highs, got {n_highs}"

    def test_detects_swing_lows_in_zigzag(self):
        df = make_zigzag([200, 210, 220], [100, 105, 110], points_between=15)
        swings = detect_swings(df, swing_length=5)
        n_lows = swings["swing_low"].sum()
        assert n_lows >= 2, f"Expected at least 2 swing lows, got {n_lows}"

    def test_swing_high_price_is_correct(self):
        df = make_zigzag([200, 210], [100, 105], points_between=15)
        swings = detect_swings(df, swing_length=3)
        sh_prices = swings["swing_high_price"].dropna()
        # Each swing high price should be close to the peak values
        for p in sh_prices:
            assert p > 150, f"Swing high price {p} seems too low for peaks at 200, 210"

    def test_swing_low_price_is_correct(self):
        df = make_zigzag([200, 210], [100, 105], points_between=15)
        swings = detect_swings(df, swing_length=3)
        sl_prices = swings["swing_low_price"].dropna()
        for p in sl_prices:
            assert p < 150, f"Swing low price {p} seems too high for troughs at 100, 105"

    def test_no_overlap_high_low(self):
        df = make_zigzag([200, 210, 220], [100, 105, 110], points_between=15)
        swings = detect_swings(df, swing_length=5)
        # Same candle should not be both swing high and swing low
        overlap = swings["swing_high"] & swings["swing_low"]
        assert overlap.sum() == 0

    def test_larger_swing_length_fewer_swings(self):
        df = make_zigzag([200, 210, 220, 230], [100, 105, 110, 115], points_between=10)
        swings_3 = detect_swings(df, swing_length=3)
        swings_7 = detect_swings(df, swing_length=7)
        # Larger swing_length should produce fewer or equal swing points
        assert swings_7["swing_high"].sum() <= swings_3["swing_high"].sum()

    def test_flat_data_no_swings(self):
        n = 50
        df = pd.DataFrame({
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "open": np.full(n, 100.0),
            "close": np.full(n, 100.0),
        })
        swings = detect_swings(df, swing_length=5)
        assert swings["swing_high"].sum() == 0
        assert swings["swing_low"].sum() == 0


class TestGetSwingPoints:
    def test_returns_sorted_points(self):
        df = make_zigzag([200, 210], [100, 105], points_between=15)
        swings = detect_swings(df, swing_length=3)
        points = get_swing_points(df, swings)
        assert len(points) > 0
        # Should be sorted by orig_index
        assert points["orig_index"].is_monotonic_increasing

    def test_directions_are_correct(self):
        df = make_zigzag([200, 210], [100, 105], points_between=15)
        swings = detect_swings(df, swing_length=3)
        points = get_swing_points(df, swings)
        assert set(points["direction"].unique()) <= {1, -1}

    def test_all_active_initially(self):
        df = make_zigzag([200, 210], [100, 105], points_between=15)
        swings = detect_swings(df, swing_length=3)
        points = get_swing_points(df, swings)
        assert (points["status"] == SwingStatus.ACTIVE).all()


class TestUpdateSwingStatus:
    def test_sweep_swing_high(self):
        points = pd.DataFrame({
            "orig_index": [10, 20],
            "direction": [1, -1],
            "level": [200.0, 100.0],
            "status": [SwingStatus.ACTIVE, SwingStatus.ACTIVE],
        })
        updated = update_swing_status(points, current_high=201.0, current_low=101.0)
        assert updated.iloc[0]["status"] == SwingStatus.SWEPT
        assert updated.iloc[1]["status"] == SwingStatus.ACTIVE

    def test_sweep_swing_low(self):
        points = pd.DataFrame({
            "orig_index": [10, 20],
            "direction": [1, -1],
            "level": [200.0, 100.0],
            "status": [SwingStatus.ACTIVE, SwingStatus.ACTIVE],
        })
        updated = update_swing_status(points, current_high=199.0, current_low=99.0)
        assert updated.iloc[0]["status"] == SwingStatus.ACTIVE
        assert updated.iloc[1]["status"] == SwingStatus.SWEPT


class TestFractalsRealData:
    def test_detect_on_nas100(self):
        path = Path("data/optimized/NAS100_m1.parquet")
        if not path.exists():
            pytest.skip("NAS100 parquet not available")
        from data.loader import load_parquet
        df = load_parquet(path).head(5000)
        swings = detect_swings(df, swing_length=5)
        n_highs = swings["swing_high"].sum()
        n_lows = swings["swing_low"].sum()
        assert n_highs > 0, "Should detect at least 1 swing high in 5000 candles"
        assert n_lows > 0, "Should detect at least 1 swing low in 5000 candles"
        # No overlap
        assert (swings["swing_high"] & swings["swing_low"]).sum() == 0
