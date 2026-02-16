"""Tests for Fair Value Gap (FVG) detection and lifecycle."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.fvg import FVGStatus, detect_fvg, track_fvg_lifecycle, update_fvg_status


def make_bullish_fvg():
    """Create data with a clear bullish FVG."""
    return pd.DataFrame({
        "high":  [100, 105, 112],  # candle 1 high=100, candle 3 low=108 > 100
        "low":   [98,  102, 108],
        "close": [99,  104, 111],
        "open":  [99,  101, 107],
    })


def make_bearish_fvg():
    """Create data with a clear bearish FVG."""
    return pd.DataFrame({
        "high":  [100, 95,  88],  # candle 1 low=98, candle 3 high=88 < 98
        "low":   [98,  90,  85],
        "close": [99,  91,  86],
        "open":  [99,  96,  90],
    })


def make_no_fvg():
    """Create data without any FVG (overlapping candles)."""
    return pd.DataFrame({
        "high":  [100, 102, 101],
        "low":   [98,  99,  99],
        "close": [99,  101, 100],
        "open":  [99,  100, 101],
    })


class TestDetectFVG:
    def test_detects_bullish_fvg(self):
        df = make_bullish_fvg()
        fvgs = detect_fvg(df, min_gap_pct=0)
        bullish = fvgs[fvgs["direction"] == 1]
        assert len(bullish) >= 1
        fvg = bullish.iloc[0]
        assert fvg["bottom"] == 100  # high of candle 1
        assert fvg["top"] == 108     # low of candle 3
        assert fvg["status"] == FVGStatus.FRESH

    def test_detects_bearish_fvg(self):
        df = make_bearish_fvg()
        fvgs = detect_fvg(df, min_gap_pct=0)
        bearish = fvgs[fvgs["direction"] == -1]
        assert len(bearish) >= 1
        fvg = bearish.iloc[0]
        assert fvg["top"] == 98   # low of candle 1
        assert fvg["bottom"] == 88  # high of candle 3

    def test_no_fvg_on_overlapping(self):
        df = make_no_fvg()
        fvgs = detect_fvg(df, min_gap_pct=0)
        assert len(fvgs) == 0

    def test_min_gap_filter(self):
        df = make_bullish_fvg()
        # Gap is 108-100=8, price~111, 8/111=7.2%
        fvgs_no_filter = detect_fvg(df, min_gap_pct=0)
        fvgs_strict = detect_fvg(df, min_gap_pct=0.1)  # 10% filter
        assert len(fvgs_no_filter) >= len(fvgs_strict)

    def test_midpoint_calculation(self):
        df = make_bullish_fvg()
        fvgs = detect_fvg(df, min_gap_pct=0)
        fvg = fvgs.iloc[0]
        assert fvg["midpoint"] == (fvg["top"] + fvg["bottom"]) / 2

    def test_returns_correct_columns(self):
        df = make_bullish_fvg()
        fvgs = detect_fvg(df, min_gap_pct=0)
        expected = {"direction", "top", "bottom", "midpoint",
                    "start_index", "creation_index", "status"}
        assert expected <= set(fvgs.columns)

    def test_start_index_is_first_candle(self):
        df = make_bullish_fvg()
        fvgs = detect_fvg(df, min_gap_pct=0)
        fvg = fvgs.iloc[0]
        assert fvg["start_index"] == 0  # First candle of pattern
        assert fvg["creation_index"] == 2  # Third candle

    def test_empty_on_tiny_data(self):
        df = pd.DataFrame({"high": [100], "low": [99], "close": [99.5], "open": [99.5]})
        fvgs = detect_fvg(df)
        assert len(fvgs) == 0


class TestUpdateFVGStatus:
    def test_tested_on_wick_touch(self):
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "status": FVGStatus.FRESH,
        }])
        updated = update_fvg_status(fvgs, candle_high=110, candle_low=106, candle_close=109, mitigation_mode="close")
        assert updated.iloc[0]["status"] == FVGStatus.TESTED

    def test_partially_filled_on_midpoint_close(self):
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "status": FVGStatus.FRESH,
        }])
        updated = update_fvg_status(fvgs, candle_high=110, candle_low=103, candle_close=103, mitigation_mode="close")
        assert updated.iloc[0]["status"] == FVGStatus.PARTIALLY_FILLED

    def test_inverted_on_close_through(self):
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "status": FVGStatus.FRESH,
        }])
        updated = update_fvg_status(fvgs, candle_high=110, candle_low=95, candle_close=97, mitigation_mode="close")
        assert updated.iloc[0]["status"] == FVGStatus.INVERTED

    def test_bearish_fvg_tested(self):
        fvgs = pd.DataFrame([{
            "direction": -1, "top": 98.0, "bottom": 88.0,
            "midpoint": 93.0, "status": FVGStatus.FRESH,
        }])
        updated = update_fvg_status(fvgs, candle_high=90, candle_low=85, candle_close=89, mitigation_mode="close")
        assert updated.iloc[0]["status"] == FVGStatus.TESTED

    def test_mitigated_not_updated_again(self):
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "status": FVGStatus.MITIGATED,
        }])
        updated = update_fvg_status(fvgs, candle_high=200, candle_low=50, candle_close=100)
        assert updated.iloc[0]["status"] == FVGStatus.MITIGATED

    def test_ce_mode_mitigates_at_midpoint(self):
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "status": FVGStatus.FRESH,
        }])
        updated = update_fvg_status(fvgs, candle_high=110, candle_low=103, candle_close=105, mitigation_mode="ce")
        assert updated.iloc[0]["status"] == FVGStatus.MITIGATED


class TestTrackFVGLifecycle:
    """Tests for bar-by-bar FVG lifecycle tracking."""

    def _make_lifecycle_data(self):
        """Bullish FVG at bars 0-2, then subsequent price action."""
        # Bars 0-2: create bullish FVG (top=108, bottom=100)
        # Bar 3: price stays above → FRESH
        # Bar 4: wick touches zone (low=106) → TESTED
        # Bar 5: deeper into zone past midpoint (low=103) → PARTIALLY_FILLED
        # Bar 6: closes below bottom → INVERTED
        return pd.DataFrame({
            "high":  [100, 105, 112, 115, 113, 112, 110],
            "low":   [98,  102, 108, 110, 106, 103, 95],
            "close": [99,  104, 111, 114, 112, 108, 97],
            "open":  [99,  101, 107, 111, 110, 110, 108],
        })

    def test_basic_lifecycle_to_inversion(self):
        df = self._make_lifecycle_data()
        fvgs = detect_fvg(df.iloc[:3], min_gap_pct=0)
        assert len(fvgs) == 1

        results = track_fvg_lifecycle(df, fvgs, mitigation_mode="close", max_age_bars=50)
        assert len(results) == 1
        r = results[0]
        assert r["status"] == FVGStatus.INVERTED
        assert r["inversion_index"] == 6
        assert r["end_index"] == 6
        assert r["fill_level"] == 95  # Deepest penetration

    def test_partial_fill_no_inversion(self):
        """FVG partially filled but not inverted → stays PARTIALLY_FILLED."""
        df = pd.DataFrame({
            "high":  [100, 105, 112, 115, 113, 112, 111, 110],
            "low":   [98,  102, 108, 110, 106, 103, 105, 106],
            "close": [99,  104, 111, 114, 112, 108, 109, 109],
            "open":  [99,  101, 107, 111, 110, 110, 107, 108],
        })
        fvgs = detect_fvg(df.iloc[:3], min_gap_pct=0)
        results = track_fvg_lifecycle(df, fvgs, mitigation_mode="close", max_age_bars=50)
        assert len(results) == 1
        r = results[0]
        assert r["status"] == FVGStatus.PARTIALLY_FILLED
        assert r["inversion_index"] is None
        assert r["fill_level"] == 103  # Deepest was bar 5

    def test_max_age_expiry(self):
        """FVG expires after max_age_bars if untouched."""
        df = pd.DataFrame({
            "high":  [100, 105, 112] + [115] * 10,
            "low":   [98,  102, 108] + [110] * 10,
            "close": [99,  104, 111] + [114] * 10,
            "open":  [99,  101, 107] + [111] * 10,
        })
        fvgs = detect_fvg(df.iloc[:3], min_gap_pct=0)
        results = track_fvg_lifecycle(df, fvgs, mitigation_mode="close", max_age_bars=5)
        assert len(results) == 1
        r = results[0]
        assert r["status"] == FVGStatus.FRESH  # Never touched
        assert r["fill_level"] is None
        # end_index should be at max_age offset from creation
        assert r["end_index"] == 7  # creation(2) + max_age(5) = 7

    def test_bearish_fvg_lifecycle(self):
        """Bearish FVG gets inverted when close goes above top."""
        df = pd.DataFrame({
            "high":  [100, 95, 88, 85, 90, 95, 102],
            "low":   [98,  90, 85, 82, 84, 88, 96],
            "close": [99,  91, 86, 83, 89, 94, 101],
            "open":  [99,  96, 90, 86, 85, 89, 95],
        })
        fvgs = detect_fvg(df.iloc[:3], min_gap_pct=0)
        bearish = fvgs[fvgs["direction"] == -1]
        assert len(bearish) >= 1

        results = track_fvg_lifecycle(df, bearish.reset_index(drop=True),
                                      mitigation_mode="close", max_age_bars=50)
        assert len(results) == 1
        r = results[0]
        assert r["status"] == FVGStatus.INVERTED
        assert r["inversion_index"] == 6


class TestFVGRealData:
    def test_detect_on_nas100(self):
        path = Path("data/optimized/NAS100_m1.parquet")
        if not path.exists():
            pytest.skip("NAS100 parquet not available")
        from data.loader import load_parquet
        df = load_parquet(path).head(5000).reset_index(drop=True)
        fvgs = detect_fvg(df, min_gap_pct=0.0005)
        assert len(fvgs) > 0, "Should detect FVGs in 5000 NAS100 candles"
        # Both directions should be present
        assert 1 in fvgs["direction"].values
        assert -1 in fvgs["direction"].values
        # All should be FRESH
        assert (fvgs["status"] == FVGStatus.FRESH).all()
