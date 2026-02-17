"""Tests for FTA (First Trouble Area) detection and handling."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.fta_handler import (
    check_fta_invalidation,
    check_fta_validation,
    classify_fta_distance,
    detect_fta,
    should_enter_with_fta,
)


def _make_pois(rows: list[dict]) -> pd.DataFrame:
    """Build a POI DataFrame from a list of dicts."""
    cols = ["direction", "top", "bottom", "midpoint", "score", "components",
            "component_count", "status"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df.columns:
            if col == "components":
                df[col] = [[] for _ in range(len(df))]
            elif col == "component_count":
                df[col] = 0
            elif col == "score":
                df[col] = 1.0
    return df


def _make_fta(direction: int, top: float, bottom: float, score: float = 1.0) -> dict:
    """Build an FTA dict for testing."""
    return {
        "direction": direction,
        "top": top,
        "bottom": bottom,
        "midpoint": (top + bottom) / 2,
        "score": score,
    }


class TestDetectFta:
    def test_finds_bearish_fta_for_long(self):
        """Supply zone between price and target is detected as FTA for long."""
        pois = _make_pois([
            {"direction": -1, "top": 108.0, "bottom": 105.0, "midpoint": 106.5,
             "score": 3.0, "status": "ACTIVE"},
        ])
        fta = detect_fta(current_price=100.0, target=110.0, direction=1,
                         active_pois=pois)
        assert fta is not None
        assert fta["direction"] == -1
        assert fta["bottom"] == 105.0
        assert fta["top"] == 108.0

    def test_finds_bullish_fta_for_short(self):
        """Demand zone between price and target is detected as FTA for short."""
        pois = _make_pois([
            {"direction": 1, "top": 95.0, "bottom": 92.0, "midpoint": 93.5,
             "score": 2.0, "status": "ACTIVE"},
        ])
        fta = detect_fta(current_price=100.0, target=90.0, direction=-1,
                         active_pois=pois)
        assert fta is not None
        assert fta["direction"] == 1
        assert fta["top"] == 95.0
        assert fta["bottom"] == 92.0

    def test_no_fta_when_path_clear(self):
        """No opposing POIs in path returns None."""
        # Only same-direction POIs (bullish), no bearish in path for long
        pois = _make_pois([
            {"direction": 1, "top": 108.0, "bottom": 105.0, "midpoint": 106.5,
             "score": 3.0, "status": "ACTIVE"},
        ])
        fta = detect_fta(current_price=100.0, target=110.0, direction=1,
                         active_pois=pois)
        assert fta is None

    def test_no_fta_when_empty_pois(self):
        """Empty DataFrame returns None."""
        pois = _make_pois([])
        fta = detect_fta(current_price=100.0, target=110.0, direction=1,
                         active_pois=pois)
        assert fta is None

    def test_picks_closest_fta(self):
        """Multiple opposing POIs: picks the one nearest to current price."""
        pois = _make_pois([
            # Farther supply zone
            {"direction": -1, "top": 109.0, "bottom": 107.0, "midpoint": 108.0,
             "score": 2.0, "status": "ACTIVE"},
            # Closer supply zone
            {"direction": -1, "top": 104.0, "bottom": 102.0, "midpoint": 103.0,
             "score": 4.0, "status": "ACTIVE"},
        ])
        fta = detect_fta(current_price=100.0, target=110.0, direction=1,
                         active_pois=pois)
        assert fta is not None
        # Should pick the one with bottom=102 (closest to price)
        assert fta["bottom"] == 102.0
        assert fta["top"] == 104.0

    def test_ignores_mitigated_pois(self):
        """Mitigated POIs are not considered as FTA."""
        pois = _make_pois([
            {"direction": -1, "top": 108.0, "bottom": 105.0, "midpoint": 106.5,
             "score": 3.0, "status": "MITIGATED"},
        ])
        fta = detect_fta(current_price=100.0, target=110.0, direction=1,
                         active_pois=pois)
        assert fta is None

    def test_tested_poi_can_be_fta(self):
        """TESTED POIs are still valid FTA candidates."""
        pois = _make_pois([
            {"direction": -1, "top": 108.0, "bottom": 105.0, "midpoint": 106.5,
             "score": 3.0, "status": "TESTED"},
        ])
        fta = detect_fta(current_price=100.0, target=110.0, direction=1,
                         active_pois=pois)
        assert fta is not None

    def test_poi_outside_path_ignored(self):
        """POI above the target is not in path and should be ignored."""
        pois = _make_pois([
            # This bearish POI is above the target, not between price and target
            {"direction": -1, "top": 115.0, "bottom": 112.0, "midpoint": 113.5,
             "score": 3.0, "status": "ACTIVE"},
        ])
        fta = detect_fta(current_price=100.0, target=110.0, direction=1,
                         active_pois=pois)
        assert fta is None


class TestClassifyFtaDistance:
    def test_close_fta(self):
        """FTA within threshold fraction is classified as close."""
        # price=100, target=110, FTA mid=102 -> fraction=2/10=0.2 < 0.3
        fta = _make_fta(direction=-1, top=103.0, bottom=101.0, score=1.0)
        result = classify_fta_distance(fta, current_price=100.0, target=110.0,
                                       close_threshold_pct=0.3)
        assert result == "close"

    def test_far_fta(self):
        """FTA beyond threshold fraction is classified as far."""
        # price=100, target=110, FTA mid=107 -> fraction=7/10=0.7 > 0.3
        fta = _make_fta(direction=-1, top=108.0, bottom=106.0, score=1.0)
        result = classify_fta_distance(fta, current_price=100.0, target=110.0,
                                       close_threshold_pct=0.3)
        assert result == "far"

    def test_boundary_is_close(self):
        """FTA at exactly threshold is classified as close."""
        # price=100, target=110, FTA mid=103 -> fraction=3/10=0.3 == 0.3
        fta = _make_fta(direction=-1, top=104.0, bottom=102.0, score=1.0)
        result = classify_fta_distance(fta, current_price=100.0, target=110.0,
                                       close_threshold_pct=0.3)
        assert result == "close"

    def test_zero_range_is_close(self):
        """When price equals target (zero range), FTA is close."""
        fta = _make_fta(direction=-1, top=101.0, bottom=99.0, score=1.0)
        result = classify_fta_distance(fta, current_price=100.0, target=100.0)
        assert result == "close"


class TestFtaInvalidation:
    def test_invalidated_for_long(self):
        """For long: candle close above FTA top invalidates it."""
        fta = _make_fta(direction=-1, top=108.0, bottom=105.0)
        assert check_fta_invalidation(fta, candle_close=109.0, direction=1) is True

    def test_not_invalidated_for_long(self):
        """For long: candle close below FTA top means FTA still holds."""
        fta = _make_fta(direction=-1, top=108.0, bottom=105.0)
        assert check_fta_invalidation(fta, candle_close=107.0, direction=1) is False

    def test_invalidated_for_short(self):
        """For short: candle close below FTA bottom invalidates it."""
        fta = _make_fta(direction=1, top=95.0, bottom=92.0)
        assert check_fta_invalidation(fta, candle_close=91.0, direction=-1) is True

    def test_not_invalidated_for_short(self):
        """For short: candle close above FTA bottom means FTA still holds."""
        fta = _make_fta(direction=1, top=95.0, bottom=92.0)
        assert check_fta_invalidation(fta, candle_close=93.0, direction=-1) is False

    def test_close_at_boundary_not_invalidated_long(self):
        """Close exactly at FTA top does not invalidate (must be strictly above)."""
        fta = _make_fta(direction=-1, top=108.0, bottom=105.0)
        assert check_fta_invalidation(fta, candle_close=108.0, direction=1) is False


class TestFtaValidation:
    def test_validated_for_long(self):
        """Long: price reached FTA zone but closed back below = rejection."""
        fta = _make_fta(direction=-1, top=108.0, bottom=105.0)
        assert check_fta_validation(
            fta, candle_high=106.0, candle_low=101.0, candle_close=103.0,
            direction=1,
        ) is True

    def test_not_validated_no_reach(self):
        """Long: price did not reach FTA zone at all."""
        fta = _make_fta(direction=-1, top=108.0, bottom=105.0)
        assert check_fta_validation(
            fta, candle_high=104.0, candle_low=100.0, candle_close=102.0,
            direction=1,
        ) is False

    def test_not_validated_closed_through(self):
        """Long: price closed inside/through the zone, not a rejection."""
        fta = _make_fta(direction=-1, top=108.0, bottom=105.0)
        assert check_fta_validation(
            fta, candle_high=107.0, candle_low=101.0, candle_close=106.0,
            direction=1,
        ) is False

    def test_validated_for_short(self):
        """Short: price reached FTA zone but closed back above = rejection."""
        fta = _make_fta(direction=1, top=95.0, bottom=92.0)
        assert check_fta_validation(
            fta, candle_high=98.0, candle_low=94.0, candle_close=96.0,
            direction=-1,
        ) is True

    def test_not_validated_for_short_no_reach(self):
        """Short: price did not reach FTA zone."""
        fta = _make_fta(direction=1, top=95.0, bottom=92.0)
        assert check_fta_validation(
            fta, candle_high=100.0, candle_low=96.0, candle_close=97.0,
            direction=-1,
        ) is False


class TestShouldEnterWithFta:
    def test_no_fta_enter(self):
        """No FTA means clear path, should enter."""
        should_enter, reason = should_enter_with_fta(None, "")
        assert should_enter is True
        assert "no FTA" in reason

    def test_far_fta_enter(self):
        """Far FTA allows entry."""
        fta = _make_fta(direction=-1, top=108.0, bottom=105.0)
        should_enter, reason = should_enter_with_fta(fta, "far")
        assert should_enter is True
        assert "far" in reason.lower()

    def test_close_fta_dont_enter(self):
        """Close FTA blocks entry."""
        fta = _make_fta(direction=-1, top=103.0, bottom=101.0)
        should_enter, reason = should_enter_with_fta(fta, "close")
        assert should_enter is False
        assert "close" in reason.lower()
