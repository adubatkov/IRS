"""Tests for confirmation counting and validation."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ConfirmationsConfig
from concepts.fvg import FVGStatus
from concepts.structure import StructureType
from strategy.types import Confirmation, ConfirmationType
from strategy.confirmations import (
    check_additional_cbos,
    check_cvb_test,
    check_fvg_inversion,
    check_fvg_wick_reaction,
    check_inversion_test,
    check_liquidity_sweep,
    check_poi_tap,
    check_structure_break,
    collect_confirmations,
    confirmation_count,
    has_fifth_confirm_trap,
    is_ready,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic test data
# ---------------------------------------------------------------------------

def _make_fvgs(
    rows: list[dict] | None = None,
    *,
    direction: int = 1,
    top: float = 108.0,
    bottom: float = 100.0,
    status: FVGStatus = FVGStatus.FRESH,
) -> pd.DataFrame:
    """Build a small FVG DataFrame for testing."""
    if rows is not None:
        return pd.DataFrame(rows)
    midpoint = (top + bottom) / 2
    return pd.DataFrame([{
        "direction": direction,
        "top": top,
        "bottom": bottom,
        "midpoint": midpoint,
        "start_index": 0,
        "creation_index": 2,
        "status": status,
    }])


def _make_liquidity(
    rows: list[dict] | None = None,
    *,
    direction: int = -1,
    level: float = 99.0,
    status: str = "ACTIVE",
) -> pd.DataFrame:
    """Build a small liquidity DataFrame for testing."""
    if rows is not None:
        return pd.DataFrame(rows)
    return pd.DataFrame([{
        "direction": direction,
        "level": level,
        "count": 2,
        "indices": [5, 12],
        "status": status,
    }])


def _make_structure_events(
    rows: list[dict] | None = None,
    *,
    event_type: StructureType = StructureType.BOS,
    direction: int = 1,
    broken_level: float = 115.0,
    broken_index: int = 10,
    swing_index: int = 3,
) -> pd.DataFrame:
    """Build a small structure events DataFrame for testing."""
    if rows is not None:
        return pd.DataFrame(rows)
    return pd.DataFrame([{
        "type": event_type,
        "direction": direction,
        "broken_level": broken_level,
        "broken_index": broken_index,
        "swing_index": swing_index,
    }])


def _make_lifecycle(
    entries: list[dict] | None = None,
    *,
    direction: int = -1,
    top: float = 108.0,
    bottom: float = 100.0,
    status: str = "INVERTED",
    inversion_index: int | None = 10,
) -> list[dict]:
    """Build a small FVG lifecycle list for testing."""
    if entries is not None:
        return entries
    midpoint = (top + bottom) / 2
    return [{
        "fvg_idx": 0,
        "direction": direction,
        "top": top,
        "bottom": bottom,
        "midpoint": midpoint,
        "start_index": 0,
        "creation_index": 2,
        "end_index": inversion_index or 20,
        "status": status,
        "fill_level": bottom - 2,
        "inversion_index": inversion_index,
    }]


def _make_confirmation(
    ctype: ConfirmationType,
    bar_index: int = 0,
    timestamp: pd.Timestamp | None = None,
    details: dict | None = None,
) -> Confirmation:
    """Build a single Confirmation for test lists."""
    ts = timestamp or pd.Timestamp("2024-01-01 10:00", tz="UTC")
    return Confirmation(type=ctype, timestamp=ts, bar_index=bar_index, details=details or {})


def _ts(minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"2024-06-15 10:{minute:02d}:00", tz="UTC")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckPoiTap:
    def test_bullish_poi_tapped_by_candle_low(self):
        """Candle low enters the demand zone."""
        assert check_poi_tap(
            candle_high=112.0, candle_low=107.0,
            poi_top=108.0, poi_bottom=100.0, poi_direction=1,
        ) is True

    def test_bearish_poi_tapped_by_candle_high(self):
        """Candle high enters the supply zone."""
        assert check_poi_tap(
            candle_high=101.0, candle_low=95.0,
            poi_top=108.0, poi_bottom=100.0, poi_direction=-1,
        ) is True

    def test_poi_not_tapped_when_away(self):
        """Candle doesn't reach the zone at all."""
        # Bullish POI, candle fully above
        assert check_poi_tap(
            candle_high=120.0, candle_low=115.0,
            poi_top=108.0, poi_bottom=100.0, poi_direction=1,
        ) is False
        # Bearish POI, candle fully below
        assert check_poi_tap(
            candle_high=90.0, candle_low=85.0,
            poi_top=108.0, poi_bottom=100.0, poi_direction=-1,
        ) is False

    def test_bullish_exact_touch(self):
        """Candle low exactly equals poi_top -- counts as tap."""
        assert check_poi_tap(
            candle_high=112.0, candle_low=108.0,
            poi_top=108.0, poi_bottom=100.0, poi_direction=1,
        ) is True

    def test_bearish_exact_touch(self):
        """Candle high exactly equals poi_bottom -- counts as tap."""
        assert check_poi_tap(
            candle_high=100.0, candle_low=95.0,
            poi_top=108.0, poi_bottom=100.0, poi_direction=-1,
        ) is True


class TestCheckLiquiditySweep:
    def test_bullish_sweep_detected(self):
        """Low goes below sell-side level, closes back above."""
        liq = _make_liquidity(direction=-1, level=99.0)
        result = check_liquidity_sweep(
            candle_high=105.0, candle_low=98.5, candle_close=103.0,
            nearby_liquidity=liq, poi_direction=1,
        )
        assert result is not None
        assert result["level"] == 99.0
        assert result["direction"] == -1

    def test_bearish_sweep_detected(self):
        """High goes above buy-side level, closes back below."""
        liq = _make_liquidity(direction=1, level=115.0)
        result = check_liquidity_sweep(
            candle_high=116.0, candle_low=110.0, candle_close=112.0,
            nearby_liquidity=liq, poi_direction=-1,
        )
        assert result is not None
        assert result["level"] == 115.0
        assert result["direction"] == 1

    def test_no_sweep_when_close_breaks(self):
        """Close stays past level = breakout, not sweep."""
        liq = _make_liquidity(direction=-1, level=99.0)
        result = check_liquidity_sweep(
            candle_high=105.0, candle_low=97.0, candle_close=97.5,
            nearby_liquidity=liq, poi_direction=1,
        )
        assert result is None

    def test_no_sweep_empty_liquidity(self):
        empty = pd.DataFrame(columns=["direction", "level", "count", "indices", "status"])
        result = check_liquidity_sweep(
            candle_high=105.0, candle_low=98.0, candle_close=103.0,
            nearby_liquidity=empty, poi_direction=1,
        )
        assert result is None

    def test_no_sweep_wrong_direction(self):
        """Buy-side liquidity should not be swept for a bullish POI."""
        liq = _make_liquidity(direction=1, level=115.0)
        result = check_liquidity_sweep(
            candle_high=116.0, candle_low=110.0, candle_close=112.0,
            nearby_liquidity=liq, poi_direction=1,
        )
        assert result is None

    def test_no_sweep_already_swept(self):
        """SWEPT liquidity should not trigger again."""
        liq = _make_liquidity(direction=-1, level=99.0, status="SWEPT")
        result = check_liquidity_sweep(
            candle_high=105.0, candle_low=98.0, candle_close=103.0,
            nearby_liquidity=liq, poi_direction=1,
        )
        assert result is None


class TestCheckFvgInversion:
    def test_inversion_detected_at_bar(self):
        """Lifecycle entry with inversion_index == bar."""
        lifecycle = _make_lifecycle(direction=-1, inversion_index=10)
        result = check_fvg_inversion(lifecycle, bar_index=10, poi_direction=1)
        assert result is not None
        assert result["inversion_index"] == 10
        assert result["direction"] == -1

    def test_no_inversion_wrong_bar(self):
        lifecycle = _make_lifecycle(direction=-1, inversion_index=10)
        result = check_fvg_inversion(lifecycle, bar_index=11, poi_direction=1)
        assert result is None

    def test_only_opposing_direction_counts(self):
        """Same direction FVG inversion should not count."""
        lifecycle = _make_lifecycle(direction=1, inversion_index=10)
        result = check_fvg_inversion(lifecycle, bar_index=10, poi_direction=1)
        assert result is None

    def test_bearish_poi_inversion(self):
        """Bullish FVG inversion counts for bearish POI."""
        lifecycle = _make_lifecycle(direction=1, inversion_index=15)
        result = check_fvg_inversion(lifecycle, bar_index=15, poi_direction=-1)
        assert result is not None
        assert result["direction"] == 1

    def test_empty_lifecycle(self):
        result = check_fvg_inversion([], bar_index=10, poi_direction=1)
        assert result is None


class TestCheckInversionTest:
    def test_inverted_fvg_tested_by_wick(self):
        """Bullish POI: candle low dips into the IFVG zone."""
        lifecycle = _make_lifecycle(direction=-1, status="INVERTED", inversion_index=8)
        result = check_inversion_test(
            candle_high=112.0, candle_low=107.0,
            fvg_lifecycle=lifecycle, poi_direction=1,
        )
        assert result is not None
        assert result["top"] == 108.0

    def test_not_tested_when_away(self):
        """Candle stays above the IFVG zone."""
        lifecycle = _make_lifecycle(direction=-1, status="INVERTED", inversion_index=8)
        result = check_inversion_test(
            candle_high=120.0, candle_low=115.0,
            fvg_lifecycle=lifecycle, poi_direction=1,
        )
        assert result is None

    def test_bearish_poi_inversion_test(self):
        """Bearish POI: candle high enters inverted bullish FVG."""
        lifecycle = _make_lifecycle(
            direction=1, top=108.0, bottom=100.0,
            status="INVERTED", inversion_index=8,
        )
        result = check_inversion_test(
            candle_high=101.0, candle_low=95.0,
            fvg_lifecycle=lifecycle, poi_direction=-1,
        )
        assert result is not None
        assert result["bottom"] == 100.0

    def test_non_inverted_fvg_ignored(self):
        """FVG that is not INVERTED should not trigger inversion test."""
        lifecycle = _make_lifecycle(direction=-1, status="PARTIALLY_FILLED", inversion_index=None)
        result = check_inversion_test(
            candle_high=112.0, candle_low=107.0,
            fvg_lifecycle=lifecycle, poi_direction=1,
        )
        assert result is None

    def test_fvg_status_enum_works(self):
        """Status as FVGStatus enum instead of string should also work."""
        lifecycle = _make_lifecycle(direction=-1, inversion_index=8)
        lifecycle[0]["status"] = FVGStatus.INVERTED
        result = check_inversion_test(
            candle_high=112.0, candle_low=107.0,
            fvg_lifecycle=lifecycle, poi_direction=1,
        )
        assert result is not None


class TestCheckStructureBreak:
    def test_bos_at_bar_in_direction(self):
        events = _make_structure_events(
            event_type=StructureType.BOS, direction=1,
            broken_level=115.0, broken_index=10,
        )
        result = check_structure_break(events, bar_index=10, poi_direction=1)
        assert result is not None
        assert result["type"] == "BOS"
        assert result["broken_level"] == 115.0

    def test_cbos_at_bar_in_direction(self):
        events = _make_structure_events(
            event_type=StructureType.CBOS, direction=1,
            broken_level=118.0, broken_index=15,
        )
        result = check_structure_break(events, bar_index=15, poi_direction=1)
        assert result is not None
        assert result["type"] == "CBOS"

    def test_no_match_different_bar(self):
        events = _make_structure_events(broken_index=10, direction=1)
        result = check_structure_break(events, bar_index=11, poi_direction=1)
        assert result is None

    def test_no_match_wrong_direction(self):
        events = _make_structure_events(broken_index=10, direction=-1)
        result = check_structure_break(events, bar_index=10, poi_direction=1)
        assert result is None

    def test_empty_events(self):
        empty = pd.DataFrame(columns=["type", "direction", "broken_level", "broken_index", "swing_index"])
        result = check_structure_break(empty, bar_index=10, poi_direction=1)
        assert result is None


class TestCheckFvgWickReaction:
    def test_wick_reaction_detected(self):
        """Bullish POI: candle dips into bullish FVG, closes above midpoint with lower wick."""
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        result = check_fvg_wick_reaction(
            candle_open=110.0, candle_high=112.0,
            candle_low=106.0, candle_close=111.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is not None
        assert result["direction"] == 1
        assert result["wick_size"] == 110.0 - 106.0  # body_low - low

    def test_bearish_wick_reaction(self):
        """Bearish POI: candle pokes into bearish FVG, closes below midpoint with upper wick."""
        fvgs = _make_fvgs(direction=-1, top=108.0, bottom=100.0, status=FVGStatus.TESTED)
        result = check_fvg_wick_reaction(
            candle_open=97.0, candle_high=101.0,
            candle_low=95.0, candle_close=96.0,
            nearby_fvgs=fvgs, poi_direction=-1,
        )
        assert result is not None
        assert result["direction"] == -1
        # upper wick = high - max(open, close) = 101 - 97 = 4
        assert result["wick_size"] == 4.0

    def test_no_reaction_without_wick(self):
        """Candle enters FVG but has no rejection wick (doji-like)."""
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        # open==low so lower wick is 0
        result = check_fvg_wick_reaction(
            candle_open=106.0, candle_high=112.0,
            candle_low=106.0, candle_close=111.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is None

    def test_no_reaction_close_below_midpoint(self):
        """Candle enters FVG but closes below midpoint (bearish, not rejection)."""
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        # midpoint = 104, close = 102 < 104
        result = check_fvg_wick_reaction(
            candle_open=110.0, candle_high=112.0,
            candle_low=101.0, candle_close=102.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is None

    def test_mitigated_fvg_ignored(self):
        """FVGs that are MITIGATED or INVERTED should not trigger wick reaction."""
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.MITIGATED)
        result = check_fvg_wick_reaction(
            candle_open=110.0, candle_high=112.0,
            candle_low=106.0, candle_close=111.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is None

    def test_wrong_direction_fvg_ignored(self):
        """Bullish POI should not react to bearish FVG."""
        fvgs = _make_fvgs(direction=-1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        result = check_fvg_wick_reaction(
            candle_open=110.0, candle_high=112.0,
            candle_low=106.0, candle_close=111.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is None


class TestCheckCvbTest:
    def test_midpoint_touched(self):
        """Bullish POI: candle low reaches near midpoint of bullish FVG."""
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        # midpoint = 104.0, tolerance 0.1% => 104 * 1.001 = 104.104
        result = check_cvb_test(
            candle_high=112.0, candle_low=104.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is not None
        assert result["midpoint"] == 104.0

    def test_midpoint_not_touched(self):
        """Candle doesn't reach near the midpoint."""
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        # midpoint = 104.0, candle_low = 106 > 104.104
        result = check_cvb_test(
            candle_high=112.0, candle_low=106.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is None

    def test_bearish_poi_midpoint_touch(self):
        """Bearish POI: candle high reaches near midpoint of bearish FVG."""
        fvgs = _make_fvgs(direction=-1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        # midpoint = 104.0, tolerance => 104 * 0.999 = 103.896
        result = check_cvb_test(
            candle_high=104.0, candle_low=95.0,
            nearby_fvgs=fvgs, poi_direction=-1,
        )
        assert result is not None

    def test_wrong_direction_fvg_ignored(self):
        """CVB test only considers FVGs in same direction as POI."""
        fvgs = _make_fvgs(direction=-1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        result = check_cvb_test(
            candle_high=112.0, candle_low=104.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is None

    def test_inactive_fvg_ignored(self):
        """Fully filled or mitigated FVGs should be ignored."""
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FULLY_FILLED)
        result = check_cvb_test(
            candle_high=112.0, candle_low=104.0,
            nearby_fvgs=fvgs, poi_direction=1,
        )
        assert result is None


class TestCheckAdditionalCbos:
    def test_cbos_after_first_structure_break(self):
        """cBOS at bar_index counts when prior STRUCTURE_BREAK exists."""
        events = _make_structure_events(
            event_type=StructureType.CBOS, direction=1,
            broken_index=20, broken_level=120.0,
        )
        existing = [_make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=10)]
        result = check_additional_cbos(events, bar_index=20, poi_direction=1, existing_confirms=existing)
        assert result is not None
        assert result["type"] == "CBOS"
        assert result["broken_level"] == 120.0

    def test_no_cbos_without_prior_structure_break(self):
        """cBOS should not count without a prior STRUCTURE_BREAK."""
        events = _make_structure_events(
            event_type=StructureType.CBOS, direction=1,
            broken_index=20, broken_level=120.0,
        )
        existing = [_make_confirmation(ConfirmationType.POI_TAP, bar_index=5)]
        result = check_additional_cbos(events, bar_index=20, poi_direction=1, existing_confirms=existing)
        assert result is None

    def test_bos_does_not_count_as_additional(self):
        """Only CBOS events count as additional, not BOS."""
        events = _make_structure_events(
            event_type=StructureType.BOS, direction=1,
            broken_index=20, broken_level=120.0,
        )
        existing = [_make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=10)]
        result = check_additional_cbos(events, bar_index=20, poi_direction=1, existing_confirms=existing)
        assert result is None

    def test_empty_structure_events(self):
        events = pd.DataFrame(columns=["type", "direction", "broken_level", "broken_index", "swing_index"])
        existing = [_make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=10)]
        result = check_additional_cbos(events, bar_index=20, poi_direction=1, existing_confirms=existing)
        assert result is None


class TestCollectConfirmations:
    """Integration tests for the master collect_confirmations function."""

    def _default_config(self) -> ConfirmationsConfig:
        return ConfirmationsConfig(min_count=5, max_count=8)

    def _candle(
        self,
        open: float = 110.0,
        high: float = 112.0,
        low: float = 107.0,
        close: float = 111.0,
    ) -> pd.Series:
        return pd.Series({"open": open, "high": high, "low": low, "close": close})

    def _poi(self, direction: int = 1) -> dict:
        return {"direction": direction, "top": 108.0, "bottom": 100.0, "midpoint": 104.0}

    def test_single_bar_multiple_confirms(self):
        """A single bar can trigger multiple different confirmation types."""
        # Set up data so POI_TAP and LIQUIDITY_SWEEP both fire
        candle = self._candle(open=110.0, high=112.0, low=98.0, close=103.0)
        poi = self._poi(direction=1)
        liq = _make_liquidity(direction=-1, level=99.0)
        empty_fvgs = _make_fvgs(rows=[])
        empty_events = _make_structure_events(rows=[])

        result = collect_confirmations(
            candle=candle, bar_index=10, timestamp=_ts(1),
            poi_data=poi, existing_confirms=[],
            nearby_fvgs=empty_fvgs, fvg_lifecycle=[],
            nearby_liquidity=liq, structure_events=empty_events,
            config=self._default_config(),
        )
        types = [c.type for c in result]
        assert ConfirmationType.POI_TAP in types
        assert ConfirmationType.LIQUIDITY_SWEEP in types
        assert len(result) >= 2

    def test_incremental_over_bars(self):
        """Call collect_confirmations multiple times; list grows incrementally."""
        config = self._default_config()
        poi = self._poi(direction=1)
        empty_fvgs = _make_fvgs(rows=[])
        empty_events = _make_structure_events(rows=[])
        empty_liq = _make_liquidity(rows=[])

        # Bar 10: POI_TAP
        candle_1 = self._candle(low=107.0)
        confirms = collect_confirmations(
            candle=candle_1, bar_index=10, timestamp=_ts(0),
            poi_data=poi, existing_confirms=[],
            nearby_fvgs=empty_fvgs, fvg_lifecycle=[], nearby_liquidity=empty_liq,
            structure_events=empty_events, config=config,
        )
        assert len(confirms) == 1
        assert confirms[0].type == ConfirmationType.POI_TAP

        # Bar 11: POI_TAP again (new bar, same type allowed)
        candle_2 = self._candle(low=106.0)
        confirms = collect_confirmations(
            candle=candle_2, bar_index=11, timestamp=_ts(1),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=empty_fvgs, fvg_lifecycle=[], nearby_liquidity=empty_liq,
            structure_events=empty_events, config=config,
        )
        assert len(confirms) == 2

    def test_no_duplicate_same_type_same_bar(self):
        """Same type + same bar_index should not be counted twice."""
        config = self._default_config()
        poi = self._poi(direction=1)
        empty_fvgs = _make_fvgs(rows=[])
        empty_events = _make_structure_events(rows=[])
        empty_liq = _make_liquidity(rows=[])

        candle = self._candle(low=107.0)
        confirms = collect_confirmations(
            candle=candle, bar_index=10, timestamp=_ts(0),
            poi_data=poi, existing_confirms=[],
            nearby_fvgs=empty_fvgs, fvg_lifecycle=[], nearby_liquidity=empty_liq,
            structure_events=empty_events, config=config,
        )
        count_first = len(confirms)

        # Call again with same bar_index
        confirms = collect_confirmations(
            candle=candle, bar_index=10, timestamp=_ts(0),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=empty_fvgs, fvg_lifecycle=[], nearby_liquidity=empty_liq,
            structure_events=empty_events, config=config,
        )
        assert len(confirms) == count_first  # No growth

    def test_max_count_cap(self):
        """Total confirmations should not exceed config.max_count."""
        config = ConfirmationsConfig(min_count=2, max_count=3)
        poi = self._poi(direction=1)
        empty_fvgs = _make_fvgs(rows=[])
        empty_events = _make_structure_events(rows=[])

        # Pre-fill with 3 confirmations (already at max)
        existing = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.FVG_INVERSION, bar_index=3),
        ]

        candle = self._candle(low=107.0)
        liq = _make_liquidity(direction=-1, level=99.0)
        result = collect_confirmations(
            candle=candle, bar_index=10, timestamp=_ts(5),
            poi_data=poi, existing_confirms=existing,
            nearby_fvgs=empty_fvgs, fvg_lifecycle=[], nearby_liquidity=liq,
            structure_events=empty_events, config=config,
        )
        assert len(result) == 3  # Stays at max

    def test_fvg_wick_blocked_under_5(self):
        """FVG_WICK_REACTION should NOT be added when < 5 confirms exist."""
        config = ConfirmationsConfig(min_count=5, max_count=8)
        poi = self._poi(direction=1)
        # Bullish FVG that the candle would react to
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        empty_events = _make_structure_events(rows=[])
        empty_liq = _make_liquidity(rows=[])

        # Only 2 prior confirms
        existing = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
        ]

        # Candle that would trigger wick reaction: low enters FVG, close above midpoint, has wick
        candle = self._candle(open=110.0, high=112.0, low=106.0, close=111.0)
        result = collect_confirmations(
            candle=candle, bar_index=10, timestamp=_ts(5),
            poi_data=poi, existing_confirms=existing,
            nearby_fvgs=fvgs, fvg_lifecycle=[], nearby_liquidity=empty_liq,
            structure_events=empty_events, config=config,
        )
        types = [c.type for c in result]
        assert ConfirmationType.FVG_WICK_REACTION not in types

    def test_fvg_wick_allowed_after_5(self):
        """FVG_WICK_REACTION IS added when 5+ confirms already exist."""
        config = ConfirmationsConfig(min_count=5, max_count=10)
        poi = self._poi(direction=1)
        fvgs = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        empty_events = _make_structure_events(rows=[])
        empty_liq = _make_liquidity(rows=[])

        # 5 prior confirms
        existing = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.FVG_INVERSION, bar_index=3),
            _make_confirmation(ConfirmationType.INVERSION_TEST, bar_index=4),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=5),
        ]

        # Candle that triggers wick reaction
        candle = self._candle(open=110.0, high=112.0, low=106.0, close=111.0)
        result = collect_confirmations(
            candle=candle, bar_index=10, timestamp=_ts(5),
            poi_data=poi, existing_confirms=existing,
            nearby_fvgs=fvgs, fvg_lifecycle=[], nearby_liquidity=empty_liq,
            structure_events=empty_events, config=config,
        )
        types = [c.type for c in result]
        assert ConfirmationType.FVG_WICK_REACTION in types

    def test_does_not_mutate_existing_list(self):
        """collect_confirmations should return a new list, not mutate the input."""
        config = self._default_config()
        poi = self._poi(direction=1)
        empty_fvgs = _make_fvgs(rows=[])
        empty_events = _make_structure_events(rows=[])
        empty_liq = _make_liquidity(rows=[])

        existing = []
        candle = self._candle(low=107.0)
        result = collect_confirmations(
            candle=candle, bar_index=10, timestamp=_ts(0),
            poi_data=poi, existing_confirms=existing,
            nearby_fvgs=empty_fvgs, fvg_lifecycle=[], nearby_liquidity=empty_liq,
            structure_events=empty_events, config=config,
        )
        assert len(existing) == 0  # Original not mutated
        assert len(result) >= 1

    def test_all_eight_types_can_fire(self):
        """Verify all 8 confirmation types can be collected across multiple bars.

        We carefully control candle prices to avoid unintended POI_TAP triggers.
        POI zone is top=108, bottom=100 (bullish/demand). A candle with low>108
        will NOT trigger POI_TAP.
        """
        config = ConfirmationsConfig(min_count=5, max_count=20)
        poi = self._poi(direction=1)

        confirms: list[Confirmation] = []

        # Bar 10: POI_TAP only (candle low enters zone, no other data)
        confirms = collect_confirmations(
            candle=self._candle(open=110.0, high=112.0, low=107.0, close=111.0),
            bar_index=10, timestamp=_ts(0),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=_make_fvgs(rows=[]), fvg_lifecycle=[],
            nearby_liquidity=_make_liquidity(rows=[]),
            structure_events=_make_structure_events(rows=[]), config=config,
        )
        assert ConfirmationType.POI_TAP in [c.type for c in confirms]

        # Bar 11: LIQUIDITY_SWEEP only (candle low sweeps sell-side but stays above zone)
        # low=98 < level=99 -> sweep. close=109 > 99 -> valid. low=98 < 108 -> also POI_TAP.
        # That's fine, we just need LIQUIDITY_SWEEP present.
        confirms = collect_confirmations(
            candle=self._candle(open=110.0, high=112.0, low=98.0, close=109.0),
            bar_index=11, timestamp=_ts(1),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=_make_fvgs(rows=[]), fvg_lifecycle=[],
            nearby_liquidity=_make_liquidity(direction=-1, level=99.0),
            structure_events=_make_structure_events(rows=[]), config=config,
        )
        assert ConfirmationType.LIQUIDITY_SWEEP in [c.type for c in confirms]

        # Bar 12: FVG_INVERSION only (candle stays above zone)
        lifecycle_inv = _make_lifecycle(direction=-1, inversion_index=12)
        confirms = collect_confirmations(
            candle=self._candle(open=115.0, high=118.0, low=113.0, close=117.0),
            bar_index=12, timestamp=_ts(2),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=_make_fvgs(rows=[]),
            fvg_lifecycle=lifecycle_inv,
            nearby_liquidity=_make_liquidity(rows=[]),
            structure_events=_make_structure_events(rows=[]), config=config,
        )
        assert ConfirmationType.FVG_INVERSION in [c.type for c in confirms]

        # Bar 13: INVERSION_TEST only (candle low touches IFVG top=108)
        lifecycle_test = _make_lifecycle(direction=-1, status="INVERTED", inversion_index=12)
        confirms = collect_confirmations(
            candle=self._candle(open=112.0, high=114.0, low=107.5, close=113.0),
            bar_index=13, timestamp=_ts(3),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=_make_fvgs(rows=[]),
            fvg_lifecycle=lifecycle_test,
            nearby_liquidity=_make_liquidity(rows=[]),
            structure_events=_make_structure_events(rows=[]), config=config,
        )
        assert ConfirmationType.INVERSION_TEST in [c.type for c in confirms]

        # Bar 14: STRUCTURE_BREAK only (candle above zone)
        sb_events = _make_structure_events(
            event_type=StructureType.BOS, direction=1,
            broken_index=14, broken_level=115.0,
        )
        confirms = collect_confirmations(
            candle=self._candle(open=116.0, high=120.0, low=115.0, close=119.0),
            bar_index=14, timestamp=_ts(4),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=_make_fvgs(rows=[]), fvg_lifecycle=[],
            nearby_liquidity=_make_liquidity(rows=[]),
            structure_events=sb_events, config=config,
        )
        assert ConfirmationType.STRUCTURE_BREAK in [c.type for c in confirms]
        assert len(confirms) >= 5  # Must be >= 5 for wick reaction

        # Bar 15: FVG_WICK_REACTION (allowed since 5+ confirms)
        # Candle dips into bullish FVG (low=106 <= top=108), closes above midpoint=104,
        # has lower wick (min(110,111)-106 = 4 > 0). Also triggers POI_TAP.
        fvgs_wick = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        confirms = collect_confirmations(
            candle=self._candle(open=110.0, high=112.0, low=106.0, close=111.0),
            bar_index=15, timestamp=_ts(5),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=fvgs_wick, fvg_lifecycle=[],
            nearby_liquidity=_make_liquidity(rows=[]),
            structure_events=_make_structure_events(rows=[]), config=config,
        )
        assert ConfirmationType.FVG_WICK_REACTION in [c.type for c in confirms]

        # Bar 16: CVB_TEST (candle low touches midpoint=104 of bullish FVG, also POI_TAP)
        fvgs_cvb = _make_fvgs(direction=1, top=108.0, bottom=100.0, status=FVGStatus.FRESH)
        confirms = collect_confirmations(
            candle=self._candle(open=106.0, high=107.0, low=104.0, close=106.0),
            bar_index=16, timestamp=_ts(6),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=fvgs_cvb, fvg_lifecycle=[],
            nearby_liquidity=_make_liquidity(rows=[]),
            structure_events=_make_structure_events(rows=[]), config=config,
        )
        assert ConfirmationType.CVB_TEST in [c.type for c in confirms]

        # Bar 17: ADDITIONAL_CBOS (STRUCTURE_BREAK already exists, candle above zone)
        cbos_events = _make_structure_events(
            event_type=StructureType.CBOS, direction=1,
            broken_index=17, broken_level=120.0,
        )
        confirms = collect_confirmations(
            candle=self._candle(open=120.0, high=122.0, low=119.0, close=121.0),
            bar_index=17, timestamp=_ts(7),
            poi_data=poi, existing_confirms=confirms,
            nearby_fvgs=_make_fvgs(rows=[]), fvg_lifecycle=[],
            nearby_liquidity=_make_liquidity(rows=[]),
            structure_events=cbos_events, config=config,
        )
        assert ConfirmationType.ADDITIONAL_CBOS in [c.type for c in confirms]

        # Verify all 8 types present
        all_types = {c.type for c in confirms}
        for ct in ConfirmationType:
            assert ct in all_types, f"{ct} missing from collected confirmations"


class TestIsReady:
    def test_ready_at_5(self):
        config = ConfirmationsConfig(min_count=5, max_count=8)
        confirms = [_make_confirmation(ConfirmationType.POI_TAP, bar_index=i) for i in range(5)]
        assert is_ready(confirms, config) is True

    def test_not_ready_at_4(self):
        config = ConfirmationsConfig(min_count=5, max_count=8)
        confirms = [_make_confirmation(ConfirmationType.POI_TAP, bar_index=i) for i in range(4)]
        assert is_ready(confirms, config) is False

    def test_ready_with_custom_config(self):
        config = ConfirmationsConfig(min_count=3, max_count=6)
        confirms = [_make_confirmation(ConfirmationType.POI_TAP, bar_index=i) for i in range(3)]
        assert is_ready(confirms, config) is True

    def test_ready_above_min(self):
        config = ConfirmationsConfig(min_count=5, max_count=8)
        confirms = [_make_confirmation(ConfirmationType.POI_TAP, bar_index=i) for i in range(7)]
        assert is_ready(confirms, config) is True


class TestConfirmationCount:
    def test_count_empty(self):
        assert confirmation_count([]) == 0

    def test_count_several(self):
        confirms = [_make_confirmation(ConfirmationType.POI_TAP, bar_index=i) for i in range(5)]
        assert confirmation_count(confirms) == 5


class TestFifthConfirmTrap:
    def test_trap_detected(self):
        """5 confirms, none FVG-related, last is STRUCTURE_BREAK."""
        confirms = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=3),
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=4),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=5),
        ]
        assert has_fifth_confirm_trap(confirms) is True

    def test_trap_with_additional_cbos_last(self):
        """Trap also triggers if last confirm is ADDITIONAL_CBOS."""
        confirms = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=3),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=4),
            _make_confirmation(ConfirmationType.ADDITIONAL_CBOS, bar_index=5),
        ]
        assert has_fifth_confirm_trap(confirms) is True

    def test_no_trap_with_fvg_inversion(self):
        """Having FVG_INVERSION among confirms negates the trap."""
        confirms = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.FVG_INVERSION, bar_index=3),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=4),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=5),
        ]
        assert has_fifth_confirm_trap(confirms) is False

    def test_no_trap_with_inversion_test(self):
        confirms = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.INVERSION_TEST, bar_index=3),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=4),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=5),
        ]
        assert has_fifth_confirm_trap(confirms) is False

    def test_no_trap_with_wick_reaction(self):
        confirms = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.FVG_WICK_REACTION, bar_index=3),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=4),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=5),
        ]
        assert has_fifth_confirm_trap(confirms) is False

    def test_no_trap_at_4_confirms(self):
        """Less than 5 confirmations: never a trap."""
        confirms = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=3),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=4),
        ]
        assert has_fifth_confirm_trap(confirms) is False

    def test_no_trap_when_last_not_structural(self):
        """5 confirms without FVG-related, but last is POI_TAP (not structural)."""
        confirms = [
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=1),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=2),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=3),
            _make_confirmation(ConfirmationType.ADDITIONAL_CBOS, bar_index=4),
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=5),
        ]
        assert has_fifth_confirm_trap(confirms) is False

    def test_trap_with_more_than_5(self):
        """Trap can fire at 6+ confirms too."""
        confirms = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=1),
            _make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=2),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=3),
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=4),
            _make_confirmation(ConfirmationType.CVB_TEST, bar_index=5),
            _make_confirmation(ConfirmationType.STRUCTURE_BREAK, bar_index=6),
        ]
        assert has_fifth_confirm_trap(confirms) is True
