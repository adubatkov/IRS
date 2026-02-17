"""Tests for add-on position management."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from config import StrategyConfig
from strategy.types import POIPhase, POIState, SignalType
from strategy.addons import (
    evaluate_addon,
    find_addon_candidates,
    should_addon_bu,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = pd.Timestamp("2024-01-02 14:00", tz="UTC")


def _poi_data(direction: int = 1) -> dict:
    if direction == 1:
        return {"direction": 1, "top": 21100.0, "bottom": 21000.0, "midpoint": 21050.0}
    return {"direction": -1, "top": 21100.0, "bottom": 21000.0, "midpoint": 21050.0}


def _candle(open_: float, high: float, low: float, close: float) -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def _positioned_poi(
    direction: int = 1,
    sl: float = 20950.0,
    target: float = 21500.0,
) -> POIState:
    return POIState(
        poi_id="poi_main",
        poi_data=_poi_data(direction),
        phase=POIPhase.POSITIONED,
        entry_price=21120.0,
        stop_loss=sl,
        target=target,
    )


def _local_pois(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _empty_local_pois() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "direction", "top", "bottom", "midpoint", "status",
    ])


def _structure_events(direction: int = 1, broken_index: int = 300) -> pd.DataFrame:
    return pd.DataFrame([{
        "direction": direction,
        "broken_index": broken_index,
        "broken_level": 21200.0,
        "type": "BOS",
    }])


def _empty_structure() -> pd.DataFrame:
    return pd.DataFrame(columns=["direction", "broken_index", "broken_level", "type"])


# ---------------------------------------------------------------------------
# TestFindAddonCandidates
# ---------------------------------------------------------------------------

class TestFindAddonCandidates:

    def test_finds_bullish_between_price_and_target(self):
        """Long: finds bullish POIs between price and target."""
        pois = _local_pois([
            {"direction": 1, "top": 21280, "bottom": 21250, "midpoint": 21265, "status": "ACTIVE"},
            {"direction": 1, "top": 21380, "bottom": 21350, "midpoint": 21365, "status": "ACTIVE"},
            {"direction": 1, "top": 21600, "bottom": 21570, "midpoint": 21585, "status": "ACTIVE"},  # beyond target
            {"direction": -1, "top": 21300, "bottom": 21270, "midpoint": 21285, "status": "ACTIVE"},  # wrong dir
        ])
        result = find_addon_candidates(
            direction=1, current_price=21200.0, target=21500.0,
            local_pois=pois, timestamp=TS,
        )
        assert len(result) == 2
        assert result.iloc[0]["midpoint"] == 21265  # closest first
        assert result.iloc[1]["midpoint"] == 21365

    def test_finds_bearish_between_target_and_price(self):
        """Short: finds bearish POIs between target and price."""
        pois = _local_pois([
            {"direction": -1, "top": 20950, "bottom": 20920, "midpoint": 20935, "status": "ACTIVE"},
            {"direction": -1, "top": 20850, "bottom": 20820, "midpoint": 20835, "status": "ACTIVE"},
            {"direction": -1, "top": 20600, "bottom": 20570, "midpoint": 20585, "status": "ACTIVE"},  # beyond target
        ])
        result = find_addon_candidates(
            direction=-1, current_price=21000.0, target=20700.0,
            local_pois=pois, timestamp=TS,
        )
        assert len(result) == 2
        # Sorted by midpoint descending (closest to price first)
        assert result.iloc[0]["midpoint"] == 20935
        assert result.iloc[1]["midpoint"] == 20835

    def test_empty_when_no_candidates(self):
        """No matching POIs -> empty DataFrame."""
        result = find_addon_candidates(
            direction=1, current_price=21200.0, target=21500.0,
            local_pois=_empty_local_pois(), timestamp=TS,
        )
        assert len(result) == 0

    def test_sorted_by_proximity(self):
        """Results are sorted by proximity to current price."""
        pois = _local_pois([
            {"direction": 1, "top": 21380, "bottom": 21350, "midpoint": 21365, "status": "ACTIVE"},
            {"direction": 1, "top": 21280, "bottom": 21250, "midpoint": 21265, "status": "TESTED"},
            {"direction": 1, "top": 21330, "bottom": 21300, "midpoint": 21315, "status": "ACTIVE"},
        ])
        result = find_addon_candidates(
            direction=1, current_price=21200.0, target=21500.0,
            local_pois=pois, timestamp=TS,
        )
        midpoints = list(result["midpoint"])
        assert midpoints == sorted(midpoints), "Should be ascending for long"


# ---------------------------------------------------------------------------
# TestEvaluateAddon
# ---------------------------------------------------------------------------

class TestEvaluateAddon:

    def test_addon_fires_on_touch_with_structure(self):
        """Candle touches candidate POI zone + structure break -> ADD_ON signal."""
        state = _positioned_poi(direction=1, sl=20950.0, target=21500.0)
        candidate = pd.Series({
            "top": 21280.0, "bottom": 21250.0, "midpoint": 21265.0,
        })
        # For long: candle low=21270 <= poi_top=21280 -> touches
        candle = _candle(21290, 21300, 21270, 21285.0)
        events = _structure_events(direction=1, broken_index=300)
        sig = evaluate_addon(state, candidate, candle, 300, TS, events, StrategyConfig())
        assert sig is not None
        assert sig.type == SignalType.ADD_ON
        assert sig.position_size_mult == 0.5

    def test_no_addon_without_touch(self):
        """Candle doesn't touch candidate zone -> None."""
        state = _positioned_poi(direction=1, sl=20950.0, target=21500.0)
        candidate = pd.Series({
            "top": 21280.0, "bottom": 21250.0, "midpoint": 21265.0,
        })
        # For long: candle low=21290 > poi_top=21280 -> no touch
        candle = _candle(21300, 21310, 21290, 21295.0)
        events = _structure_events(direction=1, broken_index=300)
        sig = evaluate_addon(state, candidate, candle, 300, TS, events, StrategyConfig())
        assert sig is None

    def test_no_addon_without_structure(self):
        """Candle touches zone but no structure -> None."""
        state = _positioned_poi(direction=1, sl=20950.0, target=21500.0)
        candidate = pd.Series({
            "top": 21280.0, "bottom": 21250.0, "midpoint": 21265.0,
        })
        candle = _candle(21290, 21300, 21270, 21285.0)
        sig = evaluate_addon(state, candidate, candle, 300, TS, _empty_structure(), StrategyConfig())
        assert sig is None

    def test_no_addon_when_not_positioned(self):
        """Phase not POSITIONED/MANAGING -> None."""
        state = POIState(
            poi_id="x", poi_data=_poi_data(), phase=POIPhase.READY,
            entry_price=21120.0, stop_loss=20950.0, target=21500.0,
        )
        candidate = pd.Series({
            "top": 21280.0, "bottom": 21250.0, "midpoint": 21265.0,
        })
        candle = _candle(21290, 21300, 21270, 21285.0)
        events = _structure_events(direction=1, broken_index=300)
        sig = evaluate_addon(state, candidate, candle, 300, TS, events, StrategyConfig())
        assert sig is None


# ---------------------------------------------------------------------------
# TestShouldAddonBu
# ---------------------------------------------------------------------------

class TestShouldAddonBu:

    def test_long_be_when_price_moved(self):
        """LONG: price moved > 3*commission -> True."""
        entry = 21265.0
        # Need current > entry * (1 + 3*0.0006) = entry * 1.0018
        threshold = entry * (1 + 3 * 0.0006)
        assert should_addon_bu(entry, threshold + 1.0, direction=1) is True

    def test_long_no_be_when_close(self):
        """LONG: price hasn't moved enough -> False."""
        entry = 21265.0
        threshold = entry * (1 + 3 * 0.0006)
        assert should_addon_bu(entry, threshold - 1.0, direction=1) is False

    def test_short_be_when_price_moved(self):
        """SHORT: price moved down > 3*commission -> True."""
        entry = 21265.0
        threshold = entry * (1 - 3 * 0.0006)
        assert should_addon_bu(entry, threshold - 1.0, direction=-1) is True

    def test_short_no_be_when_close(self):
        """SHORT: price hasn't moved enough -> False."""
        entry = 21265.0
        threshold = entry * (1 - 3 * 0.0006)
        assert should_addon_bu(entry, threshold + 1.0, direction=-1) is False
