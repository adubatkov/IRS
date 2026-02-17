"""Tests for the POI state machine and StateMachineManager."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ConfirmationsConfig
from concepts.fvg import FVGStatus
from concepts.structure import StructureType
from strategy.types import (
    Confirmation,
    ConfirmationType,
    POIPhase,
    POIState,
)
from context.state_machine import (
    ConceptData,
    StateMachineManager,
    make_poi_id,
    transition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"2024-06-15 10:{minute:02d}:00", tz="UTC")


def _candle(
    open: float = 110.0,
    high: float = 112.0,
    low: float = 107.0,
    close: float = 111.0,
) -> pd.Series:
    return pd.Series({"open": open, "high": high, "low": low, "close": close})


def _poi_data(direction: int = 1) -> dict:
    """Default bullish POI zone: top=108, bottom=100, midpoint=104."""
    return {
        "direction": direction,
        "top": 108.0,
        "bottom": 100.0,
        "midpoint": 104.0,
        "score": 7,
        "timeframe": "4H",
        "components": [],
    }


def _empty_concept_data() -> ConceptData:
    """Concept data with no entries -- no confirmations will fire."""
    return ConceptData(
        nearby_fvgs=pd.DataFrame(columns=["direction", "top", "bottom", "midpoint", "status"]),
        fvg_lifecycle=[],
        nearby_liquidity=pd.DataFrame(columns=["direction", "level", "status"]),
        structure_events=pd.DataFrame(
            columns=["type", "direction", "broken_index", "broken_level"]
        ),
    )


def _rich_concept_data(bar_index: int, direction: int = 1) -> ConceptData:
    """Concept data set up so multiple confirmations can fire on a single bar.

    For a bullish POI (direction=1):
    - Sell-side liquidity at 99.0 (direction=-1) -> sweep when candle_low < 99 and close >= 99
    - Bearish FVG lifecycle with inversion at bar_index -> FVG_INVERSION
    - Inverted bearish FVG (IFVG) testable -> INVERSION_TEST
    - BOS at bar_index in direction -> STRUCTURE_BREAK
    - Bullish FVG for CVB test -> CVB_TEST
    """
    if direction == 1:
        liq_dir = -1
        opposing_dir = -1
    else:
        liq_dir = 1
        opposing_dir = 1

    nearby_fvgs = pd.DataFrame([{
        "direction": direction,
        "top": 108.0,
        "bottom": 100.0,
        "midpoint": 104.0,
        "start_index": 0,
        "creation_index": 2,
        "status": FVGStatus.FRESH,
    }])

    fvg_lifecycle = [
        {
            "fvg_idx": 0,
            "direction": opposing_dir,
            "top": 108.0,
            "bottom": 100.0,
            "midpoint": 104.0,
            "start_index": 0,
            "creation_index": 2,
            "end_index": bar_index,
            "status": "INVERTED",
            "fill_level": 98.0,
            "inversion_index": bar_index,
        },
    ]

    nearby_liquidity = pd.DataFrame([{
        "direction": liq_dir,
        "level": 99.0,
        "count": 2,
        "indices": [5, 12],
        "status": "ACTIVE",
    }])

    structure_events = pd.DataFrame([
        {
            "type": StructureType.BOS,
            "direction": direction,
            "broken_level": 115.0,
            "broken_index": bar_index,
            "swing_index": 3,
        },
        {
            "type": StructureType.CBOS,
            "direction": direction,
            "broken_level": 118.0,
            "broken_index": bar_index,
            "swing_index": 5,
        },
    ])

    return ConceptData(
        nearby_fvgs=nearby_fvgs,
        fvg_lifecycle=fvg_lifecycle,
        nearby_liquidity=nearby_liquidity,
        structure_events=structure_events,
    )


def _make_confirmation(
    ctype: ConfirmationType,
    bar_index: int = 0,
    timestamp: pd.Timestamp | None = None,
) -> Confirmation:
    ts = timestamp or _ts(0)
    return Confirmation(type=ctype, timestamp=ts, bar_index=bar_index, details={})


def _make_state(
    poi_id: str = "4H_1_0",
    direction: int = 1,
    phase: POIPhase = POIPhase.IDLE,
    confirmations: list[Confirmation] | None = None,
) -> POIState:
    return POIState(
        poi_id=poi_id,
        poi_data=_poi_data(direction),
        phase=phase,
        confirmations=confirmations or [],
        created_at=_ts(0),
        last_updated=_ts(0),
    )


def _default_config() -> ConfirmationsConfig:
    return ConfirmationsConfig(min_count=5, max_count=8)


# ---------------------------------------------------------------------------
# TestMakePoiId
# ---------------------------------------------------------------------------

class TestMakePoiId:
    def test_format(self):
        result = make_poi_id("4H", 1, 100)
        assert result == "4H_1_100"

    def test_negative_direction(self):
        result = make_poi_id("15m", -1, 50)
        assert result == "15m_-1_50"


# ---------------------------------------------------------------------------
# TestTransition
# ---------------------------------------------------------------------------

class TestTransition:
    def test_idle_to_tapped_then_collecting(self):
        """Candle touches the bullish POI zone -> transitions through
        POI_TAPPED to COLLECTING on the same bar."""
        state = _make_state(phase=POIPhase.IDLE)
        config = _default_config()
        # Candle low = 107 <= poi_top = 108 -> tap fires
        candle = _candle(open=110.0, high=112.0, low=107.0, close=111.0)
        concept = _empty_concept_data()

        updated, signals = transition(state, candle, 10, _ts(1), concept, config)

        # Should be COLLECTING (passed through POI_TAPPED on same bar)
        assert updated.phase == POIPhase.COLLECTING
        assert len(signals) == 0
        assert updated.last_updated == _ts(1)
        # POI_TAP confirmation should have been collected
        types = [c.type for c in updated.confirmations]
        assert ConfirmationType.POI_TAP in types

    def test_idle_stays_when_not_tapped(self):
        """Candle stays away from POI -> no transition."""
        state = _make_state(phase=POIPhase.IDLE)
        config = _default_config()
        # Candle fully above POI zone: low=115 > poi_top=108
        candle = _candle(open=118.0, high=120.0, low=115.0, close=119.0)
        concept = _empty_concept_data()

        updated, signals = transition(state, candle, 10, _ts(1), concept, config)

        assert updated.phase == POIPhase.IDLE
        assert len(signals) == 0
        assert len(updated.confirmations) == 0

    def test_collecting_accumulates(self):
        """Multiple bars in COLLECTING phase accumulate confirmations."""
        config = _default_config()
        concept = _empty_concept_data()

        # Start in COLLECTING with 1 existing confirmation
        state = _make_state(phase=POIPhase.COLLECTING)
        state.confirmations = [_make_confirmation(ConfirmationType.POI_TAP, bar_index=10)]

        # Bar 11: candle taps POI again -> another POI_TAP at new bar_index
        candle = _candle(open=110.0, high=112.0, low=106.0, close=111.0)
        updated, _ = transition(state, candle, 11, _ts(1), concept, config)

        assert updated.phase == POIPhase.COLLECTING
        assert len(updated.confirmations) >= 2

    def test_collecting_to_ready(self):
        """When is_ready() returns True, phase transitions to READY."""
        config = ConfirmationsConfig(min_count=2, max_count=8)

        # Start COLLECTING with 1 confirmation
        state = _make_state(phase=POIPhase.COLLECTING)
        state.confirmations = [_make_confirmation(ConfirmationType.LIQUIDITY_SWEEP, bar_index=9)]

        # Candle that taps POI -> adds POI_TAP -> total = 2 >= min_count=2
        candle = _candle(open=110.0, high=112.0, low=107.0, close=111.0)
        concept = _empty_concept_data()

        updated, signals = transition(state, candle, 10, _ts(1), concept, config)

        assert updated.phase == POIPhase.READY
        assert len(updated.confirmations) >= 2

    def test_ready_stays_ready(self):
        """READY phase does not auto-advance via transition()."""
        state = _make_state(phase=POIPhase.READY)
        state.confirmations = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=i)
            for i in range(5)
        ]
        config = _default_config()
        candle = _candle(low=107.0)
        concept = _empty_concept_data()

        updated, signals = transition(state, candle, 20, _ts(5), concept, config)

        assert updated.phase == POIPhase.READY
        assert len(signals) == 0

    def test_closed_is_terminal(self):
        """CLOSED state does not change regardless of input."""
        state = _make_state(phase=POIPhase.CLOSED)
        config = _default_config()
        candle = _candle(low=99.0)  # Would tap POI if not closed
        concept = _rich_concept_data(bar_index=10)

        updated, signals = transition(state, candle, 10, _ts(1), concept, config)

        assert updated.phase == POIPhase.CLOSED
        assert len(signals) == 0
        assert len(updated.confirmations) == 0


# ---------------------------------------------------------------------------
# TestStateMachineManager
# ---------------------------------------------------------------------------

class TestStateMachineManager:
    def test_register_poi(self):
        """register_poi creates a POI in IDLE phase."""
        mgr = StateMachineManager(_default_config())
        poi_id = mgr.register_poi(_poi_data(1), "4H", _ts(0))

        state = mgr.get_state(poi_id)
        assert state.phase == POIPhase.IDLE
        assert state.poi_data["direction"] == 1
        assert state.created_at == _ts(0)

    def test_unique_ids(self):
        """Each registration returns a unique ID."""
        mgr = StateMachineManager(_default_config())
        id1 = mgr.register_poi(_poi_data(1), "4H", _ts(0))
        id2 = mgr.register_poi(_poi_data(1), "4H", _ts(1))
        id3 = mgr.register_poi(_poi_data(-1), "15m", _ts(2))

        assert len({id1, id2, id3}) == 3

    def test_update_progresses_idle(self):
        """update() with a tapping candle moves IDLE -> COLLECTING."""
        mgr = StateMachineManager(_default_config())
        poi_id = mgr.register_poi(_poi_data(1), "4H", _ts(0))

        # Candle that taps the bullish POI (low=107 <= top=108)
        candle = _candle(open=110.0, high=112.0, low=107.0, close=111.0)
        concept = _empty_concept_data()
        signals = mgr.update(candle, 10, _ts(1), concept)

        state = mgr.get_state(poi_id)
        assert state.phase == POIPhase.COLLECTING
        assert len(signals) == 0
        assert len(state.confirmations) >= 1

    def test_update_skips_ready(self):
        """READY state is not modified by update()."""
        mgr = StateMachineManager(_default_config())
        poi_id = mgr.register_poi(_poi_data(1), "4H", _ts(0))

        # Manually set to READY with some confirmations
        state = mgr.get_state(poi_id)
        state.phase = POIPhase.READY
        state.confirmations = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=i)
            for i in range(5)
        ]

        # Update should skip this POI
        candle = _candle(low=107.0)
        concept = _empty_concept_data()
        signals = mgr.update(candle, 20, _ts(5), concept)

        assert mgr.get_state(poi_id).phase == POIPhase.READY
        assert len(signals) == 0

    def test_get_active_states(self):
        """get_active_states excludes CLOSED POIs."""
        mgr = StateMachineManager(_default_config())
        id1 = mgr.register_poi(_poi_data(1), "4H", _ts(0))
        id2 = mgr.register_poi(_poi_data(-1), "15m", _ts(1))

        mgr.close_poi(id1)

        active = mgr.get_active_states()
        assert len(active) == 1
        assert active[0].poi_id == id2

    def test_get_positioned_states(self):
        """get_positioned_states returns only POSITIONED and MANAGING."""
        mgr = StateMachineManager(_default_config())
        id1 = mgr.register_poi(_poi_data(1), "4H", _ts(0))
        id2 = mgr.register_poi(_poi_data(-1), "15m", _ts(1))
        mgr.register_poi(_poi_data(1), "1H", _ts(2))

        mgr.set_positioned(id1, entry_price=110.0, stop_loss=99.0, target=120.0)
        mgr.set_positioned(id2, entry_price=90.0, stop_loss=101.0, target=80.0)
        mgr.set_managing(id2)
        # id3 stays IDLE

        positioned = mgr.get_positioned_states()
        ids = {s.poi_id for s in positioned}
        assert ids == {id1, id2}

    def test_get_ready_states(self):
        """get_ready_states returns only READY phase POIs."""
        mgr = StateMachineManager(_default_config())
        id1 = mgr.register_poi(_poi_data(1), "4H", _ts(0))
        mgr.register_poi(_poi_data(-1), "15m", _ts(1))

        # Manually set id1 to READY
        state1 = mgr.get_state(id1)
        state1.phase = POIPhase.READY
        state1.confirmations = [
            _make_confirmation(ConfirmationType.POI_TAP, bar_index=i)
            for i in range(5)
        ]

        ready = mgr.get_ready_states()
        assert len(ready) == 1
        assert ready[0].poi_id == id1

    def test_set_positioned(self):
        """set_positioned updates entry/sl/target and changes phase."""
        mgr = StateMachineManager(_default_config())
        poi_id = mgr.register_poi(_poi_data(1), "4H", _ts(0))

        mgr.set_positioned(poi_id, entry_price=110.0, stop_loss=99.0, target=125.0)

        state = mgr.get_state(poi_id)
        assert state.phase == POIPhase.POSITIONED
        assert state.entry_price == 110.0
        assert state.stop_loss == 99.0
        assert state.target == 125.0

    def test_set_managing(self):
        """set_managing transitions from POSITIONED to MANAGING."""
        mgr = StateMachineManager(_default_config())
        poi_id = mgr.register_poi(_poi_data(1), "4H", _ts(0))
        mgr.set_positioned(poi_id, entry_price=110.0, stop_loss=99.0, target=125.0)

        mgr.set_managing(poi_id)

        state = mgr.get_state(poi_id)
        assert state.phase == POIPhase.MANAGING

    def test_invalidate_poi(self):
        """invalidate_poi forces a POI to CLOSED."""
        mgr = StateMachineManager(_default_config())
        poi_id = mgr.register_poi(_poi_data(1), "4H", _ts(0))

        mgr.invalidate_poi(poi_id, reason="Price broke through")

        state = mgr.get_state(poi_id)
        assert state.phase == POIPhase.CLOSED

    def test_close_poi(self):
        """close_poi forces CLOSED state."""
        mgr = StateMachineManager(_default_config())
        poi_id = mgr.register_poi(_poi_data(1), "4H", _ts(0))
        mgr.set_positioned(poi_id, entry_price=110.0, stop_loss=99.0, target=125.0)

        mgr.close_poi(poi_id)

        state = mgr.get_state(poi_id)
        assert state.phase == POIPhase.CLOSED

    def test_get_state_unknown_raises(self):
        """get_state with unknown poi_id raises KeyError."""
        mgr = StateMachineManager(_default_config())
        with pytest.raises(KeyError, match="not found"):
            mgr.get_state("NONEXISTENT_ID")

    def test_multiple_pois(self):
        """Two POIs tracked simultaneously with independent states."""
        mgr = StateMachineManager(_default_config())

        # Register a bullish and bearish POI
        bullish_data = _poi_data(1)
        bearish_data = {
            "direction": -1,
            "top": 120.0,
            "bottom": 115.0,
            "midpoint": 117.5,
            "score": 6,
            "timeframe": "15m",
            "components": [],
        }

        id_bull = mgr.register_poi(bullish_data, "4H", _ts(0))
        id_bear = mgr.register_poi(bearish_data, "15m", _ts(0))

        # Candle that taps the bullish POI (low=107 <= 108) but NOT the bearish
        # (high=112 < bottom=115)
        candle = _candle(open=110.0, high=112.0, low=107.0, close=111.0)
        concept = _empty_concept_data()
        mgr.update(candle, 10, _ts(1), concept)

        bull_state = mgr.get_state(id_bull)
        bear_state = mgr.get_state(id_bear)

        # Bullish should have progressed
        assert bull_state.phase == POIPhase.COLLECTING
        assert len(bull_state.confirmations) >= 1

        # Bearish should still be IDLE
        assert bear_state.phase == POIPhase.IDLE
        assert len(bear_state.confirmations) == 0

        # Now a candle that taps bearish (high=116 >= bottom=115) but NOT bullish
        # (low=113 > top=108) -- actually low=113 > 108, this does tap bullish
        # Let's use a candle fully above bullish zone
        candle2 = _candle(open=116.0, high=118.0, low=115.0, close=117.0)
        mgr.update(candle2, 11, _ts(2), concept)

        bear_state2 = mgr.get_state(id_bear)
        assert bear_state2.phase == POIPhase.COLLECTING

        # Bullish continues collecting (candle low=115 > poi_top=108, no new POI_TAP
        # but it's in COLLECTING so it processes)
        bull_state2 = mgr.get_state(id_bull)
        assert bull_state2.phase == POIPhase.COLLECTING

    def test_full_lifecycle_to_ready(self):
        """Integration test: register -> update across multiple bars -> READY."""
        config = ConfirmationsConfig(min_count=3, max_count=8)
        mgr = StateMachineManager(config)

        poi_id = mgr.register_poi(_poi_data(1), "4H", _ts(0))

        # Bar 10: Candle taps POI with rich concept data -> multiple confirmations
        # low=98 sweeps liquidity at 99 AND taps POI (low <= top=108)
        # close=109 >= 99 so sweep is valid
        candle1 = _candle(open=110.0, high=112.0, low=98.0, close=109.0)
        concept1 = _rich_concept_data(bar_index=10, direction=1)
        mgr.update(candle1, 10, _ts(1), concept1)

        state = mgr.get_state(poi_id)
        # Should have collected multiple confirmations on this bar:
        # POI_TAP, LIQUIDITY_SWEEP, FVG_INVERSION, INVERSION_TEST, STRUCTURE_BREAK, CVB_TEST
        # With min_count=3, should be READY
        assert state.phase == POIPhase.READY
        assert len(state.confirmations) >= 3
