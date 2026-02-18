"""Tests for exit decision logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from config import StrategyConfig, BreakevenConfig
from strategy.types import (
    POIPhase, POIState, SignalType, ExitReason, SyncMode,
)
from strategy.exits import (
    check_fta_breakeven,
    check_stop_loss_hit,
    check_structural_breakeven,
    check_target_hit,
    evaluate_exit,
    select_target,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = pd.Timestamp("2024-01-02 12:00", tz="UTC")


def _poi_data(direction: int = 1) -> dict:
    if direction == 1:
        return {"direction": 1, "top": 21100.0, "bottom": 21000.0, "midpoint": 21050.0}
    return {"direction": -1, "top": 21100.0, "bottom": 21000.0, "midpoint": 21050.0}


def _candle(open_: float, high: float, low: float, close: float) -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def _positioned_poi(
    direction: int = 1,
    entry: float = 21120.0,
    sl: float = 20950.0,
    target: float = 21500.0,
    be_level: float | None = None,
) -> POIState:
    return POIState(
        poi_id="poi_exit",
        poi_data=_poi_data(direction),
        phase=POIPhase.POSITIONED,
        entry_price=entry,
        stop_loss=sl,
        target=target,
        breakeven_level=be_level,
    )


def _structure_events(
    direction: int = 1,
    broken_index: int = 300,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "direction": direction,
        "broken_index": broken_index,
        "broken_level": 21200.0,
        "type": "BOS",
    }])


def _empty_structure() -> pd.DataFrame:
    return pd.DataFrame(columns=["direction", "broken_index", "broken_level", "type"])


# ---------------------------------------------------------------------------
# TestCheckTargetHit
# ---------------------------------------------------------------------------

class TestCheckTargetHit:

    def test_long_target_hit(self):
        """LONG: candle_high >= target -> True."""
        assert check_target_hit(21510.0, 21400.0, 21500.0, direction=1) is True

    def test_long_target_not_hit(self):
        """LONG: candle_high < target -> False."""
        assert check_target_hit(21490.0, 21400.0, 21500.0, direction=1) is False

    def test_short_target_hit(self):
        """SHORT: candle_low <= target -> True."""
        assert check_target_hit(20800.0, 20690.0, 20700.0, direction=-1) is True

    def test_short_target_not_hit(self):
        """SHORT: candle_low > target -> False."""
        assert check_target_hit(20800.0, 20710.0, 20700.0, direction=-1) is False


# ---------------------------------------------------------------------------
# TestCheckStopLossHit
# ---------------------------------------------------------------------------

class TestCheckStopLossHit:

    def test_long_sl_hit(self):
        """LONG: candle_low <= stop_loss -> True."""
        assert check_stop_loss_hit(21100.0, 20940.0, 20950.0, direction=1) is True

    def test_long_sl_not_hit(self):
        """LONG: candle_low > stop_loss -> False."""
        assert check_stop_loss_hit(21100.0, 20960.0, 20950.0, direction=1) is False

    def test_short_sl_hit(self):
        """SHORT: candle_high >= stop_loss -> True."""
        assert check_stop_loss_hit(21160.0, 21000.0, 21150.0, direction=-1) is True

    def test_short_sl_not_hit(self):
        """SHORT: candle_high < stop_loss -> False."""
        assert check_stop_loss_hit(21140.0, 21000.0, 21150.0, direction=-1) is False


# ---------------------------------------------------------------------------
# TestCheckStructuralBreakeven
# ---------------------------------------------------------------------------

class TestCheckStructuralBreakeven:

    def test_be_on_structure_break(self):
        """Structure break at bar_index -> returns breakeven level."""
        state = _positioned_poi(direction=1, entry=21120.0)
        events = _structure_events(direction=1, broken_index=300)
        be = check_structural_breakeven(state, events, bar_index=300, config=StrategyConfig())
        assert be is not None
        # BE for long = entry * (1 + 2*0.0006) = 21120 * 1.0012 = ~21145.344
        expected = 21120.0 * (1 + 2 * 0.0006)
        assert abs(be - expected) < 0.01

    def test_no_be_without_config(self):
        """structural_bu=False -> None."""
        state = _positioned_poi(direction=1, entry=21120.0)
        events = _structure_events(direction=1, broken_index=300)
        config = StrategyConfig(breakeven=BreakevenConfig(structural_bu=False))
        be = check_structural_breakeven(state, events, bar_index=300, config=config)
        assert be is None

    def test_no_be_without_entry(self):
        """entry_price is None -> None."""
        state = POIState(
            poi_id="x", poi_data=_poi_data(), phase=POIPhase.POSITIONED,
            entry_price=None, stop_loss=20950.0, target=21500.0,
        )
        events = _structure_events(direction=1, broken_index=300)
        be = check_structural_breakeven(state, events, bar_index=300, config=StrategyConfig())
        assert be is None

    def test_no_be_no_structure(self):
        """Empty structure events -> None."""
        state = _positioned_poi(direction=1, entry=21120.0)
        be = check_structural_breakeven(state, _empty_structure(), bar_index=300, config=StrategyConfig())
        assert be is None


# ---------------------------------------------------------------------------
# TestCheckFtaBreakeven
# ---------------------------------------------------------------------------

class TestCheckFtaBreakeven:

    def test_be_when_past_fta(self):
        """Price past FTA midpoint -> returns BE level."""
        state = _positioned_poi(direction=1, entry=21120.0)
        fta = {"midpoint": 21300.0, "top": 21350.0, "bottom": 21250.0}
        # current_price=21310 >= fta_midpoint=21300
        be = check_fta_breakeven(state, fta, current_price=21310.0, config=StrategyConfig())
        assert be is not None
        expected = 21120.0 * (1 + 2 * 0.0006)
        assert abs(be - expected) < 0.01

    def test_no_be_when_not_past_fta(self):
        """Price not past FTA midpoint -> None."""
        state = _positioned_poi(direction=1, entry=21120.0)
        fta = {"midpoint": 21300.0, "top": 21350.0, "bottom": 21250.0}
        be = check_fta_breakeven(state, fta, current_price=21200.0, config=StrategyConfig())
        assert be is None

    def test_no_be_without_config(self):
        """fta_bu=False -> None."""
        state = _positioned_poi(direction=1, entry=21120.0)
        fta = {"midpoint": 21300.0, "top": 21350.0, "bottom": 21250.0}
        config = StrategyConfig(breakeven=BreakevenConfig(fta_bu=False))
        be = check_fta_breakeven(state, fta, current_price=21310.0, config=config)
        assert be is None

    def test_no_be_without_fta(self):
        """No FTA dict -> None."""
        state = _positioned_poi(direction=1, entry=21120.0)
        be = check_fta_breakeven(state, None, current_price=21310.0, config=StrategyConfig())
        assert be is None


# ---------------------------------------------------------------------------
# TestSelectTarget
# ---------------------------------------------------------------------------

class TestSelectTarget:

    def test_long_target_from_swings(self):
        """Long: nearest swing high above price."""
        swings = pd.DataFrame([
            {"level": 21400.0, "direction": 1},
            {"level": 21600.0, "direction": 1},
            {"level": 20800.0, "direction": -1},
        ])
        pois = pd.DataFrame(columns=["direction", "top", "bottom"])
        target = select_target(1, 21200.0, pois, swings, SyncMode.SYNC, StrategyConfig())
        assert target == 21400.0

    def test_short_target_from_swings(self):
        """Short: nearest swing low below price."""
        swings = pd.DataFrame([
            {"level": 20900.0, "direction": -1},
            {"level": 20600.0, "direction": -1},
            {"level": 21500.0, "direction": 1},
        ])
        pois = pd.DataFrame(columns=["direction", "top", "bottom"])
        target = select_target(-1, 21000.0, pois, swings, SyncMode.SYNC, StrategyConfig())
        assert target == 20900.0

    def test_fallback_to_pois(self):
        """No matching swings -> use opposing POIs."""
        swings = pd.DataFrame(columns=["level", "direction"])
        pois = pd.DataFrame([
            {"direction": -1, "top": 21500.0, "bottom": 21400.0},
        ])
        target = select_target(1, 21200.0, pois, swings, SyncMode.SYNC, StrategyConfig())
        assert target == 21400.0

    def test_fallback_to_percentage(self):
        """No swings and no opposing POIs -> 3% fallback."""
        swings = pd.DataFrame(columns=["level", "direction"])
        pois = pd.DataFrame(columns=["direction", "top", "bottom"])
        target = select_target(1, 21000.0, pois, swings, SyncMode.SYNC, StrategyConfig())
        assert abs(target - 21000.0 * 1.03) < 0.01

        target_short = select_target(-1, 21000.0, pois, swings, SyncMode.SYNC, StrategyConfig())
        assert abs(target_short - 21000.0 * 0.97) < 0.01


# ---------------------------------------------------------------------------
# TestEvaluateExit
# ---------------------------------------------------------------------------

class TestEvaluateExit:

    def test_sl_hit_emits_exit(self):
        """Stop loss hit -> EXIT signal with STOP_LOSS_HIT."""
        state = _positioned_poi(direction=1, entry=21120.0, sl=20950.0, target=21500.0)
        candle = _candle(21050, 21060, 20940, 20960.0)  # low=20940 <= sl=20950
        sig = evaluate_exit(state, candle, 300, TS, None, _empty_structure(), StrategyConfig())
        assert sig is not None
        assert sig.type == SignalType.EXIT
        assert sig.reason == ExitReason.STOP_LOSS_HIT.value

    def test_target_hit_emits_exit(self):
        """Target hit -> EXIT signal with TARGET_HIT."""
        state = _positioned_poi(direction=1, entry=21120.0, sl=20950.0, target=21500.0)
        candle = _candle(21400, 21510, 21380, 21480.0)  # high=21510 >= target=21500
        sig = evaluate_exit(state, candle, 300, TS, None, _empty_structure(), StrategyConfig())
        assert sig is not None
        assert sig.type == SignalType.EXIT
        assert sig.reason == ExitReason.TARGET_HIT.value

    def test_structural_be_emits_modify(self):
        """Structure break -> MOVE_TO_BE signal."""
        state = _positioned_poi(direction=1, entry=21120.0, sl=20950.0, target=21500.0)
        events = _structure_events(direction=1, broken_index=300)
        candle = _candle(21200, 21250, 21180, 21230.0)  # no SL/TP hit
        sig = evaluate_exit(state, candle, 300, TS, None, events, StrategyConfig())
        assert sig is not None
        assert sig.type == SignalType.MOVE_TO_BE
        assert "structural" in sig.reason

    def test_no_exit_when_not_positioned(self):
        """Phase not POSITIONED or MANAGING -> None."""
        state = POIState(
            poi_id="x", poi_data=_poi_data(), phase=POIPhase.READY,
            entry_price=21120.0, stop_loss=20950.0, target=21500.0,
        )
        candle = _candle(21200, 21510, 20940, 21200.0)
        sig = evaluate_exit(state, candle, 300, TS, None, _empty_structure(), StrategyConfig())
        assert sig is None

    def test_sl_checked_before_target(self):
        """When both SL and target hit on the same bar, SL takes priority."""
        state = _positioned_poi(direction=1, entry=21120.0, sl=20950.0, target=21500.0)
        # Huge bar: low hits SL, high hits target
        candle = _candle(21200, 21510, 20940, 21300.0)
        sig = evaluate_exit(state, candle, 300, TS, None, _empty_structure(), StrategyConfig())
        assert sig is not None
        assert sig.type == SignalType.EXIT
        assert sig.reason == ExitReason.STOP_LOSS_HIT.value
