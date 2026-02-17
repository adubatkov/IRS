"""Tests for strategy type definitions."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.types import (
    Bias,
    Confirmation,
    ConfirmationType,
    ExitReason,
    POIPhase,
    POIState,
    Signal,
    SignalType,
    SyncMode,
)


class TestEnums:
    def test_bias_values(self):
        assert Bias.BULLISH == "BULLISH"
        assert Bias.BEARISH == "BEARISH"
        assert Bias.UNDEFINED == "UNDEFINED"

    def test_sync_mode_values(self):
        assert SyncMode.SYNC == "SYNC"
        assert SyncMode.DESYNC == "DESYNC"

    def test_poi_phase_values(self):
        phases = [p.value for p in POIPhase]
        assert "IDLE" in phases
        assert "COLLECTING" in phases
        assert "READY" in phases
        assert "POSITIONED" in phases
        assert "CLOSED" in phases

    def test_confirmation_types(self):
        assert len(ConfirmationType) == 8
        assert ConfirmationType.POI_TAP == "POI_TAP"
        assert ConfirmationType.LIQUIDITY_SWEEP == "LIQUIDITY_SWEEP"
        assert ConfirmationType.FVG_INVERSION == "FVG_INVERSION"
        assert ConfirmationType.INVERSION_TEST == "INVERSION_TEST"
        assert ConfirmationType.STRUCTURE_BREAK == "STRUCTURE_BREAK"
        assert ConfirmationType.FVG_WICK_REACTION == "FVG_WICK_REACTION"
        assert ConfirmationType.CVB_TEST == "CVB_TEST"
        assert ConfirmationType.ADDITIONAL_CBOS == "ADDITIONAL_CBOS"

    def test_signal_types(self):
        assert SignalType.ENTER == "ENTER"
        assert SignalType.EXIT == "EXIT"
        assert SignalType.MODIFY_SL == "MODIFY_SL"
        assert SignalType.ADD_ON == "ADD_ON"
        assert SignalType.MOVE_TO_BE == "MOVE_TO_BE"

    def test_exit_reasons(self):
        assert ExitReason.TARGET_HIT == "TARGET_HIT"
        assert ExitReason.STOP_LOSS_HIT == "STOP_LOSS_HIT"
        assert ExitReason.FLIP == "FLIP"

    def test_enums_are_str(self):
        assert isinstance(Bias.BULLISH, str)
        assert isinstance(POIPhase.IDLE, str)
        assert isinstance(ConfirmationType.POI_TAP, str)


class TestConfirmation:
    def test_defaults(self):
        ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")
        c = Confirmation(
            type=ConfirmationType.POI_TAP,
            timestamp=ts,
            bar_index=42,
        )
        assert c.type == ConfirmationType.POI_TAP
        assert c.bar_index == 42
        assert c.details == {}

    def test_with_details(self):
        ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")
        c = Confirmation(
            type=ConfirmationType.LIQUIDITY_SWEEP,
            timestamp=ts,
            bar_index=50,
            details={"level": 21000.0, "direction": -1},
        )
        assert c.details["level"] == 21000.0


class TestSignal:
    def test_defaults(self):
        ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")
        s = Signal(
            type=SignalType.ENTER,
            poi_id="4H_1_100",
            direction=1,
            timestamp=ts,
            price=21000.0,
        )
        assert s.stop_loss == 0.0
        assert s.target == 0.0
        assert s.position_size_mult == 1.0
        assert s.reason == ""
        assert s.metadata == {}

    def test_full_signal(self):
        ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")
        s = Signal(
            type=SignalType.ENTER,
            poi_id="1H_-1_200",
            direction=-1,
            timestamp=ts,
            price=21500.0,
            stop_loss=21600.0,
            target=21200.0,
            position_size_mult=0.5,
            reason="conservative entry",
        )
        assert s.direction == -1
        assert s.position_size_mult == 0.5


class TestPOIState:
    def test_defaults(self):
        state = POIState(
            poi_id="15m_1_50",
            poi_data={"direction": 1, "top": 21100.0, "bottom": 21000.0},
        )
        assert state.phase == POIPhase.IDLE
        assert state.confirmations == []
        assert state.entry_price is None
        assert state.stop_loss is None
        assert state.target is None
        assert state.addons == []

    def test_phase_update(self):
        state = POIState(
            poi_id="4H_1_100",
            poi_data={"direction": 1, "top": 21100.0, "bottom": 21000.0},
        )
        state.phase = POIPhase.POI_TAPPED
        assert state.phase == POIPhase.POI_TAPPED
        state.phase = POIPhase.COLLECTING
        assert state.phase == POIPhase.COLLECTING

    def test_add_confirmation(self):
        ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")
        state = POIState(
            poi_id="4H_1_100",
            poi_data={"direction": 1},
        )
        state.confirmations.append(
            Confirmation(type=ConfirmationType.POI_TAP, timestamp=ts, bar_index=100)
        )
        assert len(state.confirmations) == 1
        assert state.confirmations[0].type == ConfirmationType.POI_TAP
