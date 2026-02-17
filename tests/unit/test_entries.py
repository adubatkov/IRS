"""Tests for entry decision logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import patch

import pandas as pd

from config import StrategyConfig, EntryConfig
from strategy.types import (
    Confirmation, ConfirmationType, POIPhase, POIState,
    Signal, SignalType, SyncMode,
)
from strategy.entries import (
    check_aggressive_entry,
    check_conservative_entry,
    check_rto_entry,
    evaluate_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = pd.Timestamp("2024-01-02 10:00", tz="UTC")


def _poi_data(direction: int = 1) -> dict:
    """Standard POI data for a demand (1) or supply (-1) zone."""
    if direction == 1:
        return {"direction": 1, "top": 21100.0, "bottom": 21000.0, "midpoint": 21050.0}
    return {"direction": -1, "top": 21100.0, "bottom": 21000.0, "midpoint": 21050.0}


def _candle(open_: float, high: float, low: float, close: float) -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def _make_confirms(n: int = 5, include_fvg: bool = False) -> list[Confirmation]:
    """Build a list of n confirmations.

    If include_fvg is False, produces the 5th-confirm-trap pattern
    (no FVG-related confirms, last one is STRUCTURE_BREAK).
    """
    pool_no_fvg = [
        ConfirmationType.POI_TAP,
        ConfirmationType.LIQUIDITY_SWEEP,
        ConfirmationType.CVB_TEST,
        ConfirmationType.ADDITIONAL_CBOS,
        ConfirmationType.STRUCTURE_BREAK,  # last
    ]
    pool_with_fvg = [
        ConfirmationType.POI_TAP,
        ConfirmationType.LIQUIDITY_SWEEP,
        ConfirmationType.STRUCTURE_BREAK,
        ConfirmationType.CVB_TEST,
        ConfirmationType.FVG_INVERSION,
    ]
    pool = pool_with_fvg if include_fvg else pool_no_fvg
    confirms = []
    for i in range(n):
        ctype = pool[i % len(pool)]
        confirms.append(
            Confirmation(type=ctype, timestamp=TS, bar_index=100 + i)
        )
    return confirms


def _ready_poi(direction: int = 1, confirms: list[Confirmation] | None = None) -> POIState:
    """POIState in READY phase with confirmations."""
    return POIState(
        poi_id="poi_1",
        poi_data=_poi_data(direction),
        phase=POIPhase.READY,
        confirmations=confirms or _make_confirms(5, include_fvg=True),
    )


def _fvg_df(direction: int = 1, top: float = 21080.0, bottom: float = 21020.0,
             status: str = "FRESH") -> pd.DataFrame:
    return pd.DataFrame([{
        "direction": direction,
        "top": top,
        "bottom": bottom,
        "midpoint": (top + bottom) / 2,
        "status": status,
    }])


def _liq_df(direction: int = -1, level: float = 20950.0, status: str = "ACTIVE") -> pd.DataFrame:
    return pd.DataFrame([{
        "direction": direction,
        "level": level,
        "status": status,
    }])


def _empty_fvg() -> pd.DataFrame:
    return pd.DataFrame(columns=["direction", "top", "bottom", "midpoint", "status"])


def _empty_liq() -> pd.DataFrame:
    return pd.DataFrame(columns=["direction", "level", "status"])


# ---------------------------------------------------------------------------
# TestEvaluateEntry
# ---------------------------------------------------------------------------

class TestEvaluateEntry:
    """Tests for the main evaluate_entry function."""

    def test_only_fires_when_ready(self):
        """Non-READY phase returns None."""
        for phase in (POIPhase.IDLE, POIPhase.COLLECTING, POIPhase.POSITIONED, POIPhase.CLOSED):
            state = POIState(poi_id="x", poi_data=_poi_data(), phase=phase)
            result = evaluate_entry(
                state, _candle(21050, 21150, 21000, 21120),
                200, TS, None, "far", SyncMode.SYNC,
                _fvg_df(), _liq_df(), StrategyConfig(),
            )
            assert result is None, f"Expected None for phase {phase}"

    def test_fta_close_blocks(self):
        """FTA close classification blocks entry."""
        state = _ready_poi()
        fta = {"direction": -1, "top": 21200, "bottom": 21150, "midpoint": 21175, "score": 5}
        result = evaluate_entry(
            state, _candle(21050, 21150, 21000, 21120),
            200, TS, fta, "close", SyncMode.SYNC,
            _fvg_df(), _liq_df(), StrategyConfig(),
        )
        assert result is None

    @patch("strategy.entries._build_entry_signal")
    def test_conservative_entry_on_exit(self, mock_build):
        """Price exits POI zone in conservative mode -> signal built."""
        mock_build.return_value = Signal(
            type=SignalType.ENTER, poi_id="poi_1", direction=1,
            timestamp=TS, price=21120.0,
        )
        state = _ready_poi(direction=1)
        # close=21120 > poi_top=21100 -> conservative passes
        candle = _candle(21050, 21150, 21000, 21120.0)
        result = evaluate_entry(
            state, candle, 200, TS, None, "far", SyncMode.SYNC,
            _fvg_df(), _liq_df(), StrategyConfig(),
        )
        assert result is not None
        assert result.type == SignalType.ENTER
        mock_build.assert_called_once()

    @patch("strategy.entries._build_entry_signal")
    def test_aggressive_entry_immediate(self, mock_build):
        """Aggressive mode enters immediately when READY."""
        mock_build.return_value = Signal(
            type=SignalType.ENTER, poi_id="poi_1", direction=1,
            timestamp=TS, price=21050.0,
        )
        state = _ready_poi(direction=1)
        config = StrategyConfig(entry=EntryConfig(mode="aggressive"))
        # close=21050 is inside POI -- conservative would fail, aggressive should pass
        candle = _candle(21040, 21080, 21030, 21050.0)
        result = evaluate_entry(
            state, candle, 200, TS, None, "far", SyncMode.SYNC,
            _fvg_df(), _liq_df(), config,
        )
        assert result is not None
        mock_build.assert_called_once()

    def test_fifth_confirm_trap_waits(self):
        """5th-confirm trap with no active FVG -> None (waits for RTO)."""
        # Build the trap pattern: 5 confirms, no FVG-related, last is STRUCTURE_BREAK
        confirms = _make_confirms(5, include_fvg=False)
        state = _ready_poi(direction=1, confirms=confirms)
        # No FVGs for RTO
        candle = _candle(21050, 21150, 21040, 21120.0)
        result = evaluate_entry(
            state, candle, 200, TS, None, "far", SyncMode.SYNC,
            _empty_fvg(), _liq_df(), StrategyConfig(),
        )
        assert result is None

    @patch("strategy.entries._build_entry_signal")
    def test_rto_entry_after_trap(self, mock_build):
        """RTO triggers after 5th-confirm trap when candle touches FVG."""
        mock_build.return_value = Signal(
            type=SignalType.ENTER, poi_id="poi_1", direction=1,
            timestamp=TS, price=21060.0,
        )
        confirms = _make_confirms(5, include_fvg=False)
        state = _ready_poi(direction=1, confirms=confirms)
        # Candle low=21070 touches FVG top=21080
        candle = _candle(21090, 21100, 21070, 21060.0)
        fvgs = _fvg_df(direction=1, top=21080.0, bottom=21020.0, status="FRESH")
        result = evaluate_entry(
            state, candle, 200, TS, None, "far", SyncMode.SYNC,
            fvgs, _liq_df(), StrategyConfig(),
        )
        assert result is not None
        mock_build.assert_called_once()

    def test_returns_none_for_bad_rr(self):
        """If risk validation fails -> None. See mock-based test below."""
        # With 3x target multiplier, RR is always 3.0 by construction.
        # So we rely on the mock test below to verify this path.
        pass

    def test_returns_none_for_bad_rr_via_mock(self):
        """Validate that _build_entry_signal returns None when risk is invalid."""
        state = _ready_poi(direction=1)
        candle = _candle(21050, 21150, 21000, 21120.0)
        with patch("strategy.entries.validate_risk", return_value=(False, 1.5)):
            result = evaluate_entry(
                state, candle, 200, TS, None, "far", SyncMode.SYNC,
                _fvg_df(), _liq_df(), StrategyConfig(),
            )
        assert result is None


# ---------------------------------------------------------------------------
# TestCheckConservativeEntry
# ---------------------------------------------------------------------------

class TestCheckConservativeEntry:
    """Tests for conservative entry logic."""

    def test_long_exits_above_poi(self):
        """LONG: close above poi_top -> True."""
        state = _ready_poi(direction=1)
        candle = _candle(21050, 21150, 21000, 21120.0)  # close=21120 > top=21100
        assert check_conservative_entry(state, candle, StrategyConfig())

    def test_long_stays_in_poi(self):
        """LONG: close still within POI -> False."""
        state = _ready_poi(direction=1)
        candle = _candle(21050, 21090, 21000, 21080.0)  # close=21080 < top=21100
        assert not check_conservative_entry(state, candle, StrategyConfig())

    def test_short_exits_below_poi(self):
        """SHORT: close below poi_bottom -> True."""
        state = _ready_poi(direction=-1)
        candle = _candle(21050, 21100, 20950, 20980.0)  # close=20980 < bottom=21000
        assert check_conservative_entry(state, candle, StrategyConfig())

    def test_short_stays_in_poi(self):
        """SHORT: close still within POI -> False."""
        state = _ready_poi(direction=-1)
        candle = _candle(21050, 21100, 21010, 21020.0)  # close=21020 > bottom=21000
        assert not check_conservative_entry(state, candle, StrategyConfig())


# ---------------------------------------------------------------------------
# TestCheckAggressiveEntry
# ---------------------------------------------------------------------------

class TestCheckAggressiveEntry:
    """Tests for aggressive entry logic."""

    def test_always_true(self):
        """Aggressive entry always returns True."""
        state = _ready_poi(direction=1)
        candle = _candle(21050, 21060, 21040, 21055.0)
        assert check_aggressive_entry(state, candle, StrategyConfig()) is True

        state_short = _ready_poi(direction=-1)
        assert check_aggressive_entry(state_short, candle, StrategyConfig()) is True


# ---------------------------------------------------------------------------
# TestCheckRtoEntry
# ---------------------------------------------------------------------------

class TestCheckRtoEntry:
    """Tests for RTO (Return to Origin) entry logic."""

    def test_rto_touches_fvg(self):
        """Candle dips to bullish FVG -> True."""
        state = _ready_poi(direction=1)
        fvgs = _fvg_df(direction=1, top=21080.0, bottom=21020.0, status="FRESH")
        # candle low=21070 <= fvg top=21080
        candle = _candle(21090, 21100, 21070, 21085.0)
        assert check_rto_entry(state, candle, fvgs) is True

    def test_rto_no_fvg(self):
        """No FVGs -> False."""
        state = _ready_poi(direction=1)
        candle = _candle(21090, 21100, 21070, 21085.0)
        assert check_rto_entry(state, candle, _empty_fvg()) is False
        assert check_rto_entry(state, candle, None) is False

    def test_rto_wrong_direction_fvg(self):
        """FVG in wrong direction -> False."""
        state = _ready_poi(direction=1)
        # Bearish FVG, but POI is bullish
        fvgs = _fvg_df(direction=-1, top=21080.0, bottom=21020.0, status="FRESH")
        candle = _candle(21090, 21100, 21070, 21085.0)
        assert check_rto_entry(state, candle, fvgs) is False
