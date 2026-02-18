"""Unit tests for engine.portfolio module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
import pandas as pd

from config import BacktestConfig, RiskConfig
from strategy.types import Signal, SignalType, SyncMode
from engine.portfolio import Portfolio, apply_slippage, PositionInfo
from engine.trade_log import TradeLog
from engine.events import EventLog

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BT_CONFIG = BacktestConfig(
    initial_capital=10000,
    commission_pct=0.0006,
    slippage_pct=0.0002,
)
RISK_CONFIG = RiskConfig(
    max_concurrent_positions=3,
    max_risk_per_trade=0.02,
    position_size_sync=1.0,
    position_size_desync=0.5,
)

TIMESTAMP = pd.Timestamp("2024-01-02 10:00", tz="UTC")
EXIT_TIMESTAMP = pd.Timestamp("2024-01-02 14:00", tz="UTC")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_signal(
    poi_id: str = "test_1",
    direction: int = 1,
    price: float = 21000.0,
    sl: float = 20900.0,
    target: float = 21300.0,
) -> Signal:
    return Signal(
        type=SignalType.ENTER,
        poi_id=poi_id,
        direction=direction,
        timestamp=TIMESTAMP,
        price=price,
        stop_loss=sl,
        target=target,
    )


def _make_portfolio(n_bars: int = 100) -> tuple[Portfolio, TradeLog, EventLog]:
    """Create a Portfolio with fresh TradeLog and EventLog."""
    trade_log = TradeLog()
    event_log = EventLog()
    portfolio = Portfolio(
        backtest_config=BT_CONFIG,
        risk_config=RISK_CONFIG,
        n_bars=n_bars,
        trade_log=trade_log,
        event_log=event_log,
    )
    return portfolio, trade_log, event_log


# ---------------------------------------------------------------------------
# 1. test_apply_slippage_long_entry
# ---------------------------------------------------------------------------
class TestApplySlippageLongEntry:
    def test_apply_slippage_long_entry(self):
        """LONG entry: price * (1 + slippage)."""
        result = apply_slippage(21000.0, direction=1, is_entry=True, slippage_pct=0.0002)
        expected = 21000.0 * (1 + 0.0002)  # 21004.2
        assert result == pytest.approx(expected)
        assert result > 21000.0  # Slippage works against trader (buy higher)


# ---------------------------------------------------------------------------
# 2. test_apply_slippage_short_entry
# ---------------------------------------------------------------------------
class TestApplySlippageShortEntry:
    def test_apply_slippage_short_entry(self):
        """SHORT entry: price * (1 - slippage)."""
        result = apply_slippage(21000.0, direction=-1, is_entry=True, slippage_pct=0.0002)
        expected = 21000.0 * (1 - 0.0002)  # 20995.8
        assert result == pytest.approx(expected)
        assert result < 21000.0  # Slippage works against trader (sell lower)


# ---------------------------------------------------------------------------
# 3. test_apply_slippage_long_exit
# ---------------------------------------------------------------------------
class TestApplySlippageLongExit:
    def test_apply_slippage_long_exit(self):
        """LONG exit: price * (1 - slippage)."""
        result = apply_slippage(21300.0, direction=1, is_entry=False, slippage_pct=0.0002)
        expected = 21300.0 * (1 - 0.0002)  # 21295.74
        assert result == pytest.approx(expected)
        assert result < 21300.0  # Slippage works against trader (sell lower)


# ---------------------------------------------------------------------------
# 4. test_apply_slippage_short_exit
# ---------------------------------------------------------------------------
class TestApplySlippageShortExit:
    def test_apply_slippage_short_exit(self):
        """SHORT exit: price * (1 + slippage)."""
        result = apply_slippage(20700.0, direction=-1, is_entry=False, slippage_pct=0.0002)
        expected = 20700.0 * (1 + 0.0002)  # 20704.14
        assert result == pytest.approx(expected)
        assert result > 20700.0  # Slippage works against trader (buy higher)


# ---------------------------------------------------------------------------
# 5. test_open_position_long
# ---------------------------------------------------------------------------
class TestOpenPositionLong:
    def test_open_position_long(self):
        """Open a long, verify cash reduced by commission, trade_id returned."""
        portfolio, trade_log, event_log = _make_portfolio()

        signal = _make_signal(poi_id="poi_1", direction=1, price=21000.0, sl=20900.0)
        trade_id = portfolio.open_position(signal, SyncMode.SYNC, bar_index=0)

        assert trade_id is not None
        assert trade_id == 0

        # Verify fill price with slippage
        fill_price = 21000.0 * (1 + 0.0002)  # 21004.2

        # Position size: risk_amount / distance * sync_mult
        # equity = 10000 (no unrealized yet, _last_close=0 but no positions when calc'd)
        # Actually equity is computed before the position is added.
        # At entry time, equity = cash + unrealized. Cash = 10000 (commission not yet deducted
        # when calculate_position_size is called). But wait -- commission is deducted AFTER
        # size is calculated. So equity at time of sizing = 10000 + 0 = 10000.
        risk_amount = 10000.0 * 0.02  # 200
        distance = abs(fill_price - 20900.0)  # 104.2
        expected_size = (risk_amount / distance) * 1.0  # ~1.91939...

        # Commission on entry
        expected_commission = fill_price * expected_size * 0.0006

        # Cash should be reduced by commission only (position is tracked, not deducted from cash)
        assert portfolio.cash == pytest.approx(10000.0 - expected_commission)

        # Verify TradeLog has the trade
        record = trade_log.get_trade(trade_id)
        assert record.poi_id == "poi_1"
        assert record.direction == 1
        assert record.entry_price == pytest.approx(fill_price)
        assert record.position_size == pytest.approx(expected_size)
        assert record.commission_entry == pytest.approx(expected_commission)

        # Verify position tracking
        assert portfolio.has_position_for_poi("poi_1")
        assert portfolio.open_position_count == 1

        # Verify event emitted
        entry_events = event_log.get_events(event_type=None)
        assert len(entry_events) == 1
        assert entry_events[0].type.value == "ENTRY"


# ---------------------------------------------------------------------------
# 6. test_close_position_profit
# ---------------------------------------------------------------------------
class TestClosePositionProfit:
    def test_close_position_profit(self):
        """Open and close at profit, verify cash increased."""
        portfolio, trade_log, event_log = _make_portfolio()

        signal = _make_signal(poi_id="poi_1", direction=1, price=21000.0, sl=20900.0, target=21300.0)
        trade_id = portfolio.open_position(signal, SyncMode.SYNC, bar_index=0)
        assert trade_id is not None

        cash_after_entry = portfolio.cash

        # Close at target
        records = portfolio.close_position(
            poi_id="poi_1",
            exit_signal_price=21300.0,
            exit_reason="target_hit",
            timestamp=EXIT_TIMESTAMP,
            bar_index=10,
        )

        assert len(records) == 1
        record = records[0]

        # Exit fill: 21300 * (1 - 0.0002) = 21295.74
        exit_fill = 21300.0 * (1 - 0.0002)
        entry_fill = 21000.0 * (1 + 0.0002)

        # Gross proceeds = (exit_fill - entry_fill) * size (direction=1)
        size = record.position_size
        gross_proceeds = (exit_fill - entry_fill) * size

        # Proceeds should be positive (profit)
        assert gross_proceeds > 0

        # Cash should have increased relative to after entry
        assert portfolio.cash > cash_after_entry
        assert record.outcome == "WIN"
        assert record.exit_reason == "target_hit"
        assert record.duration_bars == 10

        # Position should be closed
        assert not portfolio.has_position_for_poi("poi_1")
        assert portfolio.open_position_count == 0


# ---------------------------------------------------------------------------
# 7. test_close_position_loss
# ---------------------------------------------------------------------------
class TestClosePositionLoss:
    def test_close_position_loss(self):
        """Open and close at loss, verify cash decreased."""
        portfolio, trade_log, event_log = _make_portfolio()

        signal = _make_signal(poi_id="poi_1", direction=1, price=21000.0, sl=20900.0, target=21300.0)
        trade_id = portfolio.open_position(signal, SyncMode.SYNC, bar_index=0)
        assert trade_id is not None

        cash_after_entry = portfolio.cash

        # Close at stop loss
        records = portfolio.close_position(
            poi_id="poi_1",
            exit_signal_price=20900.0,
            exit_reason="stop_loss_hit",
            timestamp=EXIT_TIMESTAMP,
            bar_index=5,
        )

        assert len(records) == 1
        record = records[0]

        # Exit fill: 20900 * (1 - 0.0002) = 20895.82 (long exit slippage works against)
        exit_fill = 20900.0 * (1 - 0.0002)
        entry_fill = 21000.0 * (1 + 0.0002)

        # Gross proceeds = (exit_fill - entry_fill) * size is negative
        size = record.position_size
        gross_proceeds = (exit_fill - entry_fill) * size
        assert gross_proceeds < 0

        # Cash should have decreased relative to after entry
        assert portfolio.cash < cash_after_entry
        assert record.outcome == "LOSS"
        assert record.exit_reason == "stop_loss_hit"

        # Position should be closed
        assert not portfolio.has_position_for_poi("poi_1")


# ---------------------------------------------------------------------------
# 8. test_max_positions_enforcement
# ---------------------------------------------------------------------------
class TestMaxPositionsEnforcement:
    def test_max_positions_enforcement(self):
        """Open 3 positions (different poi_ids), 4th rejected."""
        portfolio, trade_log, event_log = _make_portfolio()

        # Open 3 positions with different poi_ids
        for i in range(1, 4):
            signal = _make_signal(
                poi_id=f"poi_{i}",
                direction=1,
                price=21000.0,
                sl=20900.0,
                target=21300.0,
            )
            tid = portfolio.open_position(signal, SyncMode.SYNC, bar_index=i)
            assert tid is not None, f"Position poi_{i} should have opened"

        assert portfolio.open_position_count == 3

        # 4th position with a new poi_id should be rejected
        signal_4 = _make_signal(
            poi_id="poi_4",
            direction=1,
            price=21000.0,
            sl=20900.0,
            target=21300.0,
        )
        tid_4 = portfolio.open_position(signal_4, SyncMode.SYNC, bar_index=4)
        assert tid_4 is None

        # Still 3 open positions
        assert portfolio.open_position_count == 3

        # Verify POSITION_REJECTED event was emitted
        rejected_events = event_log.get_events(event_type=None)
        # 3 ENTRY events + 1 POSITION_REJECTED
        rejected = [e for e in rejected_events if e.type.value == "POSITION_REJECTED"]
        assert len(rejected) == 1
        assert rejected[0].details["reason"] == "max_positions_reached"


# ---------------------------------------------------------------------------
# 9. test_addon_bypasses_max_positions
# ---------------------------------------------------------------------------
class TestAddonBypassesMaxPositions:
    def test_addon_bypasses_max_positions(self):
        """At max positions, add-on to existing poi_id allowed."""
        portfolio, trade_log, event_log = _make_portfolio()

        # Fill up to max positions
        trade_ids = []
        for i in range(1, 4):
            signal = _make_signal(
                poi_id=f"poi_{i}",
                direction=1,
                price=21000.0,
                sl=20900.0,
                target=21300.0,
            )
            tid = portfolio.open_position(signal, SyncMode.SYNC, bar_index=i)
            assert tid is not None
            trade_ids.append(tid)

        assert portfolio.open_position_count == 3

        # Add-on to poi_1 (existing poi_id) should succeed despite max positions
        addon_signal = _make_signal(
            poi_id="poi_1",
            direction=1,
            price=21050.0,
            sl=20900.0,
            target=21300.0,
        )
        addon_tid = portfolio.open_position(
            addon_signal,
            SyncMode.SYNC,
            bar_index=5,
            is_addon=True,
            parent_trade_id=trade_ids[0],
        )

        assert addon_tid is not None
        # Still 3 distinct poi_ids (addon goes under poi_1)
        assert portfolio.open_position_count == 3

        # poi_1 should now have 2 positions
        positions = portfolio.get_positions_for_poi("poi_1")
        assert len(positions) == 2
        assert positions[1].is_addon is True
        assert positions[1].parent_trade_id == trade_ids[0]

        # Verify ADDON event emitted
        addon_events = [
            e for e in event_log.get_events()
            if e.type.value == "ADDON"
        ]
        assert len(addon_events) == 1


# ---------------------------------------------------------------------------
# 10. test_equity_curve_tracking
# ---------------------------------------------------------------------------
class TestEquityCurveTracking:
    def test_equity_curve_tracking(self):
        """Open position, update_mark_to_market, verify equity curve values."""
        portfolio, trade_log, event_log = _make_portfolio(n_bars=10)

        # Record initial equity at bar 0 (no positions)
        portfolio.update_mark_to_market(
            bar_index=0, candle_high=21050.0, candle_low=20950.0, candle_close=21000.0,
        )
        curve = portfolio.get_equity_curve()
        assert curve[0] == pytest.approx(10000.0)  # No positions, just initial cash

        # Open long position
        signal = _make_signal(poi_id="poi_1", direction=1, price=21000.0, sl=20900.0)
        trade_id = portfolio.open_position(signal, SyncMode.SYNC, bar_index=1)
        assert trade_id is not None

        cash_after_entry = portfolio.cash
        entry_fill = 21000.0 * (1 + 0.0002)  # 21004.2
        record = trade_log.get_trade(trade_id)
        size = record.position_size

        # Update at bar 1: close at 21100 (price moved up)
        portfolio.update_mark_to_market(
            bar_index=1, candle_high=21150.0, candle_low=20980.0, candle_close=21100.0,
        )

        # Expected equity: cash + unrealized PnL
        unrealized = 1 * (21100.0 - entry_fill) * size
        expected_equity = cash_after_entry + unrealized
        assert curve[1] == pytest.approx(expected_equity)
        assert curve[1] > cash_after_entry  # Price went up, long position gains

        # Update at bar 2: close at 20950 (price dropped below entry)
        portfolio.update_mark_to_market(
            bar_index=2, candle_high=21000.0, candle_low=20900.0, candle_close=20950.0,
        )
        unrealized_2 = 1 * (20950.0 - entry_fill) * size
        expected_equity_2 = cash_after_entry + unrealized_2
        assert curve[2] == pytest.approx(expected_equity_2)
        assert curve[2] < cash_after_entry  # Price dropped, long position loses

        # Bars 3-9 should still be NaN
        assert np.isnan(curve[3])


# ---------------------------------------------------------------------------
# 11. test_zero_size_rejected
# ---------------------------------------------------------------------------
class TestZeroSizeRejected:
    def test_zero_size_rejected(self):
        """SyncMode.UNDEFINED gives 0 multiplier, verify rejection."""
        portfolio, trade_log, event_log = _make_portfolio()

        signal = _make_signal(poi_id="poi_1", direction=1, price=21000.0, sl=20900.0)
        trade_id = portfolio.open_position(signal, SyncMode.UNDEFINED, bar_index=0)

        # Should be rejected (size = 0 due to UNDEFINED sync mode)
        assert trade_id is None

        # Cash unchanged (no commission deducted)
        assert portfolio.cash == pytest.approx(10000.0)

        # No positions tracked
        assert portfolio.open_position_count == 0
        assert not portfolio.has_position_for_poi("poi_1")

        # Verify POSITION_REJECTED event
        rejected = event_log.get_events(event_type=None)
        assert len(rejected) == 1
        assert rejected[0].type.value == "POSITION_REJECTED"
        assert rejected[0].details["reason"] == "zero_position_size"


# ---------------------------------------------------------------------------
# 12. test_modify_stop_loss
# ---------------------------------------------------------------------------
class TestModifyStopLoss:
    def test_modify_stop_loss(self):
        """Modify SL, verify all positions for poi_id updated."""
        portfolio, trade_log, event_log = _make_portfolio()

        # Open two positions for the same poi_id (one regular, one addon)
        signal_1 = _make_signal(poi_id="poi_1", direction=1, price=21000.0, sl=20900.0)
        tid_1 = portfolio.open_position(signal_1, SyncMode.SYNC, bar_index=0)
        assert tid_1 is not None

        signal_2 = _make_signal(poi_id="poi_1", direction=1, price=21050.0, sl=20900.0)
        tid_2 = portfolio.open_position(
            signal_2, SyncMode.SYNC, bar_index=1, is_addon=True, parent_trade_id=tid_1,
        )
        assert tid_2 is not None

        # Both should have original SL
        positions = portfolio.get_positions_for_poi("poi_1")
        assert len(positions) == 2
        for pos in positions:
            assert pos.stop_loss == pytest.approx(20900.0)

        # Modify stop loss to breakeven level
        new_sl = 21004.2  # Near entry fill price
        portfolio.modify_stop_loss("poi_1", new_sl)

        # All positions for poi_1 should have updated SL
        positions = portfolio.get_positions_for_poi("poi_1")
        for pos in positions:
            assert pos.stop_loss == pytest.approx(new_sl)

        # Modifying a non-existent poi_id should not raise
        portfolio.modify_stop_loss("poi_nonexistent", 21000.0)
