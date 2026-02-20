"""Unit tests for engine.trade_log module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd

from engine.trade_log import TradeLog, classify_outcome, compute_r_multiple

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
ENTRY_PRICE = 21000.0
SIGNAL_PRICE = 21000.0
STOP_LOSS = 20900.0
TARGET = 21300.0
POSITION_SIZE = 1.0
COMMISSION = ENTRY_PRICE * POSITION_SIZE * 0.0006  # 12.6

ENTRY_TIME = pd.Timestamp("2024-01-01 10:00", tz="UTC")
EXIT_TIME = pd.Timestamp("2024-01-01 14:00", tz="UTC")


def _make_log_with_open_trade(direction: int = 1) -> tuple[TradeLog, int]:
    """Helper: create a TradeLog and open one trade, returning (log, trade_id)."""
    log = TradeLog()
    tid = log.open_trade(
        poi_id="OB_15m_001",
        direction=direction,
        entry_time=ENTRY_TIME,
        entry_price=ENTRY_PRICE,
        entry_signal_price=SIGNAL_PRICE,
        position_size=POSITION_SIZE,
        stop_loss=STOP_LOSS,
        target=TARGET,
        commission=COMMISSION,
        sync_mode="aligned",
        timeframe="15m",
        confirmation_count=2,
    )
    return log, tid


# ---------------------------------------------------------------------------
# 1. test_open_trade
# ---------------------------------------------------------------------------
class TestOpenTrade:
    def test_open_trade(self):
        """Open trade, verify fields populated and trade_id returned."""
        log, tid = _make_log_with_open_trade()

        assert tid == 0
        record = log.get_trade(tid)

        assert record.trade_id == 0
        assert record.poi_id == "OB_15m_001"
        assert record.direction == 1
        assert record.entry_time == ENTRY_TIME
        assert record.entry_price == ENTRY_PRICE
        assert record.entry_signal_price == SIGNAL_PRICE
        assert record.position_size == POSITION_SIZE
        assert record.stop_loss == STOP_LOSS
        assert record.target == TARGET
        assert record.commission_entry == COMMISSION
        assert record.sync_mode == "aligned"
        assert record.timeframe == "15m"
        assert record.confirmation_count == 2
        # Exit fields should be unset
        assert record.exit_time is None
        assert record.exit_price is None
        assert record.outcome == ""


# ---------------------------------------------------------------------------
# 2. test_close_trade_win
# ---------------------------------------------------------------------------
class TestCloseTradeWin:
    def test_close_trade_win(self):
        """Open then close at profit, verify P&L > 0 and outcome = WIN."""
        log, tid = _make_log_with_open_trade(direction=1)

        exit_price = 21200.0  # +200 from entry
        record = log.close_trade(
            trade_id=tid,
            exit_time=EXIT_TIME,
            exit_price=exit_price,
            exit_signal_price=exit_price,
            exit_reason="target_hit",
            commission=COMMISSION,
            bar_count=16,
        )

        # gross_pnl = 1 * (21200 - 21000) * 1.0 = 200.0
        assert record.gross_pnl == pytest.approx(200.0)
        # realized = 200.0 - (12.6 + 12.6) = 174.8
        assert record.realized_pnl == pytest.approx(200.0 - 2 * COMMISSION)
        assert record.realized_pnl > 0
        assert record.outcome == "WIN"
        assert record.exit_time == EXIT_TIME
        assert record.exit_reason == "target_hit"
        assert record.duration_bars == 16


# ---------------------------------------------------------------------------
# 3. test_close_trade_loss
# ---------------------------------------------------------------------------
class TestCloseTradeLoss:
    def test_close_trade_loss(self):
        """Open then close at loss, verify P&L < 0 and outcome = LOSS."""
        log, tid = _make_log_with_open_trade(direction=1)

        exit_price = 20850.0  # -150 from entry (below SL)
        record = log.close_trade(
            trade_id=tid,
            exit_time=EXIT_TIME,
            exit_price=exit_price,
            exit_signal_price=exit_price,
            exit_reason="stop_loss",
            commission=COMMISSION,
            bar_count=5,
        )

        # gross_pnl = 1 * (20850 - 21000) * 1.0 = -150.0
        assert record.gross_pnl == pytest.approx(-150.0)
        # realized = -150.0 - 25.2 = -175.2
        assert record.realized_pnl == pytest.approx(-150.0 - 2 * COMMISSION)
        assert record.realized_pnl < 0
        assert record.outcome == "LOSS"
        assert record.exit_reason == "stop_loss"


# ---------------------------------------------------------------------------
# 4. test_close_trade_breakeven
# ---------------------------------------------------------------------------
class TestCloseTradeBreakeven:
    def test_close_trade_breakeven(self):
        """Close near entry (within 2*commission), verify outcome = BREAKEVEN."""
        log, tid = _make_log_with_open_trade(direction=1)

        # We need |realized_pnl| <= 2 * total_commission
        # total_commission = 2 * 12.6 = 25.2, so threshold = 2 * 25.2 = 50.4
        # gross_pnl must be near total_commission => 25.2
        # gross_pnl = (exit - 21000) * 1.0 => exit = 21025.2 gives gross=25.2
        # realized = 25.2 - 25.2 = 0.0 which is <= 50.4 => BREAKEVEN
        exit_price = 21000.0 + 2 * COMMISSION  # 21025.2
        record = log.close_trade(
            trade_id=tid,
            exit_time=EXIT_TIME,
            exit_price=exit_price,
            exit_signal_price=exit_price,
            exit_reason="manual_close",
            commission=COMMISSION,
            bar_count=3,
        )

        assert record.realized_pnl == pytest.approx(0.0)
        assert record.outcome == "BREAKEVEN"


# ---------------------------------------------------------------------------
# 5. test_update_excursion_long
# ---------------------------------------------------------------------------
class TestUpdateExcursionLong:
    def test_update_excursion_long(self):
        """Open long, update with high/low, verify MFE/MAE."""
        log, tid = _make_log_with_open_trade(direction=1)

        # Bar 1: high=21150, low=20950
        log.update_excursion(tid, candle_high=21150.0, candle_low=20950.0)
        record = log.get_trade(tid)
        assert record.max_favorable_excursion == pytest.approx(150.0)  # 21150 - 21000
        assert record.max_adverse_excursion == pytest.approx(50.0)  # 21000 - 20950

        # Bar 2: higher MFE, lower MAE
        log.update_excursion(tid, candle_high=21250.0, candle_low=20900.0)
        assert record.max_favorable_excursion == pytest.approx(250.0)  # 21250 - 21000
        assert record.max_adverse_excursion == pytest.approx(100.0)  # 21000 - 20900

        # Bar 3: less extreme -- MFE/MAE should NOT decrease
        log.update_excursion(tid, candle_high=21050.0, candle_low=20980.0)
        assert record.max_favorable_excursion == pytest.approx(250.0)
        assert record.max_adverse_excursion == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 6. test_update_excursion_short
# ---------------------------------------------------------------------------
class TestUpdateExcursionShort:
    def test_update_excursion_short(self):
        """Open short, update with high/low, verify MFE/MAE."""
        log, tid = _make_log_with_open_trade(direction=-1)

        # For SHORT: MFE = entry - low, MAE = high - entry
        log.update_excursion(tid, candle_high=21100.0, candle_low=20800.0)
        record = log.get_trade(tid)
        assert record.max_favorable_excursion == pytest.approx(200.0)  # 21000 - 20800
        assert record.max_adverse_excursion == pytest.approx(100.0)  # 21100 - 21000

        # Second bar: bigger adverse, same favorable
        log.update_excursion(tid, candle_high=21200.0, candle_low=20850.0)
        assert record.max_favorable_excursion == pytest.approx(200.0)  # unchanged
        assert record.max_adverse_excursion == pytest.approx(200.0)  # 21200 - 21000


# ---------------------------------------------------------------------------
# 7. test_r_multiple_calculation
# ---------------------------------------------------------------------------
class TestRMultipleCalculation:
    def test_r_multiple_long(self):
        """Known values: entry=100, SL=95, exit=110 (long) -> R = 2.0."""
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=110.0,
            stop_loss=95.0,
            direction=1,
        )
        # risk = 100 - 95 = 5, reward = 110 - 100 = 10, R = 10/5 = 2.0
        assert r == pytest.approx(2.0)

    def test_r_multiple_short(self):
        """Short: entry=100, SL=105, exit=90 -> R = 2.0."""
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=90.0,
            stop_loss=105.0,
            direction=-1,
        )
        # risk = 105 - 100 = 5, reward = 100 - 90 = 10, R = 10/5 = 2.0
        assert r == pytest.approx(2.0)

    def test_r_multiple_zero_risk(self):
        """If risk is zero or negative, R should be 0.0."""
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=110.0,
            stop_loss=100.0,  # SL at entry -> zero risk
            direction=1,
        )
        assert r == 0.0


# ---------------------------------------------------------------------------
# 8. test_to_dataframe
# ---------------------------------------------------------------------------
class TestToDataframe:
    def test_to_dataframe(self):
        """Open + close trades, verify DataFrame columns and row count."""
        log, tid0 = _make_log_with_open_trade(direction=1)
        _tid1 = log.open_trade(
            poi_id="OB_15m_002",
            direction=-1,
            entry_time=ENTRY_TIME,
            entry_price=ENTRY_PRICE,
            entry_signal_price=SIGNAL_PRICE,
            position_size=POSITION_SIZE,
            stop_loss=21100.0,
            target=20700.0,
            commission=COMMISSION,
        )
        # Close only the first trade
        log.close_trade(
            trade_id=tid0,
            exit_time=EXIT_TIME,
            exit_price=21200.0,
            exit_signal_price=21200.0,
            exit_reason="target_hit",
            commission=COMMISSION,
            bar_count=10,
        )

        df = log.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "trade_id" in df.columns
        assert "realized_pnl" in df.columns
        assert "outcome" in df.columns
        assert "r_multiple" in df.columns
        assert "max_favorable_excursion" in df.columns
        assert "max_adverse_excursion" in df.columns
        # First trade should be closed with outcome
        assert df.loc[0, "outcome"] == "WIN"
        # Second trade should still be open (empty outcome)
        assert df.loc[1, "outcome"] == ""

    def test_to_dataframe_empty(self):
        """Empty TradeLog produces a DataFrame with expected columns."""
        log = TradeLog()
        df = log.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "trade_id" in df.columns


# ---------------------------------------------------------------------------
# 9. test_get_open_trades
# ---------------------------------------------------------------------------
class TestGetOpenTrades:
    def test_get_open_trades(self):
        """Mix of open and closed, verify filter works."""
        log, tid0 = _make_log_with_open_trade(direction=1)
        tid1 = log.open_trade(
            poi_id="OB_15m_002",
            direction=-1,
            entry_time=ENTRY_TIME,
            entry_price=ENTRY_PRICE,
            entry_signal_price=SIGNAL_PRICE,
            position_size=POSITION_SIZE,
            stop_loss=21100.0,
            target=20700.0,
            commission=COMMISSION,
        )
        tid2 = log.open_trade(
            poi_id="OB_15m_003",
            direction=1,
            entry_time=ENTRY_TIME,
            entry_price=ENTRY_PRICE,
            entry_signal_price=SIGNAL_PRICE,
            position_size=POSITION_SIZE,
            stop_loss=STOP_LOSS,
            target=TARGET,
            commission=COMMISSION,
        )

        # All three open
        assert len(log.get_open_trades()) == 3

        # Close tid0
        log.close_trade(
            trade_id=tid0,
            exit_time=EXIT_TIME,
            exit_price=21200.0,
            exit_signal_price=21200.0,
            exit_reason="target_hit",
            commission=COMMISSION,
            bar_count=10,
        )
        open_trades = log.get_open_trades()
        assert len(open_trades) == 2
        open_ids = {t.trade_id for t in open_trades}
        assert tid0 not in open_ids
        assert tid1 in open_ids
        assert tid2 in open_ids

        # Close tid1
        log.close_trade(
            trade_id=tid1,
            exit_time=EXIT_TIME,
            exit_price=20900.0,
            exit_signal_price=20900.0,
            exit_reason="stop_loss",
            commission=COMMISSION,
            bar_count=8,
        )
        assert len(log.get_open_trades()) == 1
        assert log.get_open_trades()[0].trade_id == tid2


# ---------------------------------------------------------------------------
# 10. test_classify_outcome
# ---------------------------------------------------------------------------
class TestClassifyOutcome:
    def test_classify_win(self):
        assert classify_outcome(realized_pnl=100.0, commission_total=10.0) == "WIN"

    def test_classify_loss(self):
        assert classify_outcome(realized_pnl=-100.0, commission_total=10.0) == "LOSS"

    def test_classify_breakeven_zero_pnl(self):
        assert classify_outcome(realized_pnl=0.0, commission_total=10.0) == "BREAKEVEN"

    def test_classify_breakeven_within_threshold(self):
        # |15.0| <= 2 * 10.0 = 20.0 => BREAKEVEN
        assert classify_outcome(realized_pnl=15.0, commission_total=10.0) == "BREAKEVEN"
        assert classify_outcome(realized_pnl=-15.0, commission_total=10.0) == "BREAKEVEN"

    def test_classify_breakeven_at_boundary(self):
        # |20.0| <= 2 * 10.0 = 20.0 => BREAKEVEN (boundary)
        assert classify_outcome(realized_pnl=20.0, commission_total=10.0) == "BREAKEVEN"

    def test_classify_just_above_threshold(self):
        # |20.1| > 2 * 10.0 = 20.0 => WIN
        assert classify_outcome(realized_pnl=20.1, commission_total=10.0) == "WIN"

    def test_classify_zero_commission(self):
        # Zero commission -> uses 0.01 fallback
        # |5.0| <= 2 * 0.01 = 0.02 => False => WIN
        assert classify_outcome(realized_pnl=5.0, commission_total=0.0) == "WIN"
        assert classify_outcome(realized_pnl=-5.0, commission_total=0.0) == "LOSS"
        # Very small pnl within threshold
        assert classify_outcome(realized_pnl=0.01, commission_total=0.0) == "BREAKEVEN"
