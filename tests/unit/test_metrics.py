"""Unit tests for engine.metrics module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from engine.metrics import (
    MetricsResult,
    compute_calmar,
    compute_drawdown,
    compute_metrics,
    compute_return_metrics,
    compute_sharpe,
    compute_sortino,
    compute_sync_mode_stats,
    compute_trade_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_df(trades: list[dict]) -> pd.DataFrame:
    """Build a trade DataFrame from a list of dicts."""
    cols = ["outcome", "r_multiple", "realized_pnl", "duration_bars", "sync_mode", "exit_time"]
    if not trades:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(trades)
    for c in cols:
        if c not in df.columns:
            if c == "exit_time":
                df[c] = pd.Timestamp("2024-01-02 10:00", tz="UTC")
            elif c == "sync_mode":
                df[c] = "SYNC"
            elif c == "duration_bars":
                df[c] = 100
            else:
                df[c] = 0.0
    return df


# ---------------------------------------------------------------------------
# 1. Total return
# ---------------------------------------------------------------------------

class TestTotalReturn:
    def test_total_return(self):
        """Equity [10000, ..., 11000] -> 10% return."""
        equity = np.linspace(10000, 11000, 50)
        result = compute_return_metrics(equity, initial_capital=10000.0)
        assert result["total_return_pct"] == pytest.approx(10.0, abs=0.01)

    def test_total_return_loss(self):
        """Equity ending lower than start -> negative return."""
        equity = np.array([10000.0, 9500.0, 9000.0])
        result = compute_return_metrics(equity, initial_capital=10000.0)
        assert result["total_return_pct"] == pytest.approx(-10.0, abs=0.01)

    def test_total_return_empty(self):
        """Empty equity curve -> 0% return."""
        equity = np.array([])
        result = compute_return_metrics(equity, initial_capital=10000.0)
        assert result["total_return_pct"] == pytest.approx(0.0)
        assert result["cagr_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_max_drawdown(self):
        """Equity [10000, 11000, 9000, 10500] -> DD from peak 11000 to trough 9000 = 18.18%."""
        equity = np.array([10000.0, 11000.0, 9000.0, 10500.0])
        _, max_dd_pct, _ = compute_drawdown(equity)
        # max_dd_pct should be (11000 - 9000) / 11000 = 0.18181...
        assert max_dd_pct == pytest.approx(0.1818, abs=0.001)

    def test_no_drawdown(self):
        """Monotonically rising equity -> 0% drawdown."""
        equity = np.array([10000.0, 10100.0, 10200.0, 10300.0])
        _, max_dd_pct, max_dd_dur = compute_drawdown(equity)
        assert max_dd_pct == pytest.approx(0.0)
        assert max_dd_dur == 0


# ---------------------------------------------------------------------------
# 3. Max drawdown duration
# ---------------------------------------------------------------------------

class TestMaxDrawdownDuration:
    def test_max_drawdown_duration(self):
        """Equity that dips for 3 bars then recovers -> duration = 3."""
        # Peak at bar 1 (11000), dips for bars 2,3,4, recovers at bar 5
        equity = np.array([10000.0, 11000.0, 10800.0, 10500.0, 10700.0, 11100.0])
        _, _, max_dd_dur = compute_drawdown(equity)
        assert max_dd_dur == 3


# ---------------------------------------------------------------------------
# 4. Sharpe positive
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_sharpe_positive(self):
        """Steadily rising equity -> positive Sharpe."""
        # Create a steadily rising equity curve with small noise
        rng = np.random.default_rng(42)
        base = np.linspace(10000, 11000, 500)
        noise = rng.normal(0, 5, 500)
        equity = base + noise

        sharpe = compute_sharpe(equity, bars_per_year=252 * 390)
        assert sharpe > 0.0

    def test_sharpe_zero_if_constant(self):
        """Flat equity -> 0 Sharpe (zero std)."""
        equity = np.full(100, 10000.0)
        sharpe = compute_sharpe(equity, bars_per_year=252 * 390)
        assert sharpe == pytest.approx(0.0)

    def test_sharpe_short_curve(self):
        """Equity with fewer than 2 points -> 0 Sharpe."""
        equity = np.array([10000.0])
        sharpe = compute_sharpe(equity)
        assert sharpe == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Sortino positive
# ---------------------------------------------------------------------------

class TestSortino:
    def test_sortino_positive(self):
        """Rising equity with some negative bars -> positive Sortino."""
        # Mostly rising but with periodic dips
        equity = np.array([
            10000, 10050, 10030, 10100, 10080, 10150,
            10120, 10200, 10180, 10250, 10300, 10350,
        ], dtype=float)
        sortino = compute_sortino(equity, bars_per_year=252 * 390)
        assert sortino > 0.0

    def test_sortino_no_downside(self):
        """Monotonically rising equity -> 0 Sortino (no downside returns)."""
        equity = np.array([10000.0, 10100.0, 10200.0, 10300.0, 10400.0])
        sortino = compute_sortino(equity, bars_per_year=252 * 390)
        assert sortino == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Win rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_win_rate(self):
        """6 wins, 4 losses -> 60% win rate."""
        trades = []
        for _ in range(6):
            trades.append({
                "outcome": "WIN", "r_multiple": 2.0,
                "realized_pnl": 200.0, "duration_bars": 50,
            })
        for _ in range(4):
            trades.append({
                "outcome": "LOSS", "r_multiple": -1.0,
                "realized_pnl": -100.0, "duration_bars": 30,
            })

        df = _make_trade_df(trades)
        stats = compute_trade_stats(df)

        assert stats["total_trades"] == 10
        assert stats["winning_trades"] == 6
        assert stats["losing_trades"] == 4
        assert stats["win_rate_pct"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# 7. Profit factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_profit_factor(self):
        """Known wins/losses -> gross_profit / gross_loss."""
        trades = [
            {"outcome": "WIN", "r_multiple": 2.0, "realized_pnl": 500.0, "duration_bars": 50},
            {"outcome": "WIN", "r_multiple": 1.5, "realized_pnl": 300.0, "duration_bars": 40},
            {"outcome": "LOSS", "r_multiple": -1.0, "realized_pnl": -200.0, "duration_bars": 30},
            {"outcome": "LOSS", "r_multiple": -1.0, "realized_pnl": -100.0, "duration_bars": 25},
        ]
        df = _make_trade_df(trades)
        stats = compute_trade_stats(df)

        # gross_profit = 500 + 300 = 800; gross_loss = |(-200) + (-100)| = 300
        assert stats["profit_factor"] == pytest.approx(800.0 / 300.0, abs=0.01)

    def test_profit_factor_no_losses(self):
        """All wins, no losses -> profit_factor = 0 (division guard)."""
        trades = [
            {"outcome": "WIN", "r_multiple": 2.0, "realized_pnl": 500.0, "duration_bars": 50},
        ]
        df = _make_trade_df(trades)
        stats = compute_trade_stats(df)

        # gross_loss = 0, so profit_factor returns 0.0 (guarded)
        assert stats["profit_factor"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. No trades safe
# ---------------------------------------------------------------------------

class TestNoTradesSafe:
    def test_no_trades_safe(self):
        """Empty trade_df -> all zeros, no crash."""
        df = _make_trade_df([])
        stats = compute_trade_stats(df)

        assert stats["total_trades"] == 0
        assert stats["winning_trades"] == 0
        assert stats["losing_trades"] == 0
        assert stats["breakeven_trades"] == 0
        assert stats["win_rate_pct"] == pytest.approx(0.0)
        assert stats["avg_rr"] == pytest.approx(0.0)
        assert stats["profit_factor"] == pytest.approx(0.0)
        assert stats["expectancy"] == pytest.approx(0.0)
        assert stats["avg_trade_duration_bars"] == 0

    def test_no_closed_trades_safe(self):
        """Trades with non-standard outcomes -> treated as zero closed."""
        df = _make_trade_df([{"outcome": "OPEN", "r_multiple": 0.0, "realized_pnl": 0.0}])
        stats = compute_trade_stats(df)
        assert stats["total_trades"] == 0


# ---------------------------------------------------------------------------
# 9. Sync mode stats
# ---------------------------------------------------------------------------

class TestSyncModeStats:
    def test_sync_mode_stats(self):
        """Mixed SYNC/DESYNC trades -> separate stats per mode."""
        trades = [
            {"outcome": "WIN", "r_multiple": 2.0, "realized_pnl": 200.0,
             "duration_bars": 50, "sync_mode": "SYNC"},
            {"outcome": "WIN", "r_multiple": 1.5, "realized_pnl": 150.0,
             "duration_bars": 40, "sync_mode": "SYNC"},
            {"outcome": "LOSS", "r_multiple": -1.0, "realized_pnl": -100.0,
             "duration_bars": 30, "sync_mode": "SYNC"},
            {"outcome": "WIN", "r_multiple": 3.0, "realized_pnl": 300.0,
             "duration_bars": 60, "sync_mode": "DESYNC"},
            {"outcome": "LOSS", "r_multiple": -1.0, "realized_pnl": -100.0,
             "duration_bars": 20, "sync_mode": "DESYNC"},
        ]
        df = _make_trade_df(trades)
        sync_stats = compute_sync_mode_stats(df)

        assert "SYNC" in sync_stats
        assert "DESYNC" in sync_stats

        # SYNC: 2 wins out of 3 trades -> 66.67% win rate
        assert sync_stats["SYNC"]["trades"] == 3
        assert sync_stats["SYNC"]["win_rate"] == pytest.approx(66.667, abs=0.1)

        # DESYNC: 1 win out of 2 trades -> 50% win rate
        assert sync_stats["DESYNC"]["trades"] == 2
        assert sync_stats["DESYNC"]["win_rate"] == pytest.approx(50.0)

    def test_sync_mode_empty(self):
        """Empty trade_df -> empty sync stats."""
        df = _make_trade_df([])
        sync_stats = compute_sync_mode_stats(df)
        assert sync_stats == {}


# ---------------------------------------------------------------------------
# 10. Full compute_metrics integration
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_compute_metrics_integration(self):
        """Full pipeline: equity curve + trades -> MetricsResult with all fields."""
        equity = np.array([
            10000, 10100, 10050, 10200, 10150, 10300,
            10250, 10400, 10350, 10500, 10600, 10700,
        ], dtype=float)

        trades = [
            {"outcome": "WIN", "r_multiple": 2.0, "realized_pnl": 300.0,
             "duration_bars": 50, "sync_mode": "SYNC"},
            {"outcome": "WIN", "r_multiple": 1.5, "realized_pnl": 200.0,
             "duration_bars": 40, "sync_mode": "SYNC"},
            {"outcome": "LOSS", "r_multiple": -1.0, "realized_pnl": -100.0,
             "duration_bars": 30, "sync_mode": "DESYNC"},
        ]
        df = _make_trade_df(trades)

        result = compute_metrics(
            trade_df=df,
            equity_curve=equity,
            initial_capital=10000.0,
            bars_per_year=252 * 390,
        )

        assert isinstance(result, MetricsResult)
        assert result.total_return_pct == pytest.approx(7.0, abs=0.01)
        assert result.total_trades == 3
        assert result.winning_trades == 2
        assert result.losing_trades == 1
        assert result.win_rate_pct == pytest.approx(66.667, abs=0.1)
        assert result.final_equity == pytest.approx(10700.0)
        assert result.peak_equity == pytest.approx(10700.0)
        assert result.sharpe_ratio > 0.0
        assert result.max_drawdown_pct > 0.0
        assert "SYNC" in result.sync_stats
        assert "DESYNC" in result.sync_stats

    def test_compute_metrics_with_nan(self):
        """Equity curve with NaN values should be handled gracefully."""
        equity = np.array([10000, np.nan, 10100, 10200, np.nan, 10300])

        df = _make_trade_df([])
        result = compute_metrics(
            trade_df=df,
            equity_curve=equity,
            initial_capital=10000.0,
        )

        assert isinstance(result, MetricsResult)
        assert result.total_return_pct == pytest.approx(3.0, abs=0.01)
        assert result.total_trades == 0


# ---------------------------------------------------------------------------
# 11. Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmar:
    def test_calmar_ratio(self):
        """Calmar = CAGR / max_drawdown_pct."""
        assert compute_calmar(20.0, 10.0) == pytest.approx(2.0)

    def test_calmar_zero_drawdown(self):
        """Zero drawdown -> 0 calmar (guarded)."""
        assert compute_calmar(20.0, 0.0) == pytest.approx(0.0)
