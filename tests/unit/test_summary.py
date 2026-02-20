"""Unit tests for reporting.summary module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from config import Config
from engine.backtester import BacktestResult
from engine.metrics import MetricsResult
from reporting.summary import format_metrics_table, format_sync_stats, print_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_metrics(**overrides) -> MetricsResult:
    """Build a MetricsResult with sensible non-zero defaults."""
    defaults = dict(
        total_return_pct=5.23,
        cagr_pct=12.45,
        max_drawdown_pct=3.45,
        max_drawdown_duration_bars=42,
        sharpe_ratio=1.23,
        sortino_ratio=1.56,
        calmar_ratio=3.61,
        total_trades=15,
        winning_trades=9,
        losing_trades=5,
        breakeven_trades=1,
        win_rate_pct=60.0,
        avg_rr=0.45,
        avg_win_rr=1.23,
        avg_loss_rr=-0.67,
        profit_factor=2.15,
        expectancy=0.45,
        avg_trade_duration_bars=23,
        avg_win_duration_bars=18,
        avg_loss_duration_bars=31,
        final_equity=10523.0,
        peak_equity=10800.0,
    )
    defaults.update(overrides)
    return MetricsResult(**defaults)


def _make_result(metrics: MetricsResult | None = None,
                 n_bars: int = 100) -> BacktestResult:
    """Build a minimal BacktestResult for testing."""
    if metrics is None:
        metrics = _make_metrics()
    equity = np.linspace(10000, 10500, n_bars)
    timestamps = pd.date_range(
        "2024-01-02 09:00", periods=n_bars, freq="1min", tz="UTC"
    )
    trade_log = pd.DataFrame(columns=[
        "outcome", "r_multiple", "realized_pnl", "duration_bars",
        "sync_mode", "exit_time",
    ])
    events = pd.DataFrame(columns=["type", "timestamp", "poi_id"])
    config = Config()
    return BacktestResult(
        trade_log=trade_log,
        equity_curve=equity,
        metrics=metrics,
        signals=[],
        events=events,
        config=config,
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrintSummary:
    """Tests for print_summary."""

    def test_returns_string(self):
        """print_summary returns a non-empty string."""
        result = _make_result()
        text = print_summary(result)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_contains_sections(self):
        """Output contains all major section headings."""
        result = _make_result()
        text = print_summary(result)
        assert "Returns" in text
        assert "Risk" in text
        assert "Trades" in text
        assert "Trade Quality" in text
        assert "Duration" in text

    def test_contains_period_info(self):
        """Output contains date range and capital info."""
        result = _make_result()
        text = print_summary(result)
        assert "2024-01-02" in text
        assert "$10,000.00" in text
        assert "100" in text  # data bars

    def test_zero_trades(self):
        """When total_trades == 0, shows 'No trades executed'."""
        metrics = _make_metrics(total_trades=0)
        result = _make_result(metrics=metrics)
        text = print_summary(result)
        assert "No trades executed" in text
        # Should still show returns section
        assert "Returns" in text
        # Should NOT show trade-specific sections
        assert "Trade Quality" not in text

    def test_header_and_footer_bars(self):
        """Output starts and ends with '=' border lines."""
        result = _make_result()
        text = print_summary(result)
        lines = text.strip().split("\n")
        assert lines[0] == "=" * 80
        assert lines[-1] == "=" * 80


class TestFormatSyncStats:
    """Tests for format_sync_stats."""

    def test_empty_dict_returns_empty_string(self):
        """Empty sync_stats produces empty string."""
        assert format_sync_stats({}) == ""

    def test_with_data(self):
        """Non-empty sync_stats returns formatted block with mode names."""
        stats = {
            "SYNC": {
                "trades": 10,
                "win_rate": 70.0,
                "avg_rr": 0.65,
                "profit_factor": 2.50,
            },
            "DESYNC": {
                "trades": 5,
                "win_rate": 40.0,
                "avg_rr": 0.10,
                "profit_factor": 1.10,
            },
        }
        text = format_sync_stats(stats)
        assert "SYNC" in text
        assert "DESYNC" in text
        assert "10 trades" in text
        assert "70.0% WR" in text
        assert "PF 2.50" in text

    def test_sync_stats_in_full_summary(self):
        """Sync stats section appears in print_summary when data present."""
        stats = {
            "SYNC": {
                "trades": 8,
                "win_rate": 62.5,
                "avg_rr": 0.55,
                "profit_factor": 2.00,
            },
        }
        metrics = _make_metrics(sync_stats=stats)
        result = _make_result(metrics=metrics)
        text = print_summary(result)
        assert "Sync Mode Breakdown" in text
        assert "SYNC" in text


class TestFormatMetricsTable:
    """Tests for format_metrics_table."""

    def test_returns_string(self):
        """format_metrics_table returns a non-empty string."""
        metrics = _make_metrics()
        text = format_metrics_table(metrics)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_contains_all_sections(self):
        """Metrics table includes all section dividers."""
        metrics = _make_metrics()
        text = format_metrics_table(metrics)
        assert "Returns" in text
        assert "Risk" in text
        assert "Trades" in text
        assert "Trade Quality" in text
        assert "Duration" in text
