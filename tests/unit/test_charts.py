"""Unit tests for reporting.charts module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from config import Config
from engine.backtester import BacktestResult
from engine.metrics import MetricsResult
from reporting.charts import (
    create_equity_curve_chart,
    create_monthly_heatmap,
    create_trade_scatter,
    create_r_distribution,
    create_mae_mfe_scatter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_log() -> pd.DataFrame:
    """Build a minimal trade log DataFrame with 5 mock trades."""
    base_time = pd.Timestamp("2024-01-02 09:30", tz="UTC")
    trades = [
        {
            "trade_id": 1,
            "poi_id": "POI_001",
            "direction": "LONG",
            "entry_time": base_time,
            "entry_price": 100.0,
            "exit_time": base_time + pd.Timedelta(minutes=10),
            "exit_price": 102.0,
            "realized_pnl": 200.0,
            "r_multiple": 2.0,
            "outcome": "WIN",
            "duration_bars": 10,
            "max_favorable_excursion": 0.025,
            "max_adverse_excursion": 0.005,
            "sync_mode": "SYNC",
        },
        {
            "trade_id": 2,
            "poi_id": "POI_002",
            "direction": "SHORT",
            "entry_time": base_time + pd.Timedelta(minutes=20),
            "entry_price": 101.0,
            "exit_time": base_time + pd.Timedelta(minutes=35),
            "exit_price": 102.5,
            "realized_pnl": -150.0,
            "r_multiple": -1.5,
            "outcome": "LOSS",
            "duration_bars": 15,
            "max_favorable_excursion": 0.008,
            "max_adverse_excursion": 0.018,
            "sync_mode": "SYNC",
        },
        {
            "trade_id": 3,
            "poi_id": "POI_003",
            "direction": "LONG",
            "entry_time": base_time + pd.Timedelta(minutes=40),
            "entry_price": 102.0,
            "exit_time": base_time + pd.Timedelta(minutes=50),
            "exit_price": 105.0,
            "realized_pnl": 300.0,
            "r_multiple": 3.0,
            "outcome": "WIN",
            "duration_bars": 10,
            "max_favorable_excursion": 0.035,
            "max_adverse_excursion": 0.003,
            "sync_mode": "PARTIAL",
        },
        {
            "trade_id": 4,
            "poi_id": "POI_004",
            "direction": "SHORT",
            "entry_time": base_time + pd.Timedelta(minutes=55),
            "entry_price": 104.0,
            "exit_time": base_time + pd.Timedelta(minutes=65),
            "exit_price": 105.0,
            "realized_pnl": -100.0,
            "r_multiple": -1.0,
            "outcome": "LOSS",
            "duration_bars": 10,
            "max_favorable_excursion": 0.005,
            "max_adverse_excursion": 0.012,
            "sync_mode": "SYNC",
        },
        {
            "trade_id": 5,
            "poi_id": "POI_005",
            "direction": "LONG",
            "entry_time": base_time + pd.Timedelta(minutes=70),
            "entry_price": 103.0,
            "exit_time": base_time + pd.Timedelta(minutes=80),
            "exit_price": 103.0,
            "realized_pnl": 0.0,
            "r_multiple": 0.0,
            "outcome": "BREAKEVEN",
            "duration_bars": 10,
            "max_favorable_excursion": 0.002,
            "max_adverse_excursion": 0.002,
            "sync_mode": "SYNC",
        },
    ]
    return pd.DataFrame(trades)


def _make_monthly_returns() -> pd.DataFrame:
    """Build a small monthly returns DataFrame."""
    return pd.DataFrame([
        {"month": "2024-01", "return_pct": 2.5, "trade_count": 3},
        {"month": "2024-02", "return_pct": -1.2, "trade_count": 2},
        {"month": "2024-03", "return_pct": 4.0, "trade_count": 4},
    ])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backtest_result() -> BacktestResult:
    """Build a minimal but realistic BacktestResult for chart testing."""
    equity = np.linspace(10000, 11000, 100)
    timestamps = pd.date_range("2024-01-02 09:00", periods=100, freq="1min", tz="UTC")

    metrics = MetricsResult(
        total_return_pct=10.0,
        max_drawdown_pct=3.5,
        sharpe_ratio=1.8,
        total_trades=5,
        winning_trades=2,
        losing_trades=2,
        breakeven_trades=1,
        win_rate_pct=40.0,
        avg_rr=0.5,
        avg_win_rr=2.5,
        avg_loss_rr=-1.25,
        profit_factor=2.0,
        expectancy=0.5,
        monthly_returns=_make_monthly_returns(),
        final_equity=11000.0,
        peak_equity=11000.0,
    )

    return BacktestResult(
        trade_log=_make_trade_log(),
        equity_curve=equity,
        metrics=metrics,
        signals=[],
        events=pd.DataFrame(),
        config=Config(),
        timestamps=timestamps,
    )


@pytest.fixture
def empty_backtest_result() -> BacktestResult:
    """BacktestResult with no trades and flat equity."""
    equity = np.full(10, 10000.0)
    timestamps = pd.date_range("2024-01-02 09:00", periods=10, freq="1min", tz="UTC")

    metrics = MetricsResult(monthly_returns=None)

    trade_cols = [
        "trade_id", "poi_id", "direction", "entry_time", "entry_price",
        "exit_time", "exit_price", "realized_pnl", "r_multiple", "outcome",
        "duration_bars", "max_favorable_excursion", "max_adverse_excursion",
        "sync_mode",
    ]

    return BacktestResult(
        trade_log=pd.DataFrame(columns=trade_cols),
        equity_curve=equity,
        metrics=metrics,
        signals=[],
        events=pd.DataFrame(),
        config=Config(),
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEquityCurveChart:

    def test_equity_curve_returns_figure(self, backtest_result: BacktestResult) -> None:
        """create_equity_curve_chart returns a Figure with data traces."""
        fig = create_equity_curve_chart(backtest_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # equity line + drawdown fill


class TestMonthlyHeatmap:

    def test_monthly_heatmap_returns_figure(self, backtest_result: BacktestResult) -> None:
        """create_monthly_heatmap returns a Figure with heatmap data."""
        fig = create_monthly_heatmap(backtest_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        # Should contain a Heatmap trace
        assert isinstance(fig.data[0], go.Heatmap)

    def test_monthly_heatmap_none_returns_empty(self, backtest_result: BacktestResult) -> None:
        """When monthly_returns is None, returns an empty figure with annotation."""
        backtest_result.metrics.monthly_returns = None
        fig = create_monthly_heatmap(backtest_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        # Should have an annotation explaining the empty state
        assert len(fig.layout.annotations) > 0


class TestTradeScatter:

    def test_trade_scatter_returns_figure(self, backtest_result: BacktestResult) -> None:
        """create_trade_scatter returns a Figure with scatter data."""
        fig = create_trade_scatter(backtest_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Scatter)

    def test_trade_scatter_empty_trades(self, empty_backtest_result: BacktestResult) -> None:
        """Empty trade_log returns a figure with annotation, no data traces."""
        fig = create_trade_scatter(empty_backtest_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert len(fig.layout.annotations) > 0


class TestRDistribution:

    def test_r_distribution_returns_figure(self, backtest_result: BacktestResult) -> None:
        """create_r_distribution returns a Figure with bar data."""
        fig = create_r_distribution(backtest_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Bar)


class TestMAEMFEScatter:

    def test_mae_mfe_scatter_returns_figure(self, backtest_result: BacktestResult) -> None:
        """create_mae_mfe_scatter returns a Figure with scatter data."""
        fig = create_mae_mfe_scatter(backtest_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Scatter)


class TestAllChartsEmptyData:

    def test_all_charts_empty_data(self, empty_backtest_result: BacktestResult) -> None:
        """All chart functions handle an empty BacktestResult gracefully."""
        result = empty_backtest_result

        # Each should return a go.Figure without raising
        fig_equity = create_equity_curve_chart(result)
        assert isinstance(fig_equity, go.Figure)

        fig_heatmap = create_monthly_heatmap(result)
        assert isinstance(fig_heatmap, go.Figure)

        fig_scatter = create_trade_scatter(result)
        assert isinstance(fig_scatter, go.Figure)

        fig_rdist = create_r_distribution(result)
        assert isinstance(fig_rdist, go.Figure)

        fig_mae = create_mae_mfe_scatter(result)
        assert isinstance(fig_mae, go.Figure)

        # The trade-dependent charts should have annotations (empty state)
        for fig in [fig_heatmap, fig_scatter, fig_rdist, fig_mae]:
            assert len(fig.layout.annotations) > 0, (
                "Expected annotation on empty chart, got none"
            )
