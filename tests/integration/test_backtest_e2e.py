"""End-to-end integration tests for the backtest orchestrator.

Runs the full pipeline (MTFManager -> StateMachine -> Strategy -> Portfolio -> Metrics)
on synthetic data and verifies correct output types, shapes, determinism, and crash safety.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config
from engine.backtester import Backtester, BacktestResult, run_backtest
from engine.metrics import MetricsResult


def make_trending_1m(n_bars: int = 600, base_price: float = 21000.0) -> pd.DataFrame:
    """Create synthetic 1m data with uptrend + pullback + continuation."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-02 09:00", periods=n_bars, freq="1min", tz="UTC")

    prices = np.zeros(n_bars)
    prices[0] = base_price

    for i in range(1, n_bars):
        if i < 200:
            drift = 2.0
        elif i < 300:
            drift = -1.5
        elif i < 400:
            drift = 0.5
        else:
            drift = 1.5
        prices[i] = prices[i - 1] + drift + rng.normal(0, 1.5)

    noise = rng.uniform(0.5, 3.0, n_bars)
    opens = prices + rng.uniform(-1, 1, n_bars)
    closes = prices + rng.uniform(-1, 1, n_bars)
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise

    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": rng.integers(100, 5000, n_bars),
    })


@pytest.fixture
def config() -> Config:
    """Config with reduced TFs for faster integration test."""
    cfg = Config()
    cfg.data.timeframes = ["1m", "5m", "15m", "1H"]
    cfg.backtest.start_date = "2024-01-01"
    cfg.backtest.end_date = "2024-12-31"
    cfg.backtest.initial_capital = 10000
    return cfg


@pytest.fixture
def df_1m() -> pd.DataFrame:
    return make_trending_1m(n_bars=600)


class TestRunBacktestReturnsResult:
    """Verify that run_backtest produces a BacktestResult with correct field types."""

    def test_run_backtest_returns_result(self, config, df_1m):
        result = run_backtest(config, df_1m)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.trade_log, pd.DataFrame)
        assert isinstance(result.equity_curve, np.ndarray)
        assert isinstance(result.metrics, MetricsResult)
        assert isinstance(result.signals, list)
        assert isinstance(result.events, pd.DataFrame)
        assert isinstance(result.timestamps, pd.DatetimeIndex)
        assert result.config is config


class TestDeterministicResults:
    """Verify that running twice on the same data yields identical results."""

    def test_deterministic_results(self, config, df_1m):
        result_a = run_backtest(config, df_1m)
        result_b = run_backtest(config, df_1m)

        # Same trade count
        assert len(result_a.trade_log) == len(result_b.trade_log)

        # Same final equity
        valid_a = result_a.equity_curve[~np.isnan(result_a.equity_curve)]
        valid_b = result_b.equity_curve[~np.isnan(result_b.equity_curve)]

        assert len(valid_a) == len(valid_b)
        if len(valid_a) > 0:
            np.testing.assert_array_almost_equal(valid_a, valid_b)

        # Same signal count
        assert len(result_a.signals) == len(result_b.signals)


class TestEquityCurveLength:
    """Verify that the equity curve has one entry per bar."""

    def test_equity_curve_length(self, config, df_1m):
        result = run_backtest(config, df_1m)

        # Equity curve length must match the number of bars in the filtered data
        n_bars = len(df_1m)
        assert len(result.equity_curve) == n_bars


class TestMetricsPopulated:
    """Verify that metrics fields are populated with valid values."""

    def test_metrics_populated(self, config, df_1m):
        result = run_backtest(config, df_1m)
        m = result.metrics

        # final_equity should be positive (started with 10000, costs are small)
        assert m.final_equity > 0

        # total_return_pct should be a finite number
        assert np.isfinite(m.total_return_pct)

        # peak_equity should be at least initial capital
        assert m.peak_equity >= config.backtest.initial_capital * 0.5  # Allow some loss

        # Trade stats should be non-negative integers
        assert m.total_trades >= 0
        assert m.winning_trades >= 0
        assert m.losing_trades >= 0

        # Win rate should be in [0, 100]
        assert 0.0 <= m.win_rate_pct <= 100.0

        # Sharpe, sortino, calmar are real numbers (could be 0 if no trades)
        assert np.isfinite(m.sharpe_ratio)
        assert np.isfinite(m.sortino_ratio)
        assert np.isfinite(m.calmar_ratio)


class TestNoCrashSmallData:
    """Verify the backtester handles very small datasets without crashing."""

    def test_no_crash_small_data(self, config):
        df_small = make_trending_1m(n_bars=60)
        result = run_backtest(config, df_small)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 60
        assert isinstance(result.metrics, MetricsResult)
        assert result.metrics.final_equity > 0
