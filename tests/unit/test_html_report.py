"""Unit tests for reporting.html_report module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from config import Config
from engine.backtester import BacktestResult
from engine.metrics import MetricsResult
from reporting.html_report import generate_report, _trade_log_to_html


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


def _make_empty_trade_log() -> pd.DataFrame:
    """Build an empty trade log with the expected columns."""
    trade_cols = [
        "trade_id", "poi_id", "direction", "entry_time", "entry_price",
        "exit_time", "exit_price", "realized_pnl", "r_multiple", "outcome",
        "duration_bars", "max_favorable_excursion", "max_adverse_excursion",
        "sync_mode",
    ]
    return pd.DataFrame(columns=trade_cols)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backtest_result() -> BacktestResult:
    """Build a minimal but realistic BacktestResult for report testing."""
    equity = np.linspace(10000, 11000, 100)
    timestamps = pd.date_range(
        "2024-01-02 09:00", periods=100, freq="1min", tz="UTC"
    )

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
    timestamps = pd.date_range(
        "2024-01-02 09:00", periods=10, freq="1min", tz="UTC"
    )

    metrics = MetricsResult(monthly_returns=None)

    return BacktestResult(
        trade_log=_make_empty_trade_log(),
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

class TestGenerateReport:

    def test_generate_report_creates_file(
        self, backtest_result: BacktestResult, tmp_path: Path
    ) -> None:
        """generate_report creates report.html in the output directory."""
        report_path = generate_report(backtest_result, tmp_path)
        assert report_path.exists()
        assert report_path.name == "report.html"
        assert report_path.parent == tmp_path

    def test_report_html_contains_sections(
        self, backtest_result: BacktestResult, tmp_path: Path
    ) -> None:
        """The generated HTML contains key section headings."""
        report_path = generate_report(backtest_result, tmp_path)
        html = report_path.read_text(encoding="utf-8")

        assert "Summary" in html
        assert "Equity Curve" in html
        assert "Trade Log" in html

    def test_report_html_contains_plotly_js(
        self, backtest_result: BacktestResult, tmp_path: Path
    ) -> None:
        """The generated HTML includes the Plotly.js CDN script tag."""
        report_path = generate_report(backtest_result, tmp_path)
        html = report_path.read_text(encoding="utf-8")

        assert "plotly" in html.lower()
        assert "cdn.plot.ly" in html

    def test_generate_report_zero_trades(
        self, empty_backtest_result: BacktestResult, tmp_path: Path
    ) -> None:
        """generate_report works with an empty trade log (zero trades)."""
        report_path = generate_report(empty_backtest_result, tmp_path)
        assert report_path.exists()

        html = report_path.read_text(encoding="utf-8")
        assert "No trades executed." in html

    def test_output_dir_created(
        self, backtest_result: BacktestResult, tmp_path: Path
    ) -> None:
        """generate_report creates the output directory if it doesn't exist."""
        nested_dir = tmp_path / "sub" / "reports"
        assert not nested_dir.exists()

        report_path = generate_report(backtest_result, nested_dir)
        assert nested_dir.exists()
        assert report_path.exists()


class TestTradeLogToHtml:

    def test_trade_log_to_html_empty(self) -> None:
        """An empty DataFrame returns a 'No trades' message."""
        empty_df = _make_empty_trade_log()
        html = _trade_log_to_html(empty_df)
        assert "No trades executed." in html

    def test_trade_log_to_html_none(self) -> None:
        """A None DataFrame returns a 'No trades' message."""
        html = _trade_log_to_html(None)
        assert "No trades executed." in html

    def test_trade_log_to_html_with_trades(self) -> None:
        """Trade log HTML contains table headers and trade data."""
        trade_df = _make_trade_log()
        html = _trade_log_to_html(trade_df)

        # Check table headers
        assert "<th>#</th>" in html
        assert "<th>Dir</th>" in html
        assert "<th>R-Multiple</th>" in html
        assert "<th>PnL</th>" in html
        assert "<th>Outcome</th>" in html

        # Check outcome CSS classes
        assert 'class="win"' in html
        assert 'class="loss"' in html
        assert 'class="be"' in html

    def test_trade_log_to_html_formatting(self) -> None:
        """Trade data is formatted correctly (prices, R-multiples, PnL)."""
        trade_df = _make_trade_log()
        html = _trade_log_to_html(trade_df)

        # R-multiple format: X.XXR
        assert "2.00R" in html
        assert "-1.50R" in html

        # PnL format: $X,XXX.XX
        assert "$200.00" in html
        assert "$-150.00" in html

        # Duration format: X bars
        assert "10 bars" in html
        assert "15 bars" in html

    def test_trade_log_to_html_truncation(self) -> None:
        """When more than 200 trades, the table is truncated with a note."""
        base_time = pd.Timestamp("2024-01-02 09:30", tz="UTC")
        rows = []
        for i in range(250):
            rows.append({
                "trade_id": i + 1,
                "poi_id": f"POI_{i:03d}",
                "direction": "LONG",
                "entry_time": base_time + pd.Timedelta(minutes=i * 10),
                "entry_price": 100.0,
                "exit_time": base_time + pd.Timedelta(minutes=i * 10 + 5),
                "exit_price": 101.0,
                "realized_pnl": 100.0,
                "r_multiple": 1.0,
                "outcome": "WIN",
                "duration_bars": 5,
                "sync_mode": "SYNC",
            })
        big_df = pd.DataFrame(rows)
        html = _trade_log_to_html(big_df)

        assert "Showing 200 of 250 trades" in html
