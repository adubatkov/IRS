"""End-to-end integration tests for the reporting layer.

Runs the full pipeline: run_backtest() -> generate_report() / print_summary()
and verifies correct output.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config
from engine.backtester import run_backtest
from reporting import generate_report, print_summary
from tests.conftest import make_trending_1m


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.data.timeframes = ["1m", "5m", "15m", "1H"]
    cfg.backtest.initial_capital = 10000
    return cfg


@pytest.fixture
def df_1m():
    return make_trending_1m(n_bars=600)


@pytest.fixture
def result(config, df_1m):
    return run_backtest(config, df_1m)


class TestGenerateReportE2E:

    def test_full_pipeline_produces_html(self, result, tmp_path):
        """run_backtest -> generate_report produces a valid HTML file."""
        path = generate_report(result, output_dir=tmp_path)

        assert path.exists()
        assert path.name == "report.html"

        html = path.read_text(encoding="utf-8")
        assert len(html) > 1000
        assert "plotly" in html.lower()
        assert "Equity Curve" in html
        assert "Summary" in html

    def test_print_summary_returns_text(self, result):
        """print_summary returns non-empty text with correct values."""
        text = print_summary(result)

        assert isinstance(text, str)
        assert len(text) > 100
        assert "BACKTEST RESULTS SUMMARY" in text
        assert "Returns" in text

        # Verify final equity appears in the text
        final_eq = result.metrics.final_equity
        assert f"${final_eq:,.2f}" in text

    def test_report_self_contained(self, result, tmp_path):
        """Report file is self-contained (single HTML with all charts)."""
        path = generate_report(result, output_dir=tmp_path)
        html = path.read_text(encoding="utf-8")

        # All chart sections present
        assert "Equity Curve" in html
        assert "Monthly Returns" in html
        assert "Trade Results" in html or "R-Multiple" in html
        assert "Trade Log" in html
