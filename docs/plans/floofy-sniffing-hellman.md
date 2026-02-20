# Phase 5: Reporting & Analysis

## Context

Phases 1-4 are complete (467 tests pass). `run_backtest(config, df_1m)` returns a `BacktestResult` with trade_log, equity_curve, metrics, signals, events, timestamps. Phase 5 builds the **reporting layer** that presents these results: console summary, Plotly charts, trade table, and a self-contained HTML report.

No modifications to Phases 1-4 are required.

## Architecture

```
engine/backtester.py
       │
       ▼
  BacktestResult ──┐
                   │
  reporting/       │
  ├── summary.py   │  print_summary(result) → console text
  ├── charts.py    │  create_*() → go.Figure objects
  ├── html_report.py  generate_report(result, output_dir) → files + console
  └── __init__.py     Public API: generate_report, print_summary
```

**Entry point**: `generate_report(result, output_dir)` writes files + prints summary.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chart library | Plotly | Already used in `visualization/chart.py`; consistent dark theme |
| HTML report | Self-contained single file | Plotly CDN embedded; easy to share |
| Console summary | Rich-text with sections | Quick scan of key metrics without opening browser |
| Chart theme | `plotly_dark` | Matches existing `visualization/chart.py` style |
| Color palette | Green=#26a69a, Red=#ef5350 | Matches existing chart colors |
| Output structure | `{output_dir}/report.html` + individual PNGs | HTML for sharing, PNGs for embedding |

## Build Order

```
Step 1: reporting/charts.py      (~180 LOC) - Pure Plotly chart functions
Step 2: reporting/summary.py     (~120 LOC) - Console text formatting
Step 3: reporting/html_report.py (~150 LOC) - HTML assembly + file I/O
Step 4: reporting/__init__.py    (~15 LOC)  - Public API exports
Step 5: Tests + integration
```

Steps 1 and 2 are independent and can be built in parallel.

---

## Step 1: `reporting/charts.py` -- Plotly Chart Generators

Pure functions that accept BacktestResult (or its components) and return `go.Figure` objects.

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.backtester import BacktestResult
from engine.metrics import MetricsResult


def create_equity_curve_chart(result: BacktestResult) -> go.Figure:
    """Equity curve with drawdown overlay (dual y-axis).

    Top: equity line (timestamps vs equity_curve).
    Bottom subplot: drawdown % as filled area (red).
    Uses plotly_dark theme, green line for equity.
    """

def create_monthly_heatmap(result: BacktestResult) -> go.Figure:
    """Monthly returns heatmap (Year rows x Month columns).

    Uses result.metrics.monthly_returns DataFrame (month, return_pct, trade_count).
    Green for positive, red for negative returns.
    Annotate cells with return % value.
    Handle edge case: monthly_returns is None → empty figure with message.
    """

def create_trade_scatter(result: BacktestResult) -> go.Figure:
    """Entry/exit scatter on timeline.

    X-axis: entry_time, Y-axis: r_multiple.
    Color: green for WIN, red for LOSS, gray for BREAKEVEN.
    Size proportional to abs(r_multiple).
    Hover: entry price, exit price, duration, poi_id.
    """

def create_r_distribution(result: BacktestResult) -> go.Figure:
    """R-multiple histogram.

    Bins of R values from trade_log['r_multiple'].
    Green bars for positive R, red for negative.
    Vertical line at 0 and at expectancy.
    """

def create_mae_mfe_scatter(result: BacktestResult) -> go.Figure:
    """MAE vs MFE scatter for trade quality analysis.

    X-axis: max_adverse_excursion, Y-axis: max_favorable_excursion.
    Color by outcome. Helps visualize entry quality.
    Diagonal line where MAE == MFE.
    """
```

Reuses: `plotly_dark` template and color palette from `visualization/chart.py`.

**Test**: `tests/unit/test_charts.py` (~8 tests) -- each chart returns go.Figure, handles empty trade_log, handles None monthly_returns.

---

## Step 2: `reporting/summary.py` -- Console Text Output

Formatted text summary printed to stdout.

```python
from engine.backtester import BacktestResult
from engine.metrics import MetricsResult


def print_summary(result: BacktestResult) -> str:
    """Print formatted backtest summary to console and return the text.

    Sections:
    1. Header: date range, data bars, initial capital
    2. Returns: total return %, CAGR %, final equity, peak equity
    3. Risk: max drawdown %, max DD duration, Sharpe, Sortino, Calmar
    4. Trades: total, win/loss/BE counts, win rate %, avg RR
    5. Trade Quality: avg win RR, avg loss RR, profit factor, expectancy
    6. Duration: avg trade/win/loss duration in bars
    7. Sync Stats: per-mode breakdown (if any)

    Uses simple text formatting with aligned columns.
    Returns the formatted string (for HTML embedding).
    """

def format_metrics_table(metrics: MetricsResult) -> str:
    """Format key metrics as aligned text table."""

def format_sync_stats(sync_stats: dict) -> str:
    """Format per sync-mode breakdown."""
```

**Test**: `tests/unit/test_summary.py` (~5 tests) -- returns non-empty string, includes key sections, handles zero trades, handles no sync stats.

---

## Step 3: `reporting/html_report.py` -- HTML Assembly & File I/O

Combines charts + summary into a self-contained HTML file.

```python
import os
from pathlib import Path
from typing import Optional

import plotly.io as pio

from engine.backtester import BacktestResult
from reporting.charts import (
    create_equity_curve_chart,
    create_monthly_heatmap,
    create_trade_scatter,
    create_r_distribution,
    create_mae_mfe_scatter,
)
from reporting.summary import print_summary, format_metrics_table


def generate_report(
    result: BacktestResult,
    output_dir: str | Path = "output",
    open_browser: bool = False,
) -> Path:
    """Generate complete backtest report.

    1. Create output_dir if needed
    2. Print console summary (print_summary)
    3. Generate all charts
    4. Build trade table HTML from trade_log DataFrame
    5. Assemble into single HTML with embedded Plotly JS
    6. Write to {output_dir}/report.html
    7. Optionally open in browser
    8. Return path to report
    """

def _build_html(
    summary_text: str,
    charts: list[tuple[str, str]],  # (title, plotly_html_div)
    trade_table_html: str,
    metrics: "MetricsResult",
) -> str:
    """Build self-contained HTML document.

    Uses Plotly CDN for JS. Dark theme CSS inline.
    Sections: Summary → Charts → Trade Table.
    """

def _trade_log_to_html(trade_df: "pd.DataFrame") -> str:
    """Convert trade DataFrame to styled HTML table.

    Columns: #, Direction, Entry Time, Entry Price, Exit Time, Exit Price,
             R-Multiple, PnL, Duration, Outcome, Sync Mode.
    Color rows by outcome (green/red/gray).
    Limit to 200 rows max with "showing X of Y" note.
    """
```

**Test**: `tests/unit/test_html_report.py` (~5 tests) -- generate_report creates file, HTML contains expected sections, handles zero trades, output_dir created if missing.

---

## Step 4: `reporting/__init__.py` -- Public API

```python
from reporting.html_report import generate_report
from reporting.summary import print_summary
```

---

## Step 5: Tests + Integration

**Unit tests** (created alongside each step):
- `tests/unit/test_charts.py` (~8 tests)
- `tests/unit/test_summary.py` (~5 tests)
- `tests/unit/test_html_report.py` (~5 tests)

**Integration test**: `tests/integration/test_reporting_e2e.py` (~3 tests)
- Full pipeline: `run_backtest()` → `generate_report()` produces valid HTML
- `print_summary()` returns non-empty text with correct metric values
- Report file is self-contained (single HTML with embedded charts)

---

## File Summary

| Step | Source File | Test File | Depends On |
|------|------------|-----------|------------|
| 1 | `reporting/charts.py` | `tests/unit/test_charts.py` | engine.backtester (BacktestResult) |
| 2 | `reporting/summary.py` | `tests/unit/test_summary.py` | engine.backtester, engine.metrics |
| 3 | `reporting/html_report.py` | `tests/unit/test_html_report.py` | charts.py, summary.py |
| 4 | `reporting/__init__.py` | -- | all reporting |
| 5 | -- | `tests/integration/test_reporting_e2e.py` | all reporting + engine |

**Total: 4 source files + 4 test files = ~8 files, ~465 LOC source + ~270 LOC tests**

## Critical Files to Read Before Implementation

- `C:\Trading\IRS\engine\backtester.py` -- BacktestResult dataclass
- `C:\Trading\IRS\engine\metrics.py` -- MetricsResult dataclass, all metric fields
- `C:\Trading\IRS\visualization\chart.py` -- Existing Plotly patterns, dark theme, colors
- `C:\Trading\IRS\tests\conftest.py` -- make_trending_1m() for test data

## Verification

After each step:
1. `pytest tests/unit/test_{module}.py -v` -- step tests pass
2. `ruff check reporting/` -- clean

After all steps:
1. `pytest tests/ -v` -- all tests pass (467 existing + ~21 new)
2. `python -c "from reporting import generate_report; print('OK')"` -- import works
3. Quick smoke test:
   ```python
   from config import Config
   from engine import run_backtest
   from reporting import generate_report
   from tests.conftest import make_trending_1m

   result = run_backtest(Config(), make_trending_1m(600))
   path = generate_report(result, "output")
   # Open output/report.html in browser → verify charts render
   ```
