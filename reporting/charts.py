"""Plotly chart generators for backtest reporting.

Each function accepts a BacktestResult and returns a go.Figure object
ready for display or export. All charts use the plotly_dark template
with the project's standard color palette.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.backtester import BacktestResult
from engine.metrics import compute_drawdown

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLOR_GREEN = "#26a69a"
COLOR_RED = "#ef5350"
COLOR_GRAY = "#888888"
COLOR_WHITE = "#ffffff"
COLOR_YELLOW = "#ffd54f"
TEMPLATE = "plotly_dark"

_OUTCOME_COLORS = {"WIN": COLOR_GREEN, "LOSS": COLOR_RED, "BREAKEVEN": COLOR_GRAY}


def _empty_figure(message: str, title: str = "") -> go.Figure:
    """Return an empty figure with a centered annotation."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14))
    fig.update_layout(template=TEMPLATE, title=title)
    return fig


# ---------------------------------------------------------------------------
# 1. Equity Curve & Drawdown
# ---------------------------------------------------------------------------

def create_equity_curve_chart(result: BacktestResult) -> go.Figure:
    """Equity curve (top) with drawdown percentage (bottom).

    Uses two vertically stacked subplots with a shared x-axis.
    NaN values in the equity curve are filtered out before plotting.
    """
    equity = result.equity_curve
    timestamps = result.timestamps

    # Filter NaN
    valid_mask = ~np.isnan(equity)
    valid_equity = equity[valid_mask]
    valid_ts = timestamps[valid_mask]

    if len(valid_equity) == 0:
        return _empty_figure("No equity data available", "Equity Curve & Drawdown")

    # Compute drawdown series using the full curve (compute_drawdown handles NaN internally)
    dd_series, _, _ = compute_drawdown(equity)
    valid_dd = dd_series[valid_mask]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity", "Drawdown %"),
    )

    # Top subplot: equity line
    fig.add_trace(
        go.Scatter(
            x=valid_ts,
            y=valid_equity,
            mode="lines",
            name="Equity",
            line=dict(color=COLOR_GREEN, width=1.5),
        ),
        row=1, col=1,
    )

    # Bottom subplot: drawdown as filled area (values are negative)
    fig.add_trace(
        go.Scatter(
            x=valid_ts,
            y=valid_dd * 100,  # convert to percentage
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color=COLOR_RED, width=1),
            fillcolor="rgba(239, 83, 80, 0.3)",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title="Equity Curve & Drawdown",
        height=600,
        template=TEMPLATE,
        showlegend=False,
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# 2. Monthly Returns Heatmap
# ---------------------------------------------------------------------------

_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def create_monthly_heatmap(result: BacktestResult) -> go.Figure:
    """Year x Month heatmap of monthly returns.

    Reads ``result.metrics.monthly_returns`` (DataFrame with columns
    ``month``, ``return_pct``, ``trade_count``).  The ``month`` column
    contains strings in ``"YYYY-MM"`` format.
    """
    monthly = result.metrics.monthly_returns

    if monthly is None or len(monthly) == 0:
        return _empty_figure("No monthly return data available", "Monthly Returns (%)")

    # Parse year / month from "YYYY-MM" strings
    monthly = monthly.copy()
    monthly["year"] = monthly["month"].str[:4].astype(int)
    monthly["month_num"] = monthly["month"].str[5:7].astype(int)

    years = sorted(monthly["year"].unique())
    months = list(range(1, 13))

    # Build grid (rows = years, cols = months 1..12)
    z_values: list[list[float | None]] = []
    annotations: list[dict] = []

    for y_idx, year in enumerate(years):
        row: list[float | None] = []
        for m_idx, m in enumerate(months):
            match = monthly[(monthly["year"] == year) & (monthly["month_num"] == m)]
            if len(match) > 0:
                val = float(match["return_pct"].iloc[0])
                row.append(val)
                annotations.append(dict(
                    x=_MONTH_LABELS[m_idx],
                    y=str(year),
                    text=f"{val:.1f}%",
                    showarrow=False,
                    font=dict(color=COLOR_WHITE, size=11),
                ))
            else:
                row.append(None)
        z_values.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=_MONTH_LABELS,
            y=[str(y) for y in years],
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z_values],
            texttemplate="%{text}",
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Monthly Returns (%)",
        template=TEMPLATE,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Trade Scatter (R-Multiple over Time)
# ---------------------------------------------------------------------------

def create_trade_scatter(result: BacktestResult) -> go.Figure:
    """Scatter plot of trades: entry_time vs r_multiple, colored by outcome."""
    trades = result.trade_log

    if trades is None or len(trades) == 0:
        return _empty_figure("No trades to display", "Trade Results (R-Multiple)")

    colors = trades["outcome"].map(_OUTCOME_COLORS).fillna(COLOR_GRAY)
    sizes = trades["r_multiple"].abs().apply(lambda v: max(v * 5, 5))

    hover_text = trades.apply(
        lambda r: (
            f"Entry: {r.get('entry_price', 'N/A')}<br>"
            f"Exit: {r.get('exit_price', 'N/A')}<br>"
            f"Duration: {r.get('duration_bars', 'N/A')} bars<br>"
            f"POI: {r.get('poi_id', 'N/A')}"
        ),
        axis=1,
    )

    fig = go.Figure(
        data=go.Scatter(
            x=trades["entry_time"],
            y=trades["r_multiple"],
            mode="markers",
            marker=dict(
                color=colors,
                size=sizes,
                line=dict(width=0.5, color=COLOR_WHITE),
                opacity=0.8,
            ),
            text=hover_text,
            hovertemplate="%{text}<br>R: %{y:.2f}<extra></extra>",
        )
    )

    # Zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_WHITE, opacity=0.4)

    fig.update_layout(
        title="Trade Results (R-Multiple)",
        xaxis_title="Entry Time",
        yaxis_title="R-Multiple",
        template=TEMPLATE,
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------------
# 4. R-Multiple Distribution
# ---------------------------------------------------------------------------

def create_r_distribution(result: BacktestResult) -> go.Figure:
    """Histogram of R-multiple values with green/red coloring by sign."""
    trades = result.trade_log

    if trades is None or len(trades) == 0:
        return _empty_figure("No trades to display", "R-Multiple Distribution")

    r_values: np.ndarray = np.asarray(trades["r_multiple"].dropna().values, dtype=float)

    if len(r_values) == 0:
        return _empty_figure("No R-multiple data", "R-Multiple Distribution")

    # Compute histogram bins manually so we can color by sign
    counts, bin_edges = np.histogram(r_values, bins="auto")
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_colors = [COLOR_GREEN if mid >= 0 else COLOR_RED for mid in bin_midpoints]

    fig = go.Figure(
        data=go.Bar(
            x=bin_midpoints,
            y=counts,
            width=np.diff(bin_edges),
            marker_color=bar_colors,
            opacity=0.85,
            hovertemplate="R: %{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )

    # Vertical reference lines
    fig.add_vline(x=0, line_dash="dash", line_color=COLOR_WHITE, opacity=0.6)

    expectancy = result.metrics.expectancy
    if expectancy != 0:
        fig.add_vline(
            x=expectancy,
            line_dash="dash",
            line_color=COLOR_YELLOW,
            opacity=0.8,
            annotation_text=f"E[R]={expectancy:.2f}",
            annotation_font_color=COLOR_YELLOW,
        )

    fig.update_layout(
        title="R-Multiple Distribution",
        xaxis_title="R-Multiple",
        yaxis_title="Count",
        template=TEMPLATE,
        bargap=0.05,
    )

    return fig


# ---------------------------------------------------------------------------
# 5. MAE vs MFE Scatter
# ---------------------------------------------------------------------------

def create_mae_mfe_scatter(result: BacktestResult) -> go.Figure:
    """Scatter: max adverse excursion vs max favorable excursion."""
    trades = result.trade_log

    if trades is None or len(trades) == 0:
        return _empty_figure("No trades to display", "MAE vs MFE")

    colors = trades["outcome"].map(_OUTCOME_COLORS).fillna(COLOR_GRAY)

    fig = go.Figure(
        data=go.Scatter(
            x=trades["max_adverse_excursion"],
            y=trades["max_favorable_excursion"],
            mode="markers",
            marker=dict(
                color=colors,
                size=8,
                line=dict(width=0.5, color=COLOR_WHITE),
                opacity=0.8,
            ),
            hovertemplate="MAE: %{x:.4f}<br>MFE: %{y:.4f}<extra></extra>",
        )
    )

    # Diagonal reference line where MAE == MFE
    mae_vals = trades["max_adverse_excursion"].values
    mfe_vals = trades["max_favorable_excursion"].values
    all_vals = np.concatenate([mae_vals, mfe_vals])
    if len(all_vals) > 0:
        line_max = float(np.nanmax(all_vals)) * 1.1
        fig.add_trace(
            go.Scatter(
                x=[0, line_max],
                y=[0, line_max],
                mode="lines",
                line=dict(color=COLOR_WHITE, dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="MAE vs MFE",
        xaxis_title="Max Adverse Excursion",
        yaxis_title="Max Favorable Excursion",
        template=TEMPLATE,
        showlegend=False,
    )

    return fig
