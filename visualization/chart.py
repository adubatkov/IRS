"""Core charting utilities for the IRS backtesting system.

Provides plotly-based candlestick charts with timeframe selection,
date range filtering, and overlay support.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_chart(
    df: pd.DataFrame,
    title: str = "",
    height: int = 700,
    show_volume: bool = True,
    range_start: str | None = None,
    range_end: str | None = None,
) -> go.Figure:
    """Create an interactive candlestick chart.

    Args:
        df: DataFrame with 'time', 'open', 'high', 'low', 'close' columns.
        title: Chart title.
        height: Chart height in pixels.
        show_volume: Whether to show volume subplot.
        range_start: Optional start date filter (ISO format string).
        range_end: Optional end date filter (ISO format string).

    Returns:
        Plotly Figure object.
    """
    data = df.copy()

    # Apply date range filter
    if range_start:
        data = data[data["time"] >= pd.Timestamp(range_start, tz="UTC")]
    if range_end:
        data = data[data["time"] <= pd.Timestamp(range_end, tz="UTC")]

    if len(data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data in selected range", showarrow=False)
        return fig

    has_volume = show_volume and "tick_volume" in data.columns and data["tick_volume"].sum() > 0

    if has_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data["time"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Volume bars
    if has_volume:
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(data["open"], data["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=data["time"],
                y=data["tick_volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=False,
    )

    # Add range selector buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=5, label="5D", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ])
        ),
        row=1, col=1,
    )

    return fig


def multi_timeframe_chart(
    data_by_tf: dict[str, pd.DataFrame],
    symbol: str = "",
    height_per_chart: int = 400,
) -> go.Figure:
    """Create a grid of candlestick charts for multiple timeframes.

    Args:
        data_by_tf: Dict mapping timeframe name to DataFrame.
        symbol: Instrument symbol for title.
        height_per_chart: Height per chart row.

    Returns:
        Plotly Figure with subplots.
    """
    n = len(data_by_tf)
    if n == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data provided", showarrow=False)
        return fig

    tf_names = list(data_by_tf.keys())

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.02,
        subplot_titles=[f"{symbol} {tf}" for tf in tf_names],
    )

    for i, (tf, df) in enumerate(data_by_tf.items(), start=1):
        fig.add_trace(
            go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=tf,
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=i, col=1,
        )
        fig.update_xaxes(rangeslider_visible=False, row=i, col=1)

    fig.update_layout(
        height=height_per_chart * n,
        template="plotly_dark",
        showlegend=False,
        title=f"{symbol} Multi-Timeframe View" if symbol else "Multi-Timeframe View",
    )

    return fig


def add_markers(
    fig: go.Figure,
    times: list,
    prices: list,
    name: str = "Markers",
    symbol: str = "triangle-up",
    color: str = "yellow",
    size: int = 10,
    row: int = 1,
) -> go.Figure:
    """Add scatter markers to a chart (for swing highs/lows, entries, etc.)."""
    fig.add_trace(
        go.Scatter(
            x=times,
            y=prices,
            mode="markers",
            name=name,
            marker=dict(symbol=symbol, color=color, size=size),
        ),
        row=row, col=1,
    )
    return fig


def add_horizontal_line(
    fig: go.Figure,
    y: float,
    color: str = "white",
    dash: str = "dash",
    name: str = "",
    row: int = 1,
) -> go.Figure:
    """Add a horizontal line to a chart."""
    fig.add_hline(y=y, line_color=color, line_dash=dash, annotation_text=name, row=row)
    return fig


def add_zone(
    fig: go.Figure,
    x0: Any,
    x1: Any,
    y0: float,
    y1: float,
    color: str = "rgba(0, 255, 0, 0.1)",
    name: str = "",
    row: int = 1,
) -> go.Figure:
    """Add a rectangular zone (for FVG, OB, etc.)."""
    fig.add_shape(
        type="rect",
        x0=x0, x1=x1, y0=y0, y1=y1,
        fillcolor=color,
        line=dict(width=0),
        name=name,
        row=row, col=1,
    )
    return fig
