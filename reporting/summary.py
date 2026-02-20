"""Console text summary for backtest results."""


from engine.backtester import BacktestResult
from engine.metrics import MetricsResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header_bar() -> str:
    """Return a full-width '=' border line."""
    return "=" * WIDTH


def _section_divider(label: str) -> str:
    """Return a section divider like: -- Label ------ (padded to WIDTH)."""
    prefix = f"-- {label} "
    remaining = WIDTH - len(prefix)
    return prefix + "-" * max(remaining, 0)


def _fmt_money(value: float) -> str:
    """Format as $X,XXX.XX."""
    return f"${value:,.2f}"


def _fmt_pct(value: float, decimals: int = 2) -> str:
    """Format as X.XX%."""
    return f"{value:.{decimals}f}%"


def _fmt_rr(value: float) -> str:
    """Format as X.XXR."""
    return f"{value:.2f}R"


def _fmt_ratio(value: float) -> str:
    """Format a ratio with 2 decimal places."""
    return f"{value:.2f}"


def _row(left_label: str, left_val: str, right_label: str = "",
         right_val: str = "") -> str:
    """Build a two-column row.

    Layout:
      "  {left_label:<17}{left_val:<14}{right_label:<17}{right_val}"
    """
    left = f"  {left_label:<17}{left_val:<14}"
    if right_label:
        right = f"{right_label:<17}{right_val}"
    else:
        right = ""
    return left + right


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_sync_stats(sync_stats: dict) -> str:
    """Format the sync-mode breakdown section.

    Parameters
    ----------
    sync_stats : dict
        Mapping of mode name to stats dict with keys:
        ``trades``, ``win_rate``, ``avg_rr``, ``profit_factor``.

    Returns
    -------
    str
        Formatted text block, or empty string if *sync_stats* is empty.
    """
    if not sync_stats:
        return ""

    lines: list[str] = []
    lines.append(_section_divider("Sync Mode Breakdown"))

    for mode, stats in sync_stats.items():
        trades = stats.get("trades", 0)
        wr = stats.get("win_rate", 0.0)
        avg_rr = stats.get("avg_rr", 0.0)
        pf = stats.get("profit_factor", 0.0)
        line = (
            f"  {mode:<9}{trades} trades | "
            f"{wr:.1f}% WR | "
            f"{avg_rr:.2f} avg R | "
            f"PF {pf:.2f}"
        )
        lines.append(line)

    return "\n".join(lines)


def format_metrics_table(metrics: MetricsResult) -> str:
    """Format key metrics as an aligned text block.

    Suitable for embedding inside an HTML ``<pre>`` tag.
    """
    lines: list[str] = []

    # Returns
    lines.append(_section_divider("Returns"))
    lines.append(_row("Total Return:", _fmt_pct(metrics.total_return_pct),
                       "Final Equity:", _fmt_money(metrics.final_equity)))
    lines.append(_row("CAGR:", _fmt_pct(metrics.cagr_pct),
                       "Peak Equity:", _fmt_money(metrics.peak_equity)))

    # Risk
    lines.append(_section_divider("Risk"))
    lines.append(_row("Max Drawdown:", f"-{_fmt_pct(metrics.max_drawdown_pct)}",
                       "Max DD Duration:",
                       f"{metrics.max_drawdown_duration_bars} bars"))
    lines.append(_row("Sharpe Ratio:", _fmt_ratio(metrics.sharpe_ratio),
                       "Sortino Ratio:", _fmt_ratio(metrics.sortino_ratio)))
    lines.append(_row("Calmar Ratio:", _fmt_ratio(metrics.calmar_ratio)))

    # Trades
    lines.append(_section_divider("Trades"))
    lines.append(_row("Total:", str(metrics.total_trades),
                       "Win Rate:", _fmt_pct(metrics.win_rate_pct, 1)))
    lines.append(_row("Winners:", str(metrics.winning_trades),
                       "Losers:", str(metrics.losing_trades)))
    lines.append(_row("Breakeven:", str(metrics.breakeven_trades),
                       "Avg R-Multiple:", _fmt_rr(metrics.avg_rr)))

    # Trade Quality
    lines.append(_section_divider("Trade Quality"))
    lines.append(_row("Avg Win:", _fmt_rr(metrics.avg_win_rr),
                       "Avg Loss:", _fmt_rr(metrics.avg_loss_rr)))
    lines.append(_row("Profit Factor:", _fmt_ratio(metrics.profit_factor),
                       "Expectancy:", _fmt_rr(metrics.expectancy)))

    # Duration
    lines.append(_section_divider("Duration"))
    lines.append(_row("Avg Trade:",
                       f"{metrics.avg_trade_duration_bars} bars",
                       "Avg Win:",
                       f"{metrics.avg_win_duration_bars} bars"))
    lines.append(_row("Avg Loss:",
                       f"{metrics.avg_loss_duration_bars} bars"))

    return "\n".join(lines)


def print_summary(result: BacktestResult) -> str:
    """Print a formatted backtest summary to the console and return the text.

    Parameters
    ----------
    result : BacktestResult
        Complete output from a backtest run.

    Returns
    -------
    str
        The full summary text that was printed.
    """
    metrics = result.metrics
    ts = result.timestamps
    config = result.config
    lines: list[str] = []

    # Title
    lines.append(_header_bar())
    lines.append("BACKTEST RESULTS SUMMARY".center(WIDTH))
    lines.append(_header_bar())
    lines.append("")

    # Period info
    start_str = ts[0].strftime("%Y-%m-%d %H:%M") if len(ts) > 0 else "N/A"
    end_str = ts[-1].strftime("%Y-%m-%d %H:%M") if len(ts) > 0 else "N/A"
    lines.append(_row("Period:", f"{start_str} -- {end_str}"))
    lines.append(_row("Data bars:", str(len(ts))))
    lines.append(_row("Initial capital:",
                       _fmt_money(config.backtest.initial_capital)))
    lines.append("")

    if metrics.total_trades == 0:
        # No trades path
        lines.append(_section_divider("Returns"))
        lines.append(_row("Total Return:", _fmt_pct(metrics.total_return_pct),
                           "Final Equity:", _fmt_money(metrics.final_equity)))
        lines.append(_row("CAGR:", _fmt_pct(metrics.cagr_pct),
                           "Peak Equity:", _fmt_money(metrics.peak_equity)))
        lines.append("")
        lines.append("  No trades executed.")
    else:
        # Full metrics table
        lines.append(format_metrics_table(metrics))

        # Sync stats
        sync_block = format_sync_stats(metrics.sync_stats)
        if sync_block:
            lines.append("")
            lines.append(sync_block)

    lines.append("")
    lines.append(_header_bar())

    text = "\n".join(lines)
    print(text)
    return text
