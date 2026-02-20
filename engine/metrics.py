"""Performance analytics for backtest results."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MetricsResult:
    """Complete performance metrics from a backtest run."""

    # Returns
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate_pct: float = 0.0
    avg_rr: float = 0.0
    avg_win_rr: float = 0.0
    avg_loss_rr: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Duration
    avg_trade_duration_bars: int = 0
    avg_win_duration_bars: int = 0
    avg_loss_duration_bars: int = 0

    # Per sync mode
    sync_stats: dict = field(default_factory=dict)

    # Monthly breakdown
    monthly_returns: Optional[pd.DataFrame] = None

    # Equity
    final_equity: float = 0.0
    peak_equity: float = 0.0


def compute_drawdown(equity_curve: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Compute drawdown series, max drawdown %, max duration in bars.

    Returns: (drawdown_series, max_dd_pct, max_dd_duration_bars)
    """
    # Filter out NaN
    valid = equity_curve[~np.isnan(equity_curve)]
    if len(valid) < 2:
        return np.zeros_like(equity_curve), 0.0, 0

    peak = np.maximum.accumulate(valid)
    dd = np.where(peak > 0, (valid - peak) / peak, 0.0)
    max_dd_pct = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Max duration: longest streak below peak
    max_duration = 0
    current_duration = 0
    for i in range(len(valid)):
        if valid[i] < peak[i]:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    # Pad drawdown back to original length
    full_dd = np.full_like(equity_curve, np.nan)
    valid_mask = ~np.isnan(equity_curve)
    full_dd[valid_mask] = dd

    return full_dd, max_dd_pct, max_duration


def compute_sharpe(
    equity_curve: np.ndarray,
    bars_per_year: float = 252 * 390,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio from bar-by-bar equity returns."""
    valid = equity_curve[~np.isnan(equity_curve)]
    if len(valid) < 2:
        return 0.0

    returns = np.diff(valid) / valid[:-1]
    if len(returns) == 0:
        return 0.0

    excess = returns - risk_free_rate / bars_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0

    return float(np.mean(excess) / std * np.sqrt(bars_per_year))


def compute_sortino(
    equity_curve: np.ndarray,
    bars_per_year: float = 252 * 390,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    valid = equity_curve[~np.isnan(equity_curve)]
    if len(valid) < 2:
        return 0.0

    returns = np.diff(valid) / valid[:-1]
    if len(returns) == 0:
        return 0.0

    excess = returns - risk_free_rate / bars_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0  # No downside = undefined, return 0

    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0

    return float(np.mean(excess) / downside_std * np.sqrt(bars_per_year))


def compute_calmar(cagr: float, max_drawdown_pct: float) -> float:
    """Calmar ratio = CAGR / |max drawdown|."""
    if max_drawdown_pct == 0:
        return 0.0
    return cagr / max_drawdown_pct


def compute_return_metrics(
    equity_curve: np.ndarray,
    initial_capital: float,
    bars_per_year: float = 252 * 390,
) -> dict:
    """Total return, CAGR from equity curve."""
    valid = equity_curve[~np.isnan(equity_curve)]
    if len(valid) == 0:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0}

    final = valid[-1]
    total_return = (final - initial_capital) / initial_capital * 100

    # CAGR
    n_bars = len(valid)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1.0
    if years <= 0 or years < 0.001 or initial_capital <= 0 or final <= 0:
        cagr = 0.0
    else:
        try:
            cagr = ((final / initial_capital) ** (1 / years) - 1) * 100
        except (OverflowError, FloatingPointError):
            cagr = 0.0

    return {"total_return_pct": total_return, "cagr_pct": cagr}


def compute_trade_stats(trade_df: pd.DataFrame) -> dict:
    """Win rate, avg RR, profit factor, expectancy from trade DataFrame."""
    if len(trade_df) == 0:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "breakeven_trades": 0, "win_rate_pct": 0.0, "avg_rr": 0.0,
            "avg_win_rr": 0.0, "avg_loss_rr": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_trade_duration_bars": 0,
            "avg_win_duration_bars": 0, "avg_loss_duration_bars": 0,
        }

    # Only closed trades
    closed = trade_df[trade_df["outcome"].isin(["WIN", "LOSS", "BREAKEVEN"])]
    if len(closed) == 0:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "breakeven_trades": 0, "win_rate_pct": 0.0, "avg_rr": 0.0,
            "avg_win_rr": 0.0, "avg_loss_rr": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_trade_duration_bars": 0,
            "avg_win_duration_bars": 0, "avg_loss_duration_bars": 0,
        }

    total = len(closed)
    winners = closed[closed["outcome"] == "WIN"]
    losers = closed[closed["outcome"] == "LOSS"]
    breakevens = closed[closed["outcome"] == "BREAKEVEN"]

    win_rate = len(winners) / total * 100 if total > 0 else 0.0

    avg_rr = float(closed["r_multiple"].mean()) if total > 0 else 0.0
    avg_win_rr = float(winners["r_multiple"].mean()) if len(winners) > 0 else 0.0
    avg_loss_rr = float(losers["r_multiple"].mean()) if len(losers) > 0 else 0.0

    gross_profit = float(winners["realized_pnl"].sum()) if len(winners) > 0 else 0.0
    gross_loss = abs(float(losers["realized_pnl"].sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss) in R terms
    loss_rate = len(losers) / total if total > 0 else 0.0
    win_rate_frac = len(winners) / total if total > 0 else 0.0
    expectancy = win_rate_frac * avg_win_rr + loss_rate * avg_loss_rr  # avg_loss_rr is negative

    avg_duration = int(closed["duration_bars"].mean()) if total > 0 else 0
    avg_win_duration = int(winners["duration_bars"].mean()) if len(winners) > 0 else 0
    avg_loss_duration = int(losers["duration_bars"].mean()) if len(losers) > 0 else 0

    return {
        "total_trades": total,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "breakeven_trades": len(breakevens),
        "win_rate_pct": win_rate,
        "avg_rr": avg_rr,
        "avg_win_rr": avg_win_rr,
        "avg_loss_rr": avg_loss_rr,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_trade_duration_bars": avg_duration,
        "avg_win_duration_bars": avg_win_duration,
        "avg_loss_duration_bars": avg_loss_duration,
    }


def compute_sync_mode_stats(trade_df: pd.DataFrame) -> dict:
    """Per sync-mode breakdown of trade statistics."""
    result = {}
    if len(trade_df) == 0 or "sync_mode" not in trade_df.columns:
        return result

    for mode in trade_df["sync_mode"].unique():
        mode_trades = trade_df[trade_df["sync_mode"] == mode]
        stats = compute_trade_stats(mode_trades)
        result[mode] = {
            "trades": stats["total_trades"],
            "win_rate": stats["win_rate_pct"],
            "avg_rr": stats["avg_rr"],
            "profit_factor": stats["profit_factor"],
        }

    return result


def compute_monthly_returns(
    trade_df: pd.DataFrame,
    equity_curve: np.ndarray,
    timestamps: pd.DatetimeIndex,
    initial_capital: float,
) -> pd.DataFrame:
    """Monthly return breakdown from equity curve."""
    valid_mask = ~np.isnan(equity_curve)
    if not valid_mask.any():
        return pd.DataFrame(columns=["month", "return_pct", "trade_count"])

    valid_equity = equity_curve[valid_mask]
    valid_timestamps = timestamps[valid_mask]

    if len(valid_equity) == 0:
        return pd.DataFrame(columns=["month", "return_pct", "trade_count"])

    # Create Series for resampling
    eq_series = pd.Series(valid_equity, index=valid_timestamps)
    # Use "ME" for pandas >= 2.2, fall back to "M" for older versions
    try:
        monthly_last = eq_series.resample("ME").last()
    except ValueError:
        monthly_last = eq_series.resample("M").last()

    rows = []
    prev_equity = initial_capital
    for month_end in monthly_last.index:
        end_val = monthly_last[month_end]
        if np.isnan(end_val) or prev_equity == 0:
            continue
        ret_pct = (end_val - prev_equity) / prev_equity * 100
        month_str = month_end.strftime("%Y-%m")

        # Count trades that exited in this month
        trade_count = 0
        if len(trade_df) > 0 and "exit_time" in trade_df.columns:
            month_trades = trade_df[
                (trade_df["exit_time"].notna())
                & (trade_df["exit_time"].dt.to_period("M") == month_end.to_period("M"))
            ]
            trade_count = len(month_trades)

        rows.append({"month": month_str, "return_pct": ret_pct, "trade_count": trade_count})
        prev_equity = end_val

    return pd.DataFrame(rows)


def compute_metrics(
    trade_df: pd.DataFrame,
    equity_curve: np.ndarray,
    initial_capital: float,
    bars_per_year: float = 252 * 390,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> MetricsResult:
    """Compute all performance metrics.

    Args:
        trade_df: From TradeLog.to_dataframe() (closed trades).
        equity_curve: numpy array from Portfolio, one value per bar.
        initial_capital: Starting capital.
        bars_per_year: For annualization.
        timestamps: DatetimeIndex for monthly breakdown.
    """
    # Return metrics
    ret = compute_return_metrics(equity_curve, initial_capital, bars_per_year)

    # Drawdown
    dd_series, max_dd, max_dd_dur = compute_drawdown(equity_curve)

    # Risk-adjusted
    sharpe = compute_sharpe(equity_curve, bars_per_year)
    sortino = compute_sortino(equity_curve, bars_per_year)
    calmar = compute_calmar(ret["cagr_pct"], max_dd * 100)

    # Trade stats
    trade_stats = compute_trade_stats(trade_df)

    # Sync mode stats
    sync_stats = compute_sync_mode_stats(trade_df)

    # Monthly returns
    monthly = None
    if timestamps is not None:
        monthly = compute_monthly_returns(trade_df, equity_curve, timestamps, initial_capital)

    # Equity info
    valid = equity_curve[~np.isnan(equity_curve)]
    final_eq = float(valid[-1]) if len(valid) > 0 else initial_capital
    peak_eq = float(valid.max()) if len(valid) > 0 else initial_capital

    return MetricsResult(
        total_return_pct=ret["total_return_pct"],
        cagr_pct=ret["cagr_pct"],
        max_drawdown_pct=max_dd * 100,
        max_drawdown_duration_bars=max_dd_dur,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=trade_stats["total_trades"],
        winning_trades=trade_stats["winning_trades"],
        losing_trades=trade_stats["losing_trades"],
        breakeven_trades=trade_stats["breakeven_trades"],
        win_rate_pct=trade_stats["win_rate_pct"],
        avg_rr=trade_stats["avg_rr"],
        avg_win_rr=trade_stats["avg_win_rr"],
        avg_loss_rr=trade_stats["avg_loss_rr"],
        profit_factor=trade_stats["profit_factor"],
        expectancy=trade_stats["expectancy"],
        avg_trade_duration_bars=trade_stats["avg_trade_duration_bars"],
        avg_win_duration_bars=trade_stats["avg_win_duration_bars"],
        avg_loss_duration_bars=trade_stats["avg_loss_duration_bars"],
        sync_stats=sync_stats,
        monthly_returns=monthly,
        final_equity=final_eq,
        peak_equity=peak_eq,
    )
