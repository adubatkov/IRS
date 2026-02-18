"""Backtest engine: execution, position management, and performance analytics."""

from engine.backtester import run_backtest, Backtester, BacktestResult
from engine.portfolio import Portfolio
from engine.trade_log import TradeLog, TradeRecord
from engine.metrics import compute_metrics, MetricsResult
from engine.events import EventLog, EventType

__all__ = [
    "run_backtest",
    "Backtester",
    "BacktestResult",
    "Portfolio",
    "TradeLog",
    "TradeRecord",
    "compute_metrics",
    "MetricsResult",
    "EventLog",
    "EventType",
]
