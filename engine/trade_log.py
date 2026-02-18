"""Trade journal for recording complete trade lifecycle."""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class TradeRecord:
    """Complete record of a single trade (entry through exit)."""

    # Identification
    trade_id: int
    poi_id: str
    direction: int  # +1 long, -1 short

    # Entry
    entry_time: pd.Timestamp
    entry_price: float  # After slippage
    entry_signal_price: float  # Before slippage
    position_size: float

    # Exit (filled when trade closes)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None  # After slippage
    exit_signal_price: Optional[float] = None  # Before slippage
    exit_reason: str = ""

    # P&L
    realized_pnl: float = 0.0
    commission_entry: float = 0.0
    commission_exit: float = 0.0
    gross_pnl: float = 0.0  # Before commission

    # Excursion tracking (updated bar-by-bar while open)
    max_favorable_excursion: float = 0.0  # MFE in price units
    max_adverse_excursion: float = 0.0  # MAE in price units

    # Context
    sync_mode: str = ""
    timeframe: str = ""
    confirmation_count: int = 0
    stop_loss: float = 0.0
    target: float = 0.0
    is_addon: bool = False
    parent_trade_id: Optional[int] = None

    # Classification (set on close)
    outcome: str = ""  # WIN, LOSS, BREAKEVEN
    r_multiple: float = 0.0
    duration_bars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def classify_outcome(realized_pnl: float, commission_total: float) -> str:
    """Classify trade as WIN, LOSS, or BREAKEVEN.

    BREAKEVEN: |realized_pnl| <= 2 * commission_total (within costs).
    """
    if commission_total <= 0:
        commission_total = 0.01  # Prevent division issues
    if abs(realized_pnl) <= 2 * commission_total:
        return "BREAKEVEN"
    if realized_pnl > 0:
        return "WIN"
    return "LOSS"


def compute_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    direction: int,
) -> float:
    """Compute R-multiple: how many R's the trade realized.

    R = (exit - entry) / (entry - SL) for long
    R = (entry - exit) / (SL - entry) for short
    """
    if direction == 1:
        risk = entry_price - stop_loss
        reward = exit_price - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - exit_price

    if risk <= 0:
        return 0.0
    return reward / risk


class TradeLog:
    """Accumulates TradeRecords during backtest."""

    def __init__(self) -> None:
        self._trades: list[TradeRecord] = []
        self._next_id: int = 0
        # Index for fast lookup of open trades
        self._open_trade_ids: set[int] = set()

    def open_trade(
        self,
        poi_id: str,
        direction: int,
        entry_time: pd.Timestamp,
        entry_price: float,
        entry_signal_price: float,
        position_size: float,
        stop_loss: float,
        target: float,
        commission: float,
        sync_mode: str = "",
        timeframe: str = "",
        confirmation_count: int = 0,
        is_addon: bool = False,
        parent_trade_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Record a new trade opening. Returns trade_id."""
        trade_id = self._next_id
        self._next_id += 1

        record = TradeRecord(
            trade_id=trade_id,
            poi_id=poi_id,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_signal_price=entry_signal_price,
            position_size=position_size,
            stop_loss=stop_loss,
            target=target,
            commission_entry=commission,
            sync_mode=sync_mode,
            timeframe=timeframe,
            confirmation_count=confirmation_count,
            is_addon=is_addon,
            parent_trade_id=parent_trade_id,
            metadata=metadata or {},
        )
        self._trades.append(record)
        self._open_trade_ids.add(trade_id)
        return trade_id

    def close_trade(
        self,
        trade_id: int,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_signal_price: float,
        exit_reason: str,
        commission: float,
        bar_count: int,
    ) -> TradeRecord:
        """Record trade closing. Returns completed TradeRecord."""
        record = self.get_trade(trade_id)

        record.exit_time = exit_time
        record.exit_price = exit_price
        record.exit_signal_price = exit_signal_price
        record.exit_reason = exit_reason
        record.commission_exit = commission
        record.duration_bars = bar_count

        # Compute P&L
        record.gross_pnl = record.direction * (exit_price - record.entry_price) * record.position_size
        total_commission = record.commission_entry + record.commission_exit
        record.realized_pnl = record.gross_pnl - total_commission

        # Classify
        record.outcome = classify_outcome(record.realized_pnl, total_commission)
        record.r_multiple = compute_r_multiple(
            record.entry_price, exit_price, record.stop_loss, record.direction
        )

        self._open_trade_ids.discard(trade_id)
        return record

    def update_excursion(
        self,
        trade_id: int,
        candle_high: float,
        candle_low: float,
    ) -> None:
        """Update MFE/MAE for an open trade based on current bar.

        For LONG: MFE = max(MFE, high - entry), MAE = max(MAE, entry - low)
        For SHORT: MFE = max(MFE, entry - low), MAE = max(MAE, high - entry)
        """
        record = self.get_trade(trade_id)
        if record.direction == 1:
            favorable = candle_high - record.entry_price
            adverse = record.entry_price - candle_low
        else:
            favorable = record.entry_price - candle_low
            adverse = candle_high - record.entry_price

        if favorable > record.max_favorable_excursion:
            record.max_favorable_excursion = favorable
        if adverse > record.max_adverse_excursion:
            record.max_adverse_excursion = adverse

    def get_open_trades(self) -> list[TradeRecord]:
        """Return trades that have no exit_time yet."""
        return [self._trades[tid] for tid in self._open_trade_ids]

    def get_trade(self, trade_id: int) -> TradeRecord:
        """Get trade by ID."""
        if trade_id < 0 or trade_id >= len(self._trades):
            raise KeyError(f"Trade {trade_id} not found")
        return self._trades[trade_id]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trades to DataFrame."""
        if not self._trades:
            return pd.DataFrame(columns=[
                "trade_id", "poi_id", "direction", "entry_time", "entry_price",
                "exit_time", "exit_price", "exit_reason", "realized_pnl",
                "gross_pnl", "commission_entry", "commission_exit",
                "max_favorable_excursion", "max_adverse_excursion",
                "sync_mode", "timeframe", "confirmation_count",
                "stop_loss", "target", "is_addon", "parent_trade_id",
                "outcome", "r_multiple", "duration_bars", "position_size",
            ])
        records = []
        for t in self._trades:
            records.append({
                "trade_id": t.trade_id,
                "poi_id": t.poi_id,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "entry_signal_price": t.entry_signal_price,
                "position_size": t.position_size,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "exit_signal_price": t.exit_signal_price,
                "exit_reason": t.exit_reason,
                "realized_pnl": t.realized_pnl,
                "gross_pnl": t.gross_pnl,
                "commission_entry": t.commission_entry,
                "commission_exit": t.commission_exit,
                "max_favorable_excursion": t.max_favorable_excursion,
                "max_adverse_excursion": t.max_adverse_excursion,
                "sync_mode": t.sync_mode,
                "timeframe": t.timeframe,
                "confirmation_count": t.confirmation_count,
                "stop_loss": t.stop_loss,
                "target": t.target,
                "is_addon": t.is_addon,
                "parent_trade_id": t.parent_trade_id,
                "outcome": t.outcome,
                "r_multiple": t.r_multiple,
                "duration_bars": t.duration_bars,
            })
        return pd.DataFrame(records)

    def to_csv(self, path: str) -> None:
        """Export trade log to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
