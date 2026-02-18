"""Position and equity management for backtesting."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import BacktestConfig, RiskConfig
from strategy.types import Signal, SyncMode
from strategy.risk import calculate_position_size
from engine.trade_log import TradeLog, TradeRecord
from engine.events import EventLog, EventType


@dataclass
class PositionInfo:
    """Lightweight tracking of an open position."""
    trade_id: int
    poi_id: str
    direction: int
    entry_price: float
    position_size: float
    stop_loss: float
    target: float
    entry_bar_index: int
    is_addon: bool = False
    parent_trade_id: Optional[int] = None


def apply_slippage(
    price: float,
    direction: int,
    is_entry: bool,
    slippage_pct: float,
) -> float:
    """Apply slippage to a fill price.

    Slippage always works against the trader:
    - LONG entry: price * (1 + slippage_pct)  -- buy higher
    - SHORT entry: price * (1 - slippage_pct) -- sell lower
    - LONG exit: price * (1 - slippage_pct)   -- sell lower
    - SHORT exit: price * (1 + slippage_pct)  -- buy higher
    """
    if is_entry:
        if direction == 1:
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)
    else:
        if direction == 1:
            return price * (1 - slippage_pct)
        else:
            return price * (1 + slippage_pct)


class Portfolio:
    """Manages positions, equity, and execution during backtest."""

    def __init__(
        self,
        backtest_config: BacktestConfig,
        risk_config: RiskConfig,
        n_bars: int,
        trade_log: TradeLog,
        event_log: Optional[EventLog] = None,
    ) -> None:
        self._bt_config = backtest_config
        self._risk_config = risk_config
        self._trade_log = trade_log
        self._event_log = event_log

        self._cash: float = backtest_config.initial_capital
        self._initial_capital: float = backtest_config.initial_capital

        # Pre-allocated equity curve
        self._equity_curve: np.ndarray = np.full(n_bars, np.nan, dtype=np.float64)

        # Open positions: poi_id -> list of PositionInfo
        self._positions: dict[str, list[PositionInfo]] = {}
        # Map trade_id -> poi_id for reverse lookup
        self._trade_to_poi: dict[int, str] = {}
        # Track last known close price for unrealized P&L
        self._last_close: float = 0.0

    @property
    def equity(self) -> float:
        """Current equity = cash + unrealized P&L of all open positions."""
        unrealized = 0.0
        for positions in self._positions.values():
            for pos in positions:
                unrealized += pos.direction * (self._last_close - pos.entry_price) * pos.position_size
        return self._cash + unrealized

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def open_position_count(self) -> int:
        """Count of distinct poi_ids with open positions."""
        return len(self._positions)

    def can_open_position(self) -> bool:
        """Check if below max_concurrent_positions."""
        return self.open_position_count < self._risk_config.max_concurrent_positions

    def open_position(
        self,
        signal: Signal,
        sync_mode: SyncMode,
        bar_index: int,
        timeframe: str = "",
        confirmation_count: int = 0,
        is_addon: bool = False,
        parent_trade_id: Optional[int] = None,
    ) -> Optional[int]:
        """Execute an entry signal.

        Returns trade_id or None if rejected.
        """
        # Check max positions (skip for add-ons to existing poi_id)
        if not is_addon and signal.poi_id not in self._positions:
            if not self.can_open_position():
                if self._event_log is not None:
                    self._event_log.emit(
                        EventType.POSITION_REJECTED, signal.timestamp,
                        signal.poi_id, reason="max_positions_reached"
                    )
                return None

        # Apply slippage
        fill_price = apply_slippage(
            signal.price, signal.direction, True, self._bt_config.slippage_pct
        )

        # Update last close to fill price for accurate equity calculation
        # (in real usage, update_mark_to_market sets this each bar)
        if self._last_close == 0.0:
            self._last_close = fill_price

        # Calculate position size
        size = calculate_position_size(
            account_equity=self.equity,
            entry_price=fill_price,
            stop_loss=signal.stop_loss,
            sync_mode=sync_mode,
            risk_config=self._risk_config,
        )

        if size <= 0:
            if self._event_log is not None:
                self._event_log.emit(
                    EventType.POSITION_REJECTED, signal.timestamp,
                    signal.poi_id, reason="zero_position_size"
                )
            return None

        # Entry commission
        commission = fill_price * size * self._bt_config.commission_pct
        self._cash -= commission

        # Record in trade log
        trade_id = self._trade_log.open_trade(
            poi_id=signal.poi_id,
            direction=signal.direction,
            entry_time=signal.timestamp,
            entry_price=fill_price,
            entry_signal_price=signal.price,
            position_size=size,
            stop_loss=signal.stop_loss,
            target=signal.target,
            commission=commission,
            sync_mode=sync_mode.value if hasattr(sync_mode, 'value') else str(sync_mode),
            timeframe=timeframe,
            confirmation_count=confirmation_count,
            is_addon=is_addon,
            parent_trade_id=parent_trade_id,
            metadata=signal.metadata,
        )

        # Track position
        pos_info = PositionInfo(
            trade_id=trade_id,
            poi_id=signal.poi_id,
            direction=signal.direction,
            entry_price=fill_price,
            position_size=size,
            stop_loss=signal.stop_loss,
            target=signal.target,
            entry_bar_index=bar_index,
            is_addon=is_addon,
            parent_trade_id=parent_trade_id,
        )

        if signal.poi_id not in self._positions:
            self._positions[signal.poi_id] = []
        self._positions[signal.poi_id].append(pos_info)
        self._trade_to_poi[trade_id] = signal.poi_id

        # Emit event
        if self._event_log is not None:
            event_type = EventType.ADDON if is_addon else EventType.ENTRY
            self._event_log.emit(
                event_type, signal.timestamp, signal.poi_id,
                trade_id=trade_id, fill_price=fill_price, size=size,
            )

        return trade_id

    def close_position(
        self,
        poi_id: str,
        exit_signal_price: float,
        exit_reason: str,
        timestamp: pd.Timestamp,
        bar_index: int,
        trade_id: Optional[int] = None,
    ) -> list[TradeRecord]:
        """Execute an exit. Close all positions for poi_id (or specific trade_id).

        Returns list of closed TradeRecords.
        """
        if poi_id not in self._positions:
            return []

        positions_to_close = self._positions[poi_id]
        if trade_id is not None:
            positions_to_close = [p for p in positions_to_close if p.trade_id == trade_id]

        closed_records = []
        for pos in positions_to_close:
            # Apply slippage to exit
            fill_price = apply_slippage(
                exit_signal_price, pos.direction, False, self._bt_config.slippage_pct
            )

            # Exit commission
            commission = fill_price * pos.position_size * self._bt_config.commission_pct

            # Add proceeds to cash: direction * (fill - entry) * size - exit_commission
            proceeds = pos.direction * (fill_price - pos.entry_price) * pos.position_size
            self._cash += proceeds - commission

            # Duration
            bar_count = bar_index - pos.entry_bar_index

            # Close in trade log
            record = self._trade_log.close_trade(
                trade_id=pos.trade_id,
                exit_time=timestamp,
                exit_price=fill_price,
                exit_signal_price=exit_signal_price,
                exit_reason=exit_reason,
                commission=commission,
                bar_count=bar_count,
            )
            closed_records.append(record)

            # Remove from tracking
            del self._trade_to_poi[pos.trade_id]

        # Remove closed positions from the list
        if trade_id is not None:
            self._positions[poi_id] = [
                p for p in self._positions[poi_id] if p.trade_id != trade_id
            ]
        else:
            self._positions[poi_id] = []

        # Clean up empty poi entries
        if not self._positions[poi_id]:
            del self._positions[poi_id]

        # Emit event
        if self._event_log is not None:
            self._event_log.emit(
                EventType.EXIT, timestamp, poi_id,
                exit_reason=exit_reason, fill_price=fill_price,
                n_closed=len(closed_records),
            )

        return closed_records

    def modify_stop_loss(self, poi_id: str, new_sl: float) -> None:
        """Update stop loss for all positions on a poi_id."""
        if poi_id in self._positions:
            for pos in self._positions[poi_id]:
                pos.stop_loss = new_sl

    def update_mark_to_market(
        self,
        bar_index: int,
        candle_high: float,
        candle_low: float,
        candle_close: float,
    ) -> None:
        """Update equity curve and MFE/MAE for current bar."""
        self._last_close = candle_close

        # Update MFE/MAE for all open positions
        for positions in self._positions.values():
            for pos in positions:
                self._trade_log.update_excursion(
                    pos.trade_id, candle_high, candle_low
                )

        # Record equity
        if 0 <= bar_index < len(self._equity_curve):
            self._equity_curve[bar_index] = self.equity

    def get_equity_curve(self) -> np.ndarray:
        """Return the equity curve array."""
        return self._equity_curve

    def get_positions_for_poi(self, poi_id: str) -> list[PositionInfo]:
        """Return open positions for a specific POI."""
        return self._positions.get(poi_id, [])

    def has_position_for_poi(self, poi_id: str) -> bool:
        """Check if there are any open positions for this POI."""
        return poi_id in self._positions and len(self._positions[poi_id]) > 0
