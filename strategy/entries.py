"""Entry decision logic for the IRS strategy."""

import pandas as pd
from typing import Any, Optional

from config import StrategyConfig
from strategy.types import (
    POIPhase, POIState, Signal, SignalType, SyncMode,
)
from strategy.confirmations import has_fifth_confirm_trap
from strategy.fta_handler import should_enter_with_fta
from strategy.risk import calculate_stop_loss, validate_risk


def evaluate_entry(
    poi_state: POIState,
    candle: pd.Series,
    bar_index: int,
    timestamp: pd.Timestamp,
    fta: Optional[dict[str, Any]],
    fta_classification: str,
    sync_mode: SyncMode,
    nearby_fvgs: pd.DataFrame,
    nearby_liquidity: pd.DataFrame,
    config: StrategyConfig,
) -> Optional[Signal]:
    """Main entry evaluation.

    Decision tree:
    1. Phase must be READY
    2. FTA close? -> don't enter (return None)
    3. 5th-confirm trap? -> wait for RTO
    4. Conservative mode: check structural exit confirmation
    5. Aggressive mode: enter immediately
    6. RTO mode: wait for return to FVG/IFVG test

    Returns Signal or None.
    """
    if poi_state.phase != POIPhase.READY:
        return None

    # FTA check
    can_enter, _fta_reason = should_enter_with_fta(fta, fta_classification)
    if not can_enter:
        return None

    # 5th confirm trap
    if has_fifth_confirm_trap(poi_state.confirmations):
        if config.entry.rto_wait:
            # Check for RTO
            if check_rto_entry(poi_state, candle, nearby_fvgs):
                return _build_entry_signal(
                    poi_state, candle, bar_index, timestamp,
                    sync_mode, nearby_fvgs, nearby_liquidity,
                    config, reason="RTO entry after 5th-confirm trap"
                )
            return None
        # If rto_wait is False, allow entry anyway

    mode = config.entry.mode

    if mode == "conservative":
        if check_conservative_entry(poi_state, candle, config):
            return _build_entry_signal(
                poi_state, candle, bar_index, timestamp,
                sync_mode, nearby_fvgs, nearby_liquidity,
                config, reason="conservative entry"
            )
    elif mode == "aggressive":
        if check_aggressive_entry(poi_state, candle, config):
            return _build_entry_signal(
                poi_state, candle, bar_index, timestamp,
                sync_mode, nearby_fvgs, nearby_liquidity,
                config, reason="aggressive entry"
            )
    else:
        # Default to conservative
        if check_conservative_entry(poi_state, candle, config):
            return _build_entry_signal(
                poi_state, candle, bar_index, timestamp,
                sync_mode, nearby_fvgs, nearby_liquidity,
                config, reason="conservative entry (default)"
            )

    return None


def check_conservative_entry(
    poi_state: POIState,
    candle: pd.Series,
    config: StrategyConfig,
) -> bool:
    """Conservative entry: price has exited POI zone in favorable direction.

    For LONG: candle.close > poi_top (price moved up from demand)
    For SHORT: candle.close < poi_bottom (price moved down from supply)
    """
    direction = poi_state.poi_data["direction"]
    if direction == 1:
        return candle["close"] > poi_state.poi_data["top"]
    else:
        return candle["close"] < poi_state.poi_data["bottom"]


def check_aggressive_entry(
    poi_state: POIState,
    candle: pd.Series,
    config: StrategyConfig,
) -> bool:
    """Aggressive entry: as soon as phase is READY.

    Always returns True (entry happens immediately when READY).
    """
    return True


def check_rto_entry(
    poi_state: POIState,
    candle: pd.Series,
    nearby_fvgs: pd.DataFrame,
) -> bool:
    """RTO (Return to Origin) entry: price returns to test an FVG.

    For LONG: candle.low touches a bullish FVG zone (low <= fvg_top) that is active
    For SHORT: candle.high touches a bearish FVG zone (high >= fvg_bottom) that is active
    """
    if nearby_fvgs is None or len(nearby_fvgs) == 0:
        return False

    direction = poi_state.poi_data["direction"]
    _active = {"FRESH", "TESTED", "PARTIALLY_FILLED"}

    for _, fvg in nearby_fvgs.iterrows():
        status = fvg["status"]
        status_str = status.value if hasattr(status, "value") else str(status)
        if status_str not in _active:
            continue
        if fvg["direction"] != direction:
            continue

        if direction == 1:
            if candle["low"] <= fvg["top"]:
                return True
        else:
            if candle["high"] >= fvg["bottom"]:
                return True

    return False


def _build_entry_signal(
    poi_state: POIState,
    candle: pd.Series,
    bar_index: int,
    timestamp: pd.Timestamp,
    sync_mode: SyncMode,
    nearby_fvgs: pd.DataFrame,
    nearby_liquidity: pd.DataFrame,
    config: StrategyConfig,
    reason: str = "",
) -> Optional[Signal]:
    """Build a complete entry signal with SL and risk validation.

    Returns None if risk validation fails (RR too low).
    """
    direction = poi_state.poi_data["direction"]
    entry_price = candle["close"]

    sl = calculate_stop_loss(
        poi_data=poi_state.poi_data,
        direction=direction,
        nearby_fvgs=nearby_fvgs,
        nearby_liquidity=nearby_liquidity,
    )

    # Use a placeholder target for RR validation
    # (target selection is done by exits.py, but we need a reasonable estimate)
    # Use poi_data midpoint + 2x the SL distance as a rough target
    sl_distance = abs(entry_price - sl)
    if direction == 1:
        rough_target = entry_price + sl_distance * 3.0
    else:
        rough_target = entry_price - sl_distance * 3.0

    # Validate RR
    is_valid, actual_rr = validate_risk(entry_price, sl, rough_target, direction, min_rr=2.0)
    if not is_valid:
        return None

    # Position sizing
    from context.sync_checker import get_position_size_multiplier
    sync_mult = get_position_size_multiplier(sync_mode, config.risk)

    return Signal(
        type=SignalType.ENTER,
        poi_id=poi_state.poi_id,
        direction=direction,
        timestamp=timestamp,
        price=entry_price,
        stop_loss=sl,
        target=rough_target,
        position_size_mult=sync_mult,
        reason=reason,
        metadata={
            "bar_index": bar_index,
            "sync_mode": sync_mode.value,
            "rr": actual_rr,
            "confirmation_count": len(poi_state.confirmations),
        },
    )
