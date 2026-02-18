"""Exit decision logic: target hits, stop losses, breakeven management."""

import pandas as pd
from typing import Any, Optional

from config import StrategyConfig
from strategy.types import (
    POIState, Signal, SignalType, ExitReason, SyncMode, POIPhase,
)
from strategy.risk import calculate_breakeven_level


def check_target_hit(
    candle_high: float,
    candle_low: float,
    target: float,
    direction: int,
) -> bool:
    """Check if target was hit.

    LONG: candle_high >= target
    SHORT: candle_low <= target
    """
    if direction == 1:
        return candle_high >= target
    else:
        return candle_low <= target


def check_stop_loss_hit(
    candle_high: float,
    candle_low: float,
    stop_loss: float,
    direction: int,
) -> bool:
    """Check if stop loss was hit.

    LONG: candle_low <= stop_loss
    SHORT: candle_high >= stop_loss
    """
    if direction == 1:
        return candle_low <= stop_loss
    else:
        return candle_high >= stop_loss


def check_structural_breakeven(
    poi_state: POIState,
    structure_events: pd.DataFrame,
    bar_index: int,
    config: StrategyConfig,
) -> Optional[float]:
    """Check if a structural event warrants moving SL to breakeven.

    If there's a structure break in the trade direction after entry,
    move SL to breakeven.

    Returns breakeven level or None.
    """
    if not config.breakeven.structural_bu:
        return None

    if poi_state.entry_price is None:
        return None

    direction = poi_state.poi_data["direction"]

    if structure_events is None or len(structure_events) == 0:
        return None

    # Look for structure breaks at or after entry in our direction
    matches = structure_events[
        (structure_events["broken_index"] == bar_index)
        & (structure_events["direction"] == direction)
    ]

    if len(matches) > 0:
        return calculate_breakeven_level(
            poi_state.entry_price,
            direction,
        )

    return None


def check_fta_breakeven(
    poi_state: POIState,
    fta: Optional[dict[str, Any]],
    current_price: float,
    config: StrategyConfig,
) -> Optional[float]:
    """Check if FTA validation warrants moving to breakeven.

    If price reached FTA area and bounced, move SL to BE.
    """
    if not config.breakeven.fta_bu:
        return None

    if poi_state.entry_price is None or fta is None:
        return None

    direction = poi_state.poi_data["direction"]

    # Check if price has reached past FTA midpoint
    if direction == 1:
        if current_price >= fta["midpoint"]:
            return calculate_breakeven_level(
                poi_state.entry_price,
                direction,
            )
    else:
        if current_price <= fta["midpoint"]:
            return calculate_breakeven_level(
                poi_state.entry_price,
                direction,
            )

    return None


def select_target(
    direction: int,
    current_price: float,
    active_pois: pd.DataFrame,
    swing_points: pd.DataFrame,
    sync_mode: SyncMode,
    config: StrategyConfig,
) -> float:
    """Select target price based on sync mode and market structure.

    SYNC -> "distant" targets: look at primary TF (4H/1H) opposing POIs and swings
    DESYNC -> "local" targets: look at local TF (15m/30m) opposing swings

    For LONG: find nearest significant high above price
    For SHORT: find nearest significant low below price

    Fallback: use 3x ATR-equivalent distance from price
    """
    if direction == 1:
        # Long: look for highs above current price
        if not swing_points.empty and "level" in swing_points.columns:
            candidates = swing_points[
                (swing_points["level"] > current_price)
                & (swing_points["direction"] == 1)
            ]
            if not candidates.empty:
                # Nearest high
                candidates = candidates.sort_values("level", ascending=True)
                return float(candidates.iloc[0]["level"])

        # Fallback: use opposing POIs
        if not active_pois.empty:
            opposing = active_pois[
                (active_pois["direction"] == -1)
                & (active_pois["bottom"] > current_price)
            ]
            if not opposing.empty:
                opposing = opposing.sort_values("bottom", ascending=True)
                return float(opposing.iloc[0]["bottom"])

        # Final fallback: 3% above price
        return current_price * 1.03

    else:
        # Short: look for lows below current price
        if not swing_points.empty and "level" in swing_points.columns:
            candidates = swing_points[
                (swing_points["level"] < current_price)
                & (swing_points["direction"] == -1)
            ]
            if not candidates.empty:
                # Nearest low
                candidates = candidates.sort_values("level", ascending=False)
                return float(candidates.iloc[0]["level"])

        # Fallback: opposing POIs
        if not active_pois.empty:
            opposing = active_pois[
                (active_pois["direction"] == 1)
                & (active_pois["top"] < current_price)
            ]
            if not opposing.empty:
                opposing = opposing.sort_values("top", ascending=False)
                return float(opposing.iloc[0]["top"])

        # Final fallback: 3% below price
        return current_price * 0.97


def evaluate_exit(
    poi_state: POIState,
    candle: pd.Series,
    bar_index: int,
    timestamp: pd.Timestamp,
    fta: Optional[dict[str, Any]],
    structure_events: pd.DataFrame,
    config: StrategyConfig,
) -> Optional[Signal]:
    """Evaluate exit conditions for an active position.

    Checks in priority order:
    1. Stop loss hit -> EXIT with STOP_LOSS_HIT
    2. Target hit -> EXIT with TARGET_HIT
    3. Structural breakeven -> MODIFY_SL
    4. FTA breakeven -> MODIFY_SL

    Returns Signal or None.
    """
    if poi_state.phase not in (POIPhase.POSITIONED, POIPhase.MANAGING):
        return None

    if poi_state.entry_price is None or poi_state.stop_loss is None or poi_state.target is None:
        return None

    direction = poi_state.poi_data["direction"]
    c_high = candle["high"]
    c_low = candle["low"]
    c_close = candle["close"]

    # 1. Stop loss
    if check_stop_loss_hit(c_high, c_low, poi_state.stop_loss, direction):
        return Signal(
            type=SignalType.EXIT,
            poi_id=poi_state.poi_id,
            direction=direction,
            timestamp=timestamp,
            price=poi_state.stop_loss,
            reason=ExitReason.STOP_LOSS_HIT.value,
            metadata={"bar_index": bar_index},
        )

    # 2. Target hit
    if check_target_hit(c_high, c_low, poi_state.target, direction):
        return Signal(
            type=SignalType.EXIT,
            poi_id=poi_state.poi_id,
            direction=direction,
            timestamp=timestamp,
            price=poi_state.target,
            reason=ExitReason.TARGET_HIT.value,
            metadata={"bar_index": bar_index},
        )

    # 3. Structural breakeven
    be_level = check_structural_breakeven(poi_state, structure_events, bar_index, config)
    if be_level is not None and poi_state.breakeven_level is None:
        return Signal(
            type=SignalType.MOVE_TO_BE,
            poi_id=poi_state.poi_id,
            direction=direction,
            timestamp=timestamp,
            price=be_level,
            reason="structural breakeven",
            metadata={"bar_index": bar_index, "be_level": be_level},
        )

    # 4. FTA breakeven
    fta_be = check_fta_breakeven(poi_state, fta, c_close, config)
    if fta_be is not None and poi_state.breakeven_level is None:
        return Signal(
            type=SignalType.MOVE_TO_BE,
            poi_id=poi_state.poi_id,
            direction=direction,
            timestamp=timestamp,
            price=fta_be,
            reason="FTA breakeven",
            metadata={"bar_index": bar_index, "be_level": fta_be},
        )

    return None
