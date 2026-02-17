"""Add-on position management for the IRS strategy."""

import pandas as pd
from typing import Optional

from config import StrategyConfig
from strategy.types import (
    POIState, Signal, SignalType, POIPhase,
)


def find_addon_candidates(
    direction: int,
    current_price: float,
    target: float,
    local_pois: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Find POIs that could serve as add-on entry points.

    Add-on candidates are local TF POIs between current price and target,
    in the same direction as the main trade.

    For LONG: bullish POIs above current price but below target
    For SHORT: bearish POIs below current price but above target

    Returns filtered DataFrame of candidates sorted by proximity.
    """
    if local_pois.empty:
        return local_pois.iloc[:0]

    # Filter for same-direction, ACTIVE POIs
    candidates = local_pois[
        (local_pois["direction"] == direction)
        & (local_pois["status"].isin(["ACTIVE", "TESTED"]))
    ].copy()

    if candidates.empty:
        return candidates

    if direction == 1:
        # Long: POI midpoint between price and target
        candidates = candidates[
            (candidates["midpoint"] > current_price)
            & (candidates["midpoint"] < target)
        ]
        candidates = candidates.sort_values("midpoint", ascending=True)
    else:
        # Short: POI midpoint between target and price
        candidates = candidates[
            (candidates["midpoint"] < current_price)
            & (candidates["midpoint"] > target)
        ]
        candidates = candidates.sort_values("midpoint", ascending=False)

    return candidates.reset_index(drop=True)


def evaluate_addon(
    main_state: POIState,
    candidate_poi: pd.Series,
    candle: pd.Series,
    bar_index: int,
    timestamp: pd.Timestamp,
    structure_events: pd.DataFrame,
    config: StrategyConfig,
) -> Optional[Signal]:
    """Evaluate if an add-on entry should be taken at a candidate POI.

    Criteria:
    1. Main position must be POSITIONED or MANAGING
    2. Candle must touch the candidate POI zone
    3. There must be a recent structure break in trade direction

    Returns ADD_ON Signal or None.
    """
    if main_state.phase not in (POIPhase.POSITIONED, POIPhase.MANAGING):
        return None

    direction = main_state.poi_data["direction"]

    # Check if candle touches candidate zone
    poi_top = candidate_poi["top"]
    poi_bottom = candidate_poi["bottom"]

    touches = False
    if direction == 1:
        touches = candle["low"] <= poi_top
    else:
        touches = candle["high"] >= poi_bottom

    if not touches:
        return None

    # Check for recent structure confirmation
    has_structure = False
    if structure_events is not None and len(structure_events) > 0:
        recent = structure_events[
            (structure_events["direction"] == direction)
            & (structure_events["broken_index"] <= bar_index)
            & (structure_events["broken_index"] >= bar_index - 10)
        ]
        has_structure = len(recent) > 0

    if not has_structure:
        return None

    return Signal(
        type=SignalType.ADD_ON,
        poi_id=main_state.poi_id,
        direction=direction,
        timestamp=timestamp,
        price=candle["close"],
        stop_loss=main_state.stop_loss or 0.0,
        target=main_state.target or 0.0,
        position_size_mult=0.5,  # Add-ons are half-size
        reason=f"add-on at {candidate_poi.get('midpoint', 0):.1f}",
        metadata={
            "bar_index": bar_index,
            "addon_poi_top": poi_top,
            "addon_poi_bottom": poi_bottom,
        },
    )


def should_addon_bu(
    addon_entry_price: float,
    current_price: float,
    direction: int,
    commission_pct: float = 0.0006,
) -> bool:
    """Check if an add-on position should be moved to breakeven.

    Move to BE when price moves favorably by at least the commission cost.

    For LONG: current_price > addon_entry * (1 + 3 * commission)
    For SHORT: current_price < addon_entry * (1 - 3 * commission)
    """
    if direction == 1:
        return current_price > addon_entry_price * (1 + 3 * commission_pct)
    else:
        return current_price < addon_entry_price * (1 - 3 * commission_pct)
