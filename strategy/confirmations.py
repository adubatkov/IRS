"""Confirmation counting and validation per POI interaction.

Each POI interaction collects confirmations from 8 distinct checker functions.
When the minimum threshold is met, a trade entry can be considered.

Rules:
- Same confirmation type at the same bar_index is not counted twice.
- FVG_WICK_REACTION is only valid after 5+ confirmations already collected.
- Total confirmations are capped at config.max_count.
"""

import pandas as pd
from typing import Any

from config import ConfirmationsConfig
from strategy.types import Confirmation, ConfirmationType

# FVG statuses considered active for confirmation checks
ACTIVE_FVG_STATUSES = {"FRESH", "TESTED", "PARTIALLY_FILLED"}


# ---------------------------------------------------------------------------
# Individual checker functions
# ---------------------------------------------------------------------------

def check_poi_tap(
    candle_high: float,
    candle_low: float,
    poi_top: float,
    poi_bottom: float,
    poi_direction: int,
) -> bool:
    """Check if current candle taps (enters) the POI zone.

    For bullish POI (demand): candle low touches the zone (candle_low <= poi_top)
    For bearish POI (supply): candle high touches the zone (candle_high >= poi_bottom)
    """
    if poi_direction == 1:
        # Demand zone below price: candle dips into it
        return candle_low <= poi_top
    else:
        # Supply zone above price: candle pokes into it
        return candle_high >= poi_bottom


def check_liquidity_sweep(
    candle_high: float,
    candle_low: float,
    candle_close: float,
    nearby_liquidity: pd.DataFrame,
    poi_direction: int,
) -> dict[str, Any] | None:
    """Check if a nearby liquidity level was swept.

    For bullish POI: look for sell-side liquidity sweep
        (candle_low < level AND candle_close >= level)
    For bearish POI: look for buy-side liquidity sweep
        (candle_high > level AND candle_close <= level)

    Only considers ACTIVE liquidity levels in the correct direction:
    - Bullish POI needs sell-side liquidity (direction=-1) swept
    - Bearish POI needs buy-side liquidity (direction=+1) swept

    Returns dict with {level, direction} or None.
    """
    if nearby_liquidity is None or len(nearby_liquidity) == 0:
        return None

    # Filter for correct side and ACTIVE status
    if poi_direction == 1:
        # Bullish POI -> sweep sell-side liquidity (direction=-1)
        target_dir = -1
    else:
        # Bearish POI -> sweep buy-side liquidity (direction=+1)
        target_dir = 1

    candidates = nearby_liquidity[
        (nearby_liquidity["direction"] == target_dir)
        & (nearby_liquidity["status"] == "ACTIVE")
    ]

    for _, row in candidates.iterrows():
        level = row["level"]
        if target_dir == -1:
            # Sell-side (below): wick below level, close back above
            if candle_low < level and candle_close >= level:
                return {"level": level, "direction": target_dir}
        else:
            # Buy-side (above): wick above level, close back below
            if candle_high > level and candle_close <= level:
                return {"level": level, "direction": target_dir}

    return None


def check_fvg_inversion(
    fvg_lifecycle: list[dict],
    bar_index: int,
    poi_direction: int,
) -> dict[str, Any] | None:
    """Check if an FVG was inverted at this bar.

    An FVG opposing the POI direction gets inverted = confirmation.
    Check lifecycle entries where inversion_index == bar_index.

    For bullish POI: a bearish FVG (direction=-1) getting inverted is bullish confirmation
    For bearish POI: a bullish FVG (direction=+1) getting inverted is bearish confirmation

    Returns dict with fvg details or None.
    """
    if not fvg_lifecycle:
        return None

    # Opposing direction to POI
    opposing_dir = -poi_direction

    for entry in fvg_lifecycle:
        if (
            entry.get("inversion_index") == bar_index
            and entry.get("direction") == opposing_dir
        ):
            return {
                "fvg_idx": entry.get("fvg_idx"),
                "direction": entry["direction"],
                "top": entry["top"],
                "bottom": entry["bottom"],
                "midpoint": entry["midpoint"],
                "inversion_index": bar_index,
            }

    return None


def check_inversion_test(
    candle_high: float,
    candle_low: float,
    fvg_lifecycle: list[dict],
    poi_direction: int,
) -> dict[str, Any] | None:
    """Check if price is testing an already-inverted FVG (IFVG).

    After an FVG is inverted, it becomes an IFVG with reversed direction.
    Testing it means price touches the zone.

    For bullish POI: the inverted FVG (originally bearish, now bullish support)
                     is tested if candle_low <= ifvg_top
    For bearish POI: the inverted FVG (originally bullish, now bearish resistance)
                     is tested if candle_high >= ifvg_bottom

    Only considers lifecycle entries with status == "INVERTED" and
    inversion_index is not None.
    Returns dict with IFVG details or None.
    """
    if not fvg_lifecycle:
        return None

    # Opposing direction FVGs that got inverted now support our direction
    opposing_dir = -poi_direction

    for entry in fvg_lifecycle:
        status = entry.get("status")
        # FVGStatus is str enum, so "INVERTED" comparison works with both
        status_str = status.value if hasattr(status, "value") else str(status)
        if status_str != "INVERTED":
            continue
        if entry.get("inversion_index") is None:
            continue
        if entry.get("direction") != opposing_dir:
            continue

        top = entry["top"]
        bottom = entry["bottom"]

        if poi_direction == 1:
            # Bullish: inverted bearish FVG is now support, test from above
            if candle_low <= top:
                return {
                    "fvg_idx": entry.get("fvg_idx"),
                    "direction": entry["direction"],
                    "top": top,
                    "bottom": bottom,
                    "midpoint": entry["midpoint"],
                    "inversion_index": entry["inversion_index"],
                }
        else:
            # Bearish: inverted bullish FVG is now resistance, test from below
            if candle_high >= bottom:
                return {
                    "fvg_idx": entry.get("fvg_idx"),
                    "direction": entry["direction"],
                    "top": top,
                    "bottom": bottom,
                    "midpoint": entry["midpoint"],
                    "inversion_index": entry["inversion_index"],
                }

    return None


def check_structure_break(
    structure_events: pd.DataFrame,
    bar_index: int,
    poi_direction: int,
) -> dict[str, Any] | None:
    """Check if a BOS or CBOS occurred at this bar in the POI direction.

    Match: structure_events where broken_index == bar_index AND
           direction == poi_direction.
    Returns dict with {type, direction, broken_level} or None.
    """
    if structure_events is None or len(structure_events) == 0:
        return None

    matches = structure_events[
        (structure_events["broken_index"] == bar_index)
        & (structure_events["direction"] == poi_direction)
    ]

    if len(matches) == 0:
        return None

    row = matches.iloc[0]
    event_type = row["type"]
    type_str = event_type.value if hasattr(event_type, "value") else str(event_type)
    return {
        "type": type_str,
        "direction": int(row["direction"]),
        "broken_level": float(row["broken_level"]),
    }


def check_fvg_wick_reaction(
    candle_open: float,
    candle_high: float,
    candle_low: float,
    candle_close: float,
    nearby_fvgs: pd.DataFrame,
    poi_direction: int,
) -> dict[str, Any] | None:
    """Check if candle wick reacted to an FVG (touch + rejection wick).

    For bullish POI: candle dips into a bullish FVG zone (low <= fvg_top)
                     but closes above the midpoint (close > fvg_midpoint)
                     and has a lower wick (min(open,close) - low > 0)
    For bearish POI: candle pokes into a bearish FVG zone (high >= fvg_bottom)
                     but closes below the midpoint (close < fvg_midpoint)
                     and has an upper wick (high - max(open,close) > 0)

    Only considers FVGs with status in (FRESH, TESTED, PARTIALLY_FILLED).
    Returns dict with fvg details or None.
    """
    if nearby_fvgs is None or len(nearby_fvgs) == 0:
        return None

    _active = ACTIVE_FVG_STATUSES

    for _, fvg in nearby_fvgs.iterrows():
        status = fvg["status"]
        status_str = status.value if hasattr(status, "value") else str(status)
        if status_str not in _active:
            continue

        direction = fvg["direction"]
        top = fvg["top"]
        bottom = fvg["bottom"]
        midpoint = fvg["midpoint"]

        if poi_direction == 1 and direction == 1:
            # Bullish FVG acting as support
            body_low = min(candle_open, candle_close)
            lower_wick = body_low - candle_low
            if candle_low <= top and candle_close > midpoint and lower_wick > 0:
                return {
                    "direction": direction,
                    "top": top,
                    "bottom": bottom,
                    "midpoint": midpoint,
                    "wick_size": lower_wick,
                }

        elif poi_direction == -1 and direction == -1:
            # Bearish FVG acting as resistance
            body_high = max(candle_open, candle_close)
            upper_wick = candle_high - body_high
            if candle_high >= bottom and candle_close < midpoint and upper_wick > 0:
                return {
                    "direction": direction,
                    "top": top,
                    "bottom": bottom,
                    "midpoint": midpoint,
                    "wick_size": upper_wick,
                }

    return None


def check_cvb_test(
    candle_high: float,
    candle_low: float,
    nearby_fvgs: pd.DataFrame,
    poi_direction: int,
    tolerance_pct: float = 0.001,
) -> dict[str, Any] | None:
    """Check if price tested the CE (50% midpoint) of a nearby FVG.

    For bullish POI: candle touches near the midpoint of a bullish FVG from above
                     (candle_low <= midpoint * (1 + tolerance))
    For bearish POI: candle touches near the midpoint of a bearish FVG from below
                     (candle_high >= midpoint * (1 - tolerance))

    Only considers active FVGs in the same direction as POI.
    Returns dict with fvg details or None.
    """
    if nearby_fvgs is None or len(nearby_fvgs) == 0:
        return None

    _active = ACTIVE_FVG_STATUSES

    for _, fvg in nearby_fvgs.iterrows():
        status = fvg["status"]
        status_str = status.value if hasattr(status, "value") else str(status)
        if status_str not in _active:
            continue
        if fvg["direction"] != poi_direction:
            continue

        midpoint = fvg["midpoint"]

        if poi_direction == 1:
            # Bullish: price dips toward midpoint from above
            if candle_low <= midpoint * (1 + tolerance_pct):
                return {
                    "direction": int(fvg["direction"]),
                    "top": fvg["top"],
                    "bottom": fvg["bottom"],
                    "midpoint": midpoint,
                }
        else:
            # Bearish: price pushes toward midpoint from below
            if candle_high >= midpoint * (1 - tolerance_pct):
                return {
                    "direction": int(fvg["direction"]),
                    "top": fvg["top"],
                    "bottom": fvg["bottom"],
                    "midpoint": midpoint,
                }

    return None


def check_additional_cbos(
    structure_events: pd.DataFrame,
    bar_index: int,
    poi_direction: int,
    existing_confirms: list[Confirmation],
) -> dict[str, Any] | None:
    """Check for continuation BOS (cBOS) beyond the first structure break.

    This counts only if there's already a STRUCTURE_BREAK confirmation.
    If a cBOS event occurs at bar_index in poi_direction, it's an additional
    confirmation.

    Returns dict with details or None.
    """
    if structure_events is None or len(structure_events) == 0:
        return None

    # Must have a prior STRUCTURE_BREAK confirmation
    has_prior_sb = any(
        c.type == ConfirmationType.STRUCTURE_BREAK for c in existing_confirms
    )
    if not has_prior_sb:
        return None

    # Look for CBOS events at this bar
    matches = structure_events[
        (structure_events["broken_index"] == bar_index)
        & (structure_events["direction"] == poi_direction)
    ]

    for _, row in matches.iterrows():
        event_type = row["type"]
        type_str = event_type.value if hasattr(event_type, "value") else str(event_type)
        if type_str == "CBOS":
            return {
                "type": type_str,
                "direction": int(row["direction"]),
                "broken_level": float(row["broken_level"]),
            }

    return None


# ---------------------------------------------------------------------------
# Master collection function
# ---------------------------------------------------------------------------

def collect_confirmations(
    candle: pd.Series,
    bar_index: int,
    timestamp: pd.Timestamp,
    poi_data: dict[str, Any],
    existing_confirms: list[Confirmation],
    nearby_fvgs: pd.DataFrame,
    fvg_lifecycle: list[dict],
    nearby_liquidity: pd.DataFrame,
    structure_events: pd.DataFrame,
    config: ConfirmationsConfig,
) -> list[Confirmation]:
    """Master function: check all confirmation types and return updated list.

    Processes all 8 checkers in order and appends new confirmations.

    Rules applied:
    - Same type + same bar_index = not counted twice
    - FVG_WICK_REACTION only added if already have 5+ confirmations
    - Stops if total would exceed config.max_count

    Args:
        candle: Current bar data (open, high, low, close).
        bar_index: Current bar index in the DataFrame.
        timestamp: Current bar timestamp.
        poi_data: Dict with at least {direction, top, bottom, midpoint}.
        existing_confirms: List of previously collected confirmations.
        nearby_fvgs: Active FVGs near the POI.
        fvg_lifecycle: Full FVG lifecycle data.
        nearby_liquidity: Active liquidity levels near the POI.
        structure_events: Structure break events.
        config: Confirmation configuration.

    Returns:
        New list with any newly detected confirmations appended.
    """
    confirms = list(existing_confirms)  # shallow copy
    direction = poi_data["direction"]
    poi_top = poi_data["top"]
    poi_bottom = poi_data["bottom"]

    c_open = candle["open"]
    c_high = candle["high"]
    c_low = candle["low"]
    c_close = candle["close"]

    def _already_counted(ctype: ConfirmationType, bidx: int) -> bool:
        """Check if this exact type+bar_index combo already exists."""
        return any(c.type == ctype and c.bar_index == bidx for c in confirms)

    def _at_cap() -> bool:
        return len(confirms) >= config.max_count

    def _add(ctype: ConfirmationType, details: dict[str, Any] | None = None) -> None:
        if _at_cap():
            return
        if _already_counted(ctype, bar_index):
            return
        confirms.append(Confirmation(
            type=ctype,
            timestamp=timestamp,
            bar_index=bar_index,
            details=details or {},
        ))

    # 1. POI Tap
    if check_poi_tap(c_high, c_low, poi_top, poi_bottom, direction):
        _add(ConfirmationType.POI_TAP)

    # 2. Liquidity Sweep
    sweep = check_liquidity_sweep(c_high, c_low, c_close, nearby_liquidity, direction)
    if sweep is not None:
        _add(ConfirmationType.LIQUIDITY_SWEEP, sweep)

    # 3. FVG Inversion
    inversion = check_fvg_inversion(fvg_lifecycle, bar_index, direction)
    if inversion is not None:
        _add(ConfirmationType.FVG_INVERSION, inversion)

    # 4. Inversion Test
    inv_test = check_inversion_test(c_high, c_low, fvg_lifecycle, direction)
    if inv_test is not None:
        _add(ConfirmationType.INVERSION_TEST, inv_test)

    # 5. Structure Break
    sb = check_structure_break(structure_events, bar_index, direction)
    if sb is not None:
        _add(ConfirmationType.STRUCTURE_BREAK, sb)

    # 6. FVG Wick Reaction -- ONLY valid after 5+ pre-existing confirmations
    if len(existing_confirms) >= 5:
        wick = check_fvg_wick_reaction(c_open, c_high, c_low, c_close, nearby_fvgs, direction)
        if wick is not None:
            _add(ConfirmationType.FVG_WICK_REACTION, wick)

    # 7. CVB Test
    cvb = check_cvb_test(c_high, c_low, nearby_fvgs, direction)
    if cvb is not None:
        _add(ConfirmationType.CVB_TEST, cvb)

    # 8. Additional cBOS (uses updated confirms list for prior check)
    cbos = check_additional_cbos(structure_events, bar_index, direction, confirms)
    if cbos is not None:
        _add(ConfirmationType.ADDITIONAL_CBOS, cbos)

    return confirms


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def confirmation_count(confirms: list[Confirmation]) -> int:
    """Return the total number of confirmations."""
    return len(confirms)


def is_ready(
    confirms: list[Confirmation],
    config: ConfirmationsConfig,
) -> bool:
    """Check if minimum confirmation threshold is met."""
    return len(confirms) >= config.min_count


def has_fifth_confirm_trap(
    confirms: list[Confirmation],
) -> bool:
    """Detect the '5th confirm trap' scenario.

    Returns True if:
    - 5 or more confirmations exist
    - NONE of them are FVG_INVERSION, INVERSION_TEST, or FVG_WICK_REACTION
    - The last confirm is STRUCTURE_BREAK or ADDITIONAL_CBOS

    This means price exited POI structurally but never tested any FVG,
    so an RTO (return to test FVG/inversion) is likely.
    """
    if len(confirms) < 5:
        return False

    fvg_related = {
        ConfirmationType.FVG_INVERSION,
        ConfirmationType.INVERSION_TEST,
        ConfirmationType.FVG_WICK_REACTION,
    }

    if any(c.type in fvg_related for c in confirms):
        return False

    structural_types = {
        ConfirmationType.STRUCTURE_BREAK,
        ConfirmationType.ADDITIONAL_CBOS,
    }

    return confirms[-1].type in structural_types
