"""First Trouble Area (FTA) detection and handling.

FTA is the first opposing POI on the path to the target that could block or reverse price.
"""

import pandas as pd
from typing import Any, Optional


def detect_fta(
    current_price: float,
    target: float,
    direction: int,
    active_pois: pd.DataFrame,
) -> Optional[dict[str, Any]]:
    """Find the First Trouble Area between current price and target.

    For LONG (direction=+1):
        FTA = first active BEARISH POI where bottom > current_price AND top < target
        (opposing supply zone in the path)
        Sort by bottom ascending, take the first (closest)

    For SHORT (direction=-1):
        FTA = first active BULLISH POI where top < current_price AND bottom > target
        (opposing demand zone in the path)
        Sort by top descending, take the first (closest)

    Args:
        current_price: Current market price.
        target: Target price level.
        direction: +1 for long, -1 for short.
        active_pois: DataFrame of active POIs across timeframes.

    Returns:
        Dict with FTA info {direction, top, bottom, midpoint, score} or None.
    """
    if active_pois.empty:
        return None

    # Filter to only ACTIVE or TESTED POIs (not mitigated)
    active = active_pois[active_pois["status"].isin(["ACTIVE", "TESTED"])].copy()
    if active.empty:
        return None

    if direction == 1:
        # Long trade: look for bearish (supply) zones between price and target
        opposing = active[active["direction"] == -1]
        in_path = opposing[
            (opposing["bottom"] > current_price) & (opposing["top"] < target)
        ]
        if in_path.empty:
            return None
        # Closest to current price = lowest bottom
        in_path = in_path.sort_values("bottom", ascending=True)
        fta_row = in_path.iloc[0]
    elif direction == -1:
        # Short trade: look for bullish (demand) zones between price and target
        opposing = active[active["direction"] == 1]
        in_path = opposing[
            (opposing["top"] < current_price) & (opposing["bottom"] > target)
        ]
        if in_path.empty:
            return None
        # Closest to current price = highest top
        in_path = in_path.sort_values("top", ascending=False)
        fta_row = in_path.iloc[0]
    else:
        return None

    return {
        "direction": int(fta_row["direction"]),
        "top": float(fta_row["top"]),
        "bottom": float(fta_row["bottom"]),
        "midpoint": float(fta_row["midpoint"]),
        "score": float(fta_row["score"]),
    }


def classify_fta_distance(
    fta: dict[str, Any],
    current_price: float,
    target: float,
    close_threshold_pct: float = 0.3,
) -> str:
    """Classify FTA distance as 'far' or 'close'.

    Distance is measured as fraction of the price-to-target range.
    If FTA midpoint is within close_threshold_pct of the total range
    from current_price, it's 'close'.

    Example: price=100, target=110, FTA midpoint=103, threshold=0.3
    Range = |110-100| = 10
    FTA distance from price = |103-100| = 3
    Fraction = 3/10 = 0.3 -> "close" (equal to threshold)

    Returns: "far" or "close"
    """
    total_range = abs(target - current_price)
    if total_range == 0:
        return "close"

    fta_distance = abs(fta["midpoint"] - current_price)
    fraction = fta_distance / total_range

    if fraction <= close_threshold_pct:
        return "close"
    return "far"


def check_fta_invalidation(
    fta: dict[str, Any],
    candle_close: float,
    direction: int,
) -> bool:
    """Check if FTA has been invalidated (price closed through it).

    For LONG: FTA is bearish supply above. Invalidated if candle_close > fta["top"]
    For SHORT: FTA is bullish demand below. Invalidated if candle_close < fta["bottom"]
    """
    if direction == 1:
        return candle_close > fta["top"]
    elif direction == -1:
        return candle_close < fta["bottom"]
    return False


def check_fta_validation(
    fta: dict[str, Any],
    candle_high: float,
    candle_low: float,
    candle_close: float,
    direction: int,
) -> bool:
    """Check if FTA validated (price bounced off it = rejection).

    For LONG: FTA (bearish) validated if candle reached FTA zone (high >= fta["bottom"])
              but closed back below the zone (close < fta["bottom"])
    For SHORT: FTA (bullish) validated if candle reached FTA zone (low <= fta["top"])
               but closed back above the zone (close > fta["top"])
    """
    if direction == 1:
        reached = candle_high >= fta["bottom"]
        rejected = candle_close < fta["bottom"]
        return reached and rejected
    elif direction == -1:
        reached = candle_low <= fta["top"]
        rejected = candle_close > fta["top"]
        return reached and rejected
    return False


def should_enter_with_fta(
    fta: Optional[dict[str, Any]],
    fta_classification: str,
) -> tuple[bool, str]:
    """Decision: should we enter given the FTA situation?

    Returns (should_enter, reason):
    - No FTA: (True, "no FTA, clear path to target")
    - FTA far: (True, "FTA far, enter normally")
    - FTA close: (False, "FTA close, wait for invalidation")
    """
    if fta is None:
        return (True, "no FTA, clear path to target")

    if fta_classification == "far":
        return (True, "FTA far, enter normally")

    return (False, "FTA close, wait for invalidation")
