"""Breaker Block detection.

A Breaker Block is a failed Order Block - price closed through the OB,
inverting its role. Former support becomes resistance and vice versa.
"""

from enum import Enum

import pandas as pd

from concepts.orderblocks import OBStatus


class BreakerStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TESTED = "TESTED"
    MITIGATED = "MITIGATED"


def detect_breakers(
    obs: pd.DataFrame,
) -> pd.DataFrame:
    """Detect Breaker Blocks from broken Order Blocks.

    When an OB is BROKEN, it becomes a Breaker with inverted direction.

    Args:
        obs: DataFrame of Order Blocks with 'direction', 'top', 'bottom', 'status'.

    Returns:
        DataFrame with one row per Breaker:
        - direction: inverted from original OB (+1 becomes -1 and vice versa)
        - top, bottom: same zone as original OB
        - original_ob_index: index of the original OB
        - status: BreakerStatus.ACTIVE
    """
    broken = obs[obs["status"] == OBStatus.BROKEN]

    if len(broken) == 0:
        return pd.DataFrame(
            columns=["direction", "top", "bottom", "original_ob_index", "status"]
        )

    breakers = []
    for _, ob in broken.iterrows():
        breakers.append({
            "direction": -ob["direction"],  # Invert direction
            "top": ob["top"],
            "bottom": ob["bottom"],
            "original_ob_index": ob.get("ob_index", None),
            "status": BreakerStatus.ACTIVE,
        })

    return pd.DataFrame(breakers)


def update_breaker_status(
    breakers: pd.DataFrame,
    candle_high: float,
    candle_low: float,
    candle_close: float,
) -> pd.DataFrame:
    """Update Breaker Block statuses."""
    result = breakers.copy()

    for idx in result.index:
        if result.loc[idx, "status"] == BreakerStatus.MITIGATED:
            continue

        direction = result.loc[idx, "direction"]
        top = result.loc[idx, "top"]
        bottom = result.loc[idx, "bottom"]

        if direction == 1:  # Bullish breaker (acts as support)
            if candle_close < bottom:
                result.loc[idx, "status"] = BreakerStatus.MITIGATED
            elif candle_low <= top:
                result.loc[idx, "status"] = BreakerStatus.TESTED
        else:  # Bearish breaker (acts as resistance)
            if candle_close > top:
                result.loc[idx, "status"] = BreakerStatus.MITIGATED
            elif candle_high >= bottom:
                result.loc[idx, "status"] = BreakerStatus.TESTED

    return result
