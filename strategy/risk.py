"""Position sizing, stop-loss placement, and risk calculations."""

import pandas as pd
from typing import Optional

from config import RiskConfig
from strategy.types import SyncMode


def calculate_stop_loss(
    poi_data: dict,
    direction: int,
    nearby_fvgs: pd.DataFrame,
    nearby_liquidity: pd.DataFrame,
    method: str = "behind_liquidity",
) -> float:
    """Calculate stop-loss level for a trade.

    Methods:
    - "behind_poi": Behind the full POI zone + small buffer
      LONG: poi_bottom - buffer, SHORT: poi_top + buffer

    - "behind_fvg": Behind the nearest relevant FVG
      LONG: min(fvg_bottom for bullish FVGs near POI) - buffer
      SHORT: max(fvg_top for bearish FVGs near POI) + buffer

    - "behind_cvb": Behind the CE (50%) of the nearest relevant FVG
      LONG: min(fvg_midpoint) - buffer, SHORT: max(fvg_midpoint) + buffer

    - "behind_liquidity": Behind the nearest liquidity interaction
      LONG: min(liquidity_level for sell-side near POI) - buffer
      SHORT: max(liquidity_level for buy-side near POI) + buffer

    Buffer = 0.0005 * price (5 pips equivalent)

    Fallback: if no relevant data for the chosen method, use "behind_poi".

    Args:
        poi_data: Dict with at least {direction, top, bottom, midpoint}.
        direction: Trade direction (+1 long, -1 short).
        nearby_fvgs: Active FVGs near the POI.
        nearby_liquidity: Active liquidity levels near the POI.
        method: SL calculation method.

    Returns:
        Stop-loss price level.
    """
    price = poi_data["midpoint"]
    buffer = 0.0005 * price

    def _behind_poi() -> float:
        if direction == 1:
            return poi_data["bottom"] - buffer
        else:
            return poi_data["top"] + buffer

    def _behind_fvg() -> Optional[float]:
        if nearby_fvgs.empty:
            return None
        if direction == 1:
            # Bullish FVGs supporting the long trade
            relevant = nearby_fvgs[nearby_fvgs["direction"] == 1]
            if relevant.empty:
                return None
            return float(relevant["bottom"].min()) - buffer
        else:
            # Bearish FVGs supporting the short trade
            relevant = nearby_fvgs[nearby_fvgs["direction"] == -1]
            if relevant.empty:
                return None
            return float(relevant["top"].max()) + buffer

    def _behind_cvb() -> Optional[float]:
        if nearby_fvgs.empty:
            return None
        if direction == 1:
            relevant = nearby_fvgs[nearby_fvgs["direction"] == 1]
            if relevant.empty:
                return None
            return float(relevant["midpoint"].min()) - buffer
        else:
            relevant = nearby_fvgs[nearby_fvgs["direction"] == -1]
            if relevant.empty:
                return None
            return float(relevant["midpoint"].max()) + buffer

    def _behind_liquidity() -> Optional[float]:
        if nearby_liquidity.empty:
            return None
        if direction == 1:
            # Sell-side liquidity (below price) — direction -1
            relevant = nearby_liquidity[nearby_liquidity["direction"] == -1]
            if relevant.empty:
                return None
            return float(relevant["level"].min()) - buffer
        else:
            # Buy-side liquidity (above price) — direction +1
            relevant = nearby_liquidity[nearby_liquidity["direction"] == 1]
            if relevant.empty:
                return None
            return float(relevant["level"].max()) + buffer

    dispatch = {
        "behind_poi": lambda: _behind_poi(),
        "behind_fvg": _behind_fvg,
        "behind_cvb": _behind_cvb,
        "behind_liquidity": _behind_liquidity,
    }

    compute = dispatch.get(method, lambda: None)
    result = compute()

    # Fallback to behind_poi if method produced no result
    if result is None:
        result = _behind_poi()

    return result


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss: float,
    sync_mode: SyncMode,
    risk_config: RiskConfig,
) -> float:
    """Calculate position size based on risk per trade.

    Formula:
        risk_amount = account_equity * max_risk_per_trade
        distance = abs(entry_price - stop_loss)
        if distance == 0: return 0.0
        sync_mult = position_size_sync if SYNC else position_size_desync (0 for UNDEFINED)
        size = (risk_amount / distance) * sync_mult

    Returns:
        Position size in units.
    """
    risk_amount = account_equity * risk_config.max_risk_per_trade
    distance = abs(entry_price - stop_loss)

    if distance == 0:
        return 0.0

    if sync_mode == SyncMode.SYNC:
        sync_mult = risk_config.position_size_sync
    elif sync_mode == SyncMode.DESYNC:
        sync_mult = risk_config.position_size_desync
    else:
        sync_mult = 0.0

    size = (risk_amount / distance) * sync_mult
    return size


def validate_risk(
    entry_price: float,
    stop_loss: float,
    target: float,
    direction: int,
    min_rr: float = 2.0,
) -> tuple[bool, float]:
    """Validate that the trade meets minimum RR requirements.

    For LONG: RR = (target - entry) / (entry - stop_loss)
    For SHORT: RR = (entry - target) / (stop_loss - entry)

    Returns:
        (is_valid, actual_rr)
        is_valid is True if actual_rr >= min_rr
    """
    if direction == 1:
        reward = target - entry_price
        risk = entry_price - stop_loss
    else:
        reward = entry_price - target
        risk = stop_loss - entry_price

    if risk <= 0:
        return (False, 0.0)

    actual_rr = reward / risk
    return (actual_rr >= min_rr, actual_rr)


def calculate_breakeven_level(
    entry_price: float,
    direction: int,
    commission_pct: float = 0.0006,
) -> float:
    """Calculate breakeven price including commission costs.

    For LONG: entry_price * (1 + 2 * commission_pct)  (buy + sell commission)
    For SHORT: entry_price * (1 - 2 * commission_pct)
    """
    if direction == 1:
        return entry_price * (1 + 2 * commission_pct)
    else:
        return entry_price * (1 - 2 * commission_pct)
