"""POI (Point of Interest) Registry — aggregate all concepts into composite POIs.

A POI is a composite price zone where the trader expects a reaction.
It aggregates overlapping FVGs, Order Blocks, Breaker Blocks, IFVGs,
liquidity levels, and session levels into a single scored zone.

See doc/02_SMC_CONCEPTS.md section 10 for scoring specification.
"""

from enum import Enum

import pandas as pd


class POIStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TESTED = "TESTED"
    MITIGATED = "MITIGATED"


# --- Base scores per component type ---
_BASE_SCORES: dict[str, float] = {
    "fvg_htf": 3.0,
    "fvg_ltf": 1.0,
    "ob": 2.0,
    "breaker": 2.0,
    "ifvg": 2.0,
    "liquidity": 2.0,
    "session": 1.0,
}

# --- Freshness multipliers ---
_FRESHNESS_MULT: dict[str, float] = {
    "FRESH": 1.5,
    "ACTIVE": 1.5,
    "TESTED": 1.0,
    "PARTIALLY_FILLED": 0.5,
}

# HTF timeframes that get the higher FVG score
_HTF_TIMEFRAMES = {"4H", "1H", "4h", "1h", "240", "60"}


def build_poi_registry(
    fvgs: pd.DataFrame,
    obs: pd.DataFrame,
    breakers: pd.DataFrame,
    liquidity: pd.DataFrame,
    session_levels: pd.DataFrame,
    fvg_lifecycle: list[dict] | None = None,
    overlap_tolerance: float = 0.001,
    timeframe: str = "15m",
) -> pd.DataFrame:
    """Build a POI registry from all detected concept zones.

    Normalizes all inputs into a common zone format, then merges
    overlapping same-direction zones into composite POIs with scores.

    Args:
        fvgs: DataFrame from detect_fvg().
        obs: DataFrame from detect_orderblocks().
        breakers: DataFrame from detect_breakers().
        liquidity: DataFrame from detect_equal_levels().
        session_levels: DataFrame from detect_session_levels().
        fvg_lifecycle: Optional list from track_fvg_lifecycle().
        overlap_tolerance: Relative tolerance for zone overlap (fraction of price).
        timeframe: Source timeframe for FVG scoring (HTF vs LTF).

    Returns:
        DataFrame with one row per POI:
        - direction, top, bottom, midpoint
        - score: composite strength score
        - components: list of dicts (type, source_idx, status)
        - component_count: number of overlapping concepts
        - status: POIStatus
    """
    zones = _normalize_all(
        fvgs, obs, breakers, liquidity, session_levels,
        fvg_lifecycle, timeframe,
    )

    if not zones:
        return _empty_poi_df()

    # Group by direction, then merge overlapping
    bullish = [z for z in zones if z["direction"] == 1]
    bearish = [z for z in zones if z["direction"] == -1]

    pois = []
    pois.extend(_merge_zones(bullish, 1, overlap_tolerance))
    pois.extend(_merge_zones(bearish, -1, overlap_tolerance))

    if not pois:
        return _empty_poi_df()

    # Score each POI
    for poi in pois:
        poi["score"] = _score_poi(poi["components"])
        poi["component_count"] = len(poi["components"])
        poi["midpoint"] = (poi["top"] + poi["bottom"]) / 2
        poi["status"] = POIStatus.ACTIVE

    result = pd.DataFrame(pois)
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result


def update_poi_status(
    pois: pd.DataFrame,
    candle_high: float,
    candle_low: float,
    candle_close: float,
) -> pd.DataFrame:
    """Update POI statuses based on new price action.

    ACTIVE → TESTED on wick touch, MITIGATED on close through.
    """
    result = pois.copy()

    for idx in result.index:
        status = result.loc[idx, "status"]
        if status == POIStatus.MITIGATED:
            continue

        direction = result.loc[idx, "direction"]
        top = result.loc[idx, "top"]
        bottom = result.loc[idx, "bottom"]

        if direction == 1:  # Bullish POI (demand zone)
            if candle_close < bottom:
                result.loc[idx, "status"] = POIStatus.MITIGATED
            elif candle_low <= top:
                if status != POIStatus.TESTED:
                    result.loc[idx, "status"] = POIStatus.TESTED
        else:  # Bearish POI (supply zone)
            if candle_close > top:
                result.loc[idx, "status"] = POIStatus.MITIGATED
            elif candle_high >= bottom:
                if status != POIStatus.TESTED:
                    result.loc[idx, "status"] = POIStatus.TESTED

    return result


# ---- Internal helpers ----


def _empty_poi_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["direction", "top", "bottom", "midpoint", "score",
                 "components", "component_count", "status"]
    )


def _normalize_all(
    fvgs: pd.DataFrame,
    obs: pd.DataFrame,
    breakers: pd.DataFrame,
    liquidity: pd.DataFrame,
    session_levels: pd.DataFrame,
    fvg_lifecycle: list[dict] | None,
    timeframe: str,
) -> list[dict]:
    """Normalize all concept outputs into a common zone list."""
    zones: list[dict] = []

    is_htf = timeframe in _HTF_TIMEFRAMES

    # Build lifecycle lookup if available (keyed by fvg_idx)
    lifecycle_map: dict = {}
    if fvg_lifecycle:
        for lc in fvg_lifecycle:
            lifecycle_map[lc.get("fvg_idx", -1)] = lc

    # FVGs
    if len(fvgs) > 0:
        for i, row in fvgs.iterrows():
            status = str(row.get("status", "FRESH"))
            # Check lifecycle for updated status
            if i in lifecycle_map:
                lc = lifecycle_map[i]
                lc_status = str(lc.get("status", status))
                # INVERTED FVGs become IFVGs
                if lc_status == "INVERTED":
                    zones.append({
                        "direction": -int(row["direction"]),  # Invert direction
                        "top": float(row["top"]),
                        "bottom": float(row["bottom"]),
                        "source_type": "ifvg",
                        "source_idx": i,
                        "status": "ACTIVE",
                    })
                    continue
                status = lc_status

            if status in ("MITIGATED", "FULLY_FILLED"):
                continue

            source_type = "fvg_htf" if is_htf else "fvg_ltf"
            zones.append({
                "direction": int(row["direction"]),
                "top": float(row["top"]),
                "bottom": float(row["bottom"]),
                "source_type": source_type,
                "source_idx": i,
                "status": status,
            })

    # Order Blocks
    if len(obs) > 0:
        for i, row in obs.iterrows():
            status = str(row.get("status", "ACTIVE"))
            if status in ("MITIGATED", "BROKEN"):
                continue
            zones.append({
                "direction": int(row["direction"]),
                "top": float(row["top"]),
                "bottom": float(row["bottom"]),
                "source_type": "ob",
                "source_idx": i,
                "status": status,
            })

    # Breaker Blocks
    if len(breakers) > 0:
        for i, row in breakers.iterrows():
            status = str(row.get("status", "ACTIVE"))
            if status == "MITIGATED":
                continue
            zones.append({
                "direction": int(row["direction"]),
                "top": float(row["top"]),
                "bottom": float(row["bottom"]),
                "source_type": "breaker",
                "source_idx": i,
                "status": status,
            })

    # Liquidity levels → convert single level to thin zone
    if len(liquidity) > 0:
        for i, row in liquidity.iterrows():
            status = str(row.get("status", "ACTIVE"))
            if status == "SWEPT":
                continue
            level = float(row["level"])
            count = int(row.get("count", 2))
            if count < 3:
                continue  # Only score clusters with 3+ touches
            # Create a thin zone around the level
            zone_half = level * 0.0005  # ±0.05% band
            zones.append({
                "direction": int(row["direction"]),
                "top": level + zone_half,
                "bottom": level - zone_half,
                "source_type": "liquidity",
                "source_idx": i,
                "status": status,
            })

    # Session levels → each high/low becomes a thin zone
    if len(session_levels) > 0 and "high" in session_levels.columns:
        for i, row in session_levels.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            h_half = high * 0.0003
            l_half = low * 0.0003

            # Session high → bearish POI (resistance above)
            zones.append({
                "direction": -1,
                "top": high + h_half,
                "bottom": high - h_half,
                "source_type": "session",
                "source_idx": i,
                "status": "ACTIVE",
            })
            # Session low → bullish POI (support below)
            zones.append({
                "direction": 1,
                "top": low + l_half,
                "bottom": low - l_half,
                "source_type": "session",
                "source_idx": i,
                "status": "ACTIVE",
            })

    return zones


def _merge_zones(
    zones: list[dict],
    direction: int,
    tolerance: float,
) -> list[dict]:
    """Merge overlapping same-direction zones into composite POIs."""
    if not zones:
        return []

    # Sort by bottom price
    zones.sort(key=lambda z: z["bottom"])

    merged: list[dict] = []
    current_top = zones[0]["top"]
    current_bottom = zones[0]["bottom"]
    current_components = [{
        "type": zones[0]["source_type"],
        "source_idx": zones[0]["source_idx"],
        "status": zones[0]["status"],
    }]

    for z in zones[1:]:
        # Check overlap: zone_a.bottom <= zone_b.top and zone_b.bottom <= zone_a.top
        tol = current_top * tolerance
        if z["bottom"] <= current_top + tol:
            # Merge
            current_top = max(current_top, z["top"])
            current_bottom = min(current_bottom, z["bottom"])
            current_components.append({
                "type": z["source_type"],
                "source_idx": z["source_idx"],
                "status": z["status"],
            })
        else:
            # Emit current POI
            merged.append({
                "direction": direction,
                "top": current_top,
                "bottom": current_bottom,
                "components": current_components,
            })
            current_top = z["top"]
            current_bottom = z["bottom"]
            current_components = [{
                "type": z["source_type"],
                "source_idx": z["source_idx"],
                "status": z["status"],
            }]

    # Emit last POI
    merged.append({
        "direction": direction,
        "top": current_top,
        "bottom": current_bottom,
        "components": current_components,
    })

    return merged


def _score_poi(components: list[dict]) -> float:
    """Compute composite strength score for a POI.

    Scoring per doc/02_SMC_CONCEPTS.md section 10:
    - Base score per component type
    - Freshness multiplier
    - Confluence bonus for multiple overlapping components
    """
    total = 0.0

    for comp in components:
        base = _BASE_SCORES.get(comp["type"], 1.0)
        freshness = _FRESHNESS_MULT.get(comp.get("status", "ACTIVE"), 1.0)
        total += base * freshness

    # Confluence bonus
    n = len(components)
    if n >= 3:
        total += 4.0
    elif n == 2:
        total += 2.0

    return round(total, 2)
