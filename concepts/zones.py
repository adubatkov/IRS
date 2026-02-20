"""Premium/Discount Zones and Consequent Encroachment (CE/CVB).

Premium zone: above 50% of the range (expensive, sell zone in bearish)
Discount zone: below 50% of the range (cheap, buy zone in bullish)
CE/CVB: 50% midpoint of any FVG, OB, or range.
"""



def premium_discount_zones(
    swing_high: float,
    swing_low: float,
) -> dict:
    """Calculate premium/discount zones between a swing high and swing low.

    Returns:
        Dict with zone boundaries and key levels.
    """
    if swing_high <= swing_low:
        raise ValueError("swing_high must be > swing_low")

    range_size = swing_high - swing_low
    equilibrium = (swing_high + swing_low) / 2

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "equilibrium": equilibrium,
        "premium_zone": (equilibrium, swing_high),
        "discount_zone": (swing_low, equilibrium),
        "quarter_75": swing_low + 0.75 * range_size,
        "quarter_25": swing_low + 0.25 * range_size,
    }


def classify_price_zone(
    price: float,
    swing_high: float,
    swing_low: float,
) -> str:
    """Classify a price as premium, discount, or equilibrium.

    Returns:
        "premium", "discount", or "equilibrium"
    """
    if swing_high <= swing_low:
        return "undefined"

    pct = (price - swing_low) / (swing_high - swing_low) * 100

    if pct > 55:
        return "premium"
    elif pct < 45:
        return "discount"
    else:
        return "equilibrium"


def consequent_encroachment(top: float, bottom: float) -> float:
    """Calculate the CE (50% midpoint) of any zone (FVG, OB, range).

    This is the key reaction level within a zone.
    """
    return (top + bottom) / 2


def zone_percentage(
    price: float,
    swing_high: float,
    swing_low: float,
) -> float:
    """Calculate where price sits within the range as a percentage (0-100)."""
    if swing_high <= swing_low:
        return 50.0
    return (price - swing_low) / (swing_high - swing_low) * 100
