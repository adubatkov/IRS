"""Shared test fixtures for the IRS backtesting system."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_trending_1m(n_bars: int = 600, base_price: float = 21000.0) -> pd.DataFrame:
    """Create synthetic 1m data with uptrend + pullback + continuation.

    Pattern:
    - Bars 0-200: Strong uptrend (builds bullish structure)
    - Bars 200-300: Pullback into demand zone (POI forms)
    - Bars 300-400: Consolidation + bounce (confirmations accumulate)
    - Bars 400-600: Continuation up (trade plays out to target)
    """
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-02 09:00", periods=n_bars, freq="1min", tz="UTC")

    prices = np.zeros(n_bars)
    prices[0] = base_price

    for i in range(1, n_bars):
        if i < 200:
            drift = 2.0
        elif i < 300:
            drift = -1.5
        elif i < 400:
            drift = 0.5
        else:
            drift = 1.5
        prices[i] = prices[i - 1] + drift + rng.normal(0, 1.5)

    noise = rng.uniform(0.5, 3.0, n_bars)
    opens = prices + rng.uniform(-1, 1, n_bars)
    closes = prices + rng.uniform(-1, 1, n_bars)
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise

    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": rng.integers(100, 5000, n_bars),
    })


@pytest.fixture
def sample_ohlc() -> pd.DataFrame:
    """Create a small sample OHLC DataFrame for testing."""
    n = 100
    rng = np.random.default_rng(42)
    base_price = 15000.0
    returns = rng.normal(0, 0.001, n)
    closes = base_price * np.cumprod(1 + returns)
    highs = closes * (1 + rng.uniform(0, 0.002, n))
    lows = closes * (1 - rng.uniform(0, 0.002, n))
    opens = np.roll(closes, 1)
    opens[0] = base_price

    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": rng.integers(0, 1000, n),
    })
