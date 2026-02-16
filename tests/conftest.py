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
