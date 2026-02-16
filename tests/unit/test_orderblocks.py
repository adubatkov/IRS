"""Tests for Order Block detection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.orderblocks import OBStatus, detect_orderblocks, update_ob_status


@pytest.fixture
def simple_structure_events():
    """Bullish BOS event."""
    return pd.DataFrame([{
        "type": "BOS", "direction": 1,
        "broken_level": 110.0, "broken_index": 30, "swing_index": 15,
    }])


@pytest.fixture
def simple_ohlc():
    """OHLC data with a bearish candle at index 14 (before swing at 15)."""
    n = 50
    opens = np.full(n, 105.0)
    closes = np.full(n, 106.0)  # Bullish candles
    highs = np.full(n, 107.0)
    lows = np.full(n, 104.0)
    # Make candle at index 14 bearish (last opposing before bullish break)
    opens[14] = 106.0
    closes[14] = 103.0
    highs[14] = 107.0
    lows[14] = 102.0
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


class TestDetectOrderblocks:
    def test_detects_bullish_ob(self, simple_ohlc, simple_structure_events):
        obs = detect_orderblocks(simple_ohlc, simple_structure_events)
        assert len(obs) >= 1
        assert obs.iloc[0]["direction"] == 1
        assert obs.iloc[0]["status"] == OBStatus.ACTIVE

    def test_empty_on_no_events(self, simple_ohlc):
        empty_events = pd.DataFrame(
            columns=["type", "direction", "broken_level", "broken_index", "swing_index"]
        )
        obs = detect_orderblocks(simple_ohlc, empty_events)
        assert len(obs) == 0

    def test_ob_zone_covers_candle(self, simple_ohlc, simple_structure_events):
        obs = detect_orderblocks(simple_ohlc, simple_structure_events)
        ob = obs.iloc[0]
        assert ob["top"] >= ob["bottom"]


class TestUpdateOBStatus:
    def test_tested_on_touch(self):
        obs = pd.DataFrame([{
            "direction": 1, "top": 107.0, "bottom": 102.0,
            "ob_index": 14, "trigger_index": 30, "status": OBStatus.ACTIVE,
        }])
        updated = update_ob_status(obs, candle_high=110, candle_low=106, candle_close=109)
        assert updated.iloc[0]["status"] == OBStatus.TESTED

    def test_broken_on_close_through(self):
        obs = pd.DataFrame([{
            "direction": 1, "top": 107.0, "bottom": 102.0,
            "ob_index": 14, "trigger_index": 30, "status": OBStatus.ACTIVE,
        }])
        updated = update_ob_status(obs, candle_high=110, candle_low=99, candle_close=100)
        assert updated.iloc[0]["status"] == OBStatus.BROKEN
