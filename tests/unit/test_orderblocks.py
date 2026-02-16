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


class TestOrderBlocksSlicedData:
    """Test OB detection on DataFrames with non-zero-based index."""

    def test_detects_ob_on_sliced_dataframe(self):
        n = 50
        opens = np.full(n, 105.0)
        closes = np.full(n, 106.0)
        highs = np.full(n, 107.0)
        lows = np.full(n, 104.0)
        # Make candle at position 14 bearish
        opens[14] = 106.0
        closes[14] = 103.0

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
        # Slice to create non-zero-based index (offset by 500)
        full = pd.concat([pd.DataFrame(np.zeros((500, 4)), columns=df.columns), df],
                         ignore_index=True)
        df_sliced = full.iloc[500:550]  # Indices 500-549

        events = pd.DataFrame([{
            "type": "BOS", "direction": 1,
            "broken_level": 110.0, "broken_index": 530, "swing_index": 515,
        }])

        obs = detect_orderblocks(df_sliced, events)
        assert len(obs) >= 1, "Should detect OB on sliced DataFrame"
        assert obs.iloc[0]["direction"] == 1

    def test_swing_index_not_in_df_skipped(self):
        n = 50
        df = pd.DataFrame({
            "open": np.full(n, 105.0),
            "high": np.full(n, 107.0),
            "low": np.full(n, 104.0),
            "close": np.full(n, 106.0),
        })
        events = pd.DataFrame([{
            "type": "BOS", "direction": 1,
            "broken_level": 110.0, "broken_index": 30, "swing_index": 999,
        }])
        obs = detect_orderblocks(df, events)
        assert len(obs) == 0
