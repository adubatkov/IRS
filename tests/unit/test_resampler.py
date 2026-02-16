"""Tests for timeframe resampling."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.resampler import resample, resample_all


@pytest.fixture
def ohlc_1m():
    """Create 1-hour of 1m data (60 candles) with known values."""
    n = 60
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(123)
    base = 15000.0
    closes = base + np.cumsum(rng.normal(0, 2, n))
    opens = np.roll(closes, 1)
    opens[0] = base
    highs = np.maximum(opens, closes) + rng.uniform(0, 5, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 5, n)
    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": rng.integers(10, 100, n),
    })


@pytest.fixture
def ohlc_1m_multiday():
    """Create 2 days of 1m data for 1D resampling test."""
    # 2 trading days: concat two separate 8-hour blocks on different days
    n = 960
    day1 = pd.date_range("2024-01-02 08:00", periods=480, freq="1min", tz="UTC")
    day2 = pd.date_range("2024-01-03 08:00", periods=480, freq="1min", tz="UTC")
    dates = day1.append(day2)
    rng = np.random.default_rng(456)
    base = 15000.0
    closes = base + np.cumsum(rng.normal(0, 1, n))
    opens = np.roll(closes, 1)
    opens[0] = base
    highs = np.maximum(opens, closes) + rng.uniform(0, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 3, n)
    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": rng.integers(10, 100, n),
    })


class TestResample:
    def test_1m_returns_copy(self, ohlc_1m):
        result = resample(ohlc_1m, "1m")
        assert len(result) == len(ohlc_1m)
        # Should be a copy, not the same object
        assert result is not ohlc_1m

    def test_5m_reduces_rows(self, ohlc_1m):
        result = resample(ohlc_1m, "5m")
        assert len(result) == 12  # 60 / 5

    def test_15m_reduces_rows(self, ohlc_1m):
        result = resample(ohlc_1m, "15m")
        assert len(result) == 4  # 60 / 15

    def test_1h_single_candle(self, ohlc_1m):
        result = resample(ohlc_1m, "1H")
        assert len(result) == 1

    def test_ohlc_rules_5m(self, ohlc_1m):
        """Verify OHLC aggregation rules on first 5m candle."""
        result = resample(ohlc_1m, "5m")
        first_5 = ohlc_1m.iloc[:5]

        first_candle = result.iloc[0]
        assert first_candle["open"] == first_5["open"].iloc[0], "Open should be first"
        assert first_candle["high"] == first_5["high"].max(), "High should be max"
        assert first_candle["low"] == first_5["low"].min(), "Low should be min"
        assert first_candle["close"] == first_5["close"].iloc[-1], "Close should be last"
        assert first_candle["tick_volume"] == first_5["tick_volume"].sum(), "Volume should be sum"

    def test_1d_resampling(self, ohlc_1m_multiday):
        result = resample(ohlc_1m_multiday, "1D")
        assert len(result) == 2  # 2 days

    def test_unknown_timeframe_raises(self, ohlc_1m):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            resample(ohlc_1m, "3m")

    def test_output_has_time_column(self, ohlc_1m):
        result = resample(ohlc_1m, "5m")
        assert "time" in result.columns

    def test_high_gte_low(self, ohlc_1m):
        result = resample(ohlc_1m, "5m")
        assert (result["high"] >= result["low"]).all()


class TestResampleAll:
    def test_returns_all_timeframes(self, ohlc_1m):
        tfs = ["1m", "5m", "15m"]
        result = resample_all(ohlc_1m, tfs)
        assert set(result.keys()) == set(tfs)
        assert len(result["1m"]) == 60
        assert len(result["5m"]) == 12
        assert len(result["15m"]) == 4


class TestResampleRealData:
    def test_resample_nas100(self):
        """Test resampling on real NAS100 data (first 1000 rows)."""
        path = Path("data/optimized/NAS100_m1.parquet")
        if not path.exists():
            pytest.skip("NAS100 parquet not available")

        from data.loader import load_parquet
        df = load_parquet(path).head(1000)
        result = resample(df, "5m")
        assert len(result) > 0
        assert len(result) < len(df)
        # High >= Low for all candles
        assert (result["high"] >= result["low"]).all()
