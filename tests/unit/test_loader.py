"""Tests for data loading utilities."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.loader import (
    _clean_dataframe,
    detect_gaps,
    get_data_stats,
    load_parquet,
    validate_dataframe,
)


@pytest.fixture
def raw_ohlc_df():
    """Create a raw DataFrame with valid OHLC data."""
    n = 50
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    base = 15000.0
    closes = base + np.cumsum(rng.normal(0, 5, n))
    opens = closes + rng.uniform(-3, 3, n)
    # high must be >= max(open, close), low must be <= min(open, close)
    highs = np.maximum(opens, closes) + rng.uniform(0, 10, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 10, n)
    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": rng.integers(0, 500, n),
    })


class TestCleanDataframe:
    def test_normalizes_column_names(self):
        df = pd.DataFrame({
            "Time": ["2024-01-02T09:30:00Z"],
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
        })
        result = _clean_dataframe(df)
        assert "time" in result.columns
        assert "open" in result.columns

    def test_drops_shapes_column(self):
        df = pd.DataFrame({
            "time": ["2024-01-02T09:30:00Z"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "shapes": [0],
        })
        result = _clean_dataframe(df)
        assert "shapes" not in result.columns

    def test_adds_tick_volume_if_missing(self):
        df = pd.DataFrame({
            "time": ["2024-01-02T09:30:00Z"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        })
        result = _clean_dataframe(df)
        assert "tick_volume" in result.columns
        assert result["tick_volume"].iloc[0] == 0

    def test_parses_string_time_to_datetime(self):
        df = pd.DataFrame({
            "time": ["2024-01-02T09:30:00Z", "2024-01-02T09:31:00Z"],
            "open": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 99.5],
            "close": [100.5, 101.0],
        })
        result = _clean_dataframe(df)
        assert pd.api.types.is_datetime64_any_dtype(result["time"])

    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "time": ["2024-01-02T09:30:00Z", "2024-01-02T09:30:00Z"],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 100.5],
        })
        result = _clean_dataframe(df)
        assert len(result) == 1


class TestValidateDataframe:
    def test_valid_df_returns_empty(self, raw_ohlc_df):
        issues = validate_dataframe(raw_ohlc_df)
        assert len(issues) == 0

    def test_missing_columns(self):
        df = pd.DataFrame({"time": [1], "open": [1]})
        issues = validate_dataframe(df)
        assert any("Missing columns" in i for i in issues)

    def test_detects_nan(self, raw_ohlc_df):
        raw_ohlc_df.loc[5, "close"] = np.nan
        issues = validate_dataframe(raw_ohlc_df)
        assert any("NaN" in i for i in issues)


class TestDetectGaps:
    def test_no_gaps_in_continuous_data(self, raw_ohlc_df):
        gaps = detect_gaps(raw_ohlc_df)
        assert len(gaps) == 0

    def test_detects_gap(self):
        dates = list(pd.date_range("2024-01-02 09:30", periods=5, freq="1min", tz="UTC"))
        # Insert a 10-minute gap
        dates.append(dates[-1] + pd.Timedelta(minutes=10))
        df = pd.DataFrame({
            "time": dates,
            "open": [100] * 6,
            "high": [101] * 6,
            "low": [99] * 6,
            "close": [100.5] * 6,
        })
        gaps = detect_gaps(df)
        assert len(gaps) == 1
        assert gaps.iloc[0]["gap_minutes"] == 10.0


class TestGetDataStats:
    def test_returns_expected_keys(self, raw_ohlc_df):
        stats = get_data_stats(raw_ohlc_df)
        assert "rows" in stats
        assert "start" in stats
        assert "end" in stats
        assert "price_min" in stats
        assert stats["rows"] == 50


class TestLoadParquet:
    def test_load_real_parquet(self):
        path = Path("data/optimized/NAS100_m1.parquet")
        if not path.exists():
            pytest.skip("NAS100 parquet file not available")
        df = load_parquet(path)
        assert len(df) > 0
        assert "time" in df.columns
        assert "open" in df.columns
        issues = validate_dataframe(df)
        assert len(issues) == 0, f"Validation issues: {issues}"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_parquet("nonexistent.parquet")
