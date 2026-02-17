"""Tests for the multi-timeframe data manager (MTFManager)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config
from context.mtf_manager import MTFManager, TimeframeData


def make_synthetic_1m(n_bars: int = 480, base_price: float = 15000.0) -> pd.DataFrame:
    """Create synthetic 1m OHLC data with enough bars for multi-TF resampling.

    480 bars = 8 hours of 1m data, producing:
    - 96 x 5m candles
    - 32 x 15m candles
    - 16 x 30m candles
    - 8 x 1H candles
    - 2 x 4H candles

    Uses a wave pattern to generate realistic swing highs/lows and FVGs.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-02 09:00", periods=n_bars, freq="1min", tz="UTC")

    # Build a trending wave pattern to ensure concept detection works
    prices = np.zeros(n_bars)
    prices[0] = base_price

    # Create waves: alternating up/down legs with overall uptrend
    wave_len = 40
    for i in range(1, n_bars):
        wave_pos = i % wave_len
        # Trend component
        trend = 0.5
        # Wave component: up for first half, down for second half
        if wave_pos < wave_len // 2:
            wave = 3.0
        else:
            wave = -2.5
        prices[i] = prices[i - 1] + trend + wave + rng.normal(0, 1.0)

    # Build OHLC from price path
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
def config() -> Config:
    """Default config with a reduced set of timeframes for faster tests."""
    cfg = Config()
    # Use a subset of timeframes for faster testing
    cfg.data.timeframes = ["1m", "5m", "15m", "1H"]
    return cfg


@pytest.fixture
def config_all_tfs() -> Config:
    """Config with all 7 timeframes."""
    return Config()


@pytest.fixture
def df_1m() -> pd.DataFrame:
    """Synthetic 1m data: 480 bars."""
    return make_synthetic_1m(n_bars=480)


@pytest.fixture
def manager(config: Config, df_1m: pd.DataFrame) -> MTFManager:
    """Initialized MTFManager with 4 timeframes."""
    mgr = MTFManager(config)
    mgr.initialize(df_1m)
    return mgr


class TestInitialize:

    def test_initialize_creates_all_timeframes(self, config, df_1m):
        """All configured timeframes should be populated after initialize."""
        mgr = MTFManager(config)
        mgr.initialize(df_1m)

        for tf in config.data.timeframes:
            assert tf in mgr._data, f"Timeframe {tf} should be present"
            assert isinstance(mgr._data[tf], TimeframeData)

    def test_candle_counts_reasonable(self, manager, df_1m):
        """Resampled candle counts should be roughly correct ratios."""
        n_1m = len(df_1m)
        assert len(manager._data["1m"].candles) == n_1m

        n_5m = len(manager._data["5m"].candles)
        # 480 / 5 = 96 (approximately)
        assert 90 <= n_5m <= 100, f"Expected ~96 5m candles, got {n_5m}"

        n_15m = len(manager._data["15m"].candles)
        # 480 / 15 = 32
        assert 28 <= n_15m <= 36, f"Expected ~32 15m candles, got {n_15m}"

        n_1h = len(manager._data["1H"].candles)
        # 480 / 60 = 8
        assert 6 <= n_1h <= 10, f"Expected ~8 1H candles, got {n_1h}"

    def test_timeframe_data_has_all_fields(self, manager):
        """All TimeframeData fields should be populated (not None)."""
        for tf in manager._data:
            td = manager._data[tf]
            assert td.candles is not None
            assert isinstance(td.candles, pd.DataFrame)
            assert len(td.candles) > 0

            assert td.swings is not None
            assert isinstance(td.swings, pd.DataFrame)

            assert td.swing_points is not None
            assert isinstance(td.swing_points, pd.DataFrame)

            assert td.structure is not None
            assert isinstance(td.structure, pd.DataFrame)

            assert td.cisd is not None
            assert isinstance(td.cisd, pd.DataFrame)

            assert td.fvgs is not None
            assert isinstance(td.fvgs, pd.DataFrame)

            assert td.fvg_lifecycle is not None
            assert isinstance(td.fvg_lifecycle, list)

            assert td.liquidity is not None
            assert isinstance(td.liquidity, pd.DataFrame)

            assert td.session_levels is not None
            assert isinstance(td.session_levels, pd.DataFrame)

            assert td.pois is not None
            assert isinstance(td.pois, pd.DataFrame)


class TestGetCandleAt:

    def test_get_candle_at_no_lookahead(self, manager, df_1m):
        """Candle at timestamp T should not include data from after T."""
        # Pick a timestamp in the middle of the data
        mid_idx = len(df_1m) // 2
        mid_time = df_1m["time"].iloc[mid_idx]

        candle = manager.get_candle_at("1m", mid_time)
        assert candle is not None
        assert candle["time"] <= mid_time

        # For higher TF, same principle
        candle_5m = manager.get_candle_at("5m", mid_time)
        if candle_5m is not None:
            assert candle_5m["time"] <= mid_time

    def test_get_candle_at_returns_none_before_data(self, manager):
        """Returns None for timestamp before any data."""
        early_time = pd.Timestamp("2020-01-01", tz="UTC")
        result = manager.get_candle_at("1m", early_time)
        assert result is None

    def test_get_candle_at_returns_latest_closed(self, manager, df_1m):
        """Should return the most recent closed candle, not a future one."""
        # Use a timestamp between two 1m bars
        bar_time = df_1m["time"].iloc[10]
        candle = manager.get_candle_at("1m", bar_time)
        assert candle is not None
        assert candle["time"] == bar_time

        # One second before that bar should return the previous bar
        just_before = bar_time - pd.Timedelta(seconds=1)
        candle_before = manager.get_candle_at("1m", just_before)
        assert candle_before is not None
        assert candle_before["time"] == df_1m["time"].iloc[9]

    def test_get_candle_at_unknown_tf_raises(self, manager):
        """Accessing an unknown timeframe should raise KeyError."""
        with pytest.raises(KeyError):
            manager.get_candle_at("2m", pd.Timestamp("2024-01-02 10:00", tz="UTC"))


class TestTfJustClosed:

    def test_tf_just_closed_1m(self, manager, df_1m):
        """For 1m, tf_just_closed should always return True."""
        for i in range(min(20, len(df_1m))):
            ts = df_1m["time"].iloc[i]
            assert manager.tf_just_closed("1m", ts) is True

    def test_tf_just_closed_15m(self, manager, df_1m):
        """Should fire at 15-minute boundaries."""
        # The 15m candle opens at :00, :15, :30, :45
        # A 1m bar at :14 means the next minute is :15 (a 15m boundary), so it fires
        # A 1m bar at :10 means the next minute is :11 (not a boundary), so it doesn't

        # Find a timestamp at minute 14 (last bar of 15m candle starting at :00)
        at_14 = pd.Timestamp("2024-01-02 09:14", tz="UTC")
        at_10 = pd.Timestamp("2024-01-02 09:10", tz="UTC")
        at_29 = pd.Timestamp("2024-01-02 09:29", tz="UTC")

        # :14 + 1min = :15 which is a 15m boundary
        assert manager.tf_just_closed("15m", at_14) is True
        # :10 + 1min = :11 which is not a 15m boundary
        assert manager.tf_just_closed("15m", at_10) is False
        # :29 + 1min = :30 which is a 15m boundary
        assert manager.tf_just_closed("15m", at_29) is True

    def test_tf_just_closed_1h(self, manager, df_1m):
        """Should fire at 1-hour boundaries."""
        # :59 + 1min = next hour, which is a 1H boundary
        at_59 = pd.Timestamp("2024-01-02 09:59", tz="UTC")
        at_30 = pd.Timestamp("2024-01-02 09:30", tz="UTC")

        assert manager.tf_just_closed("1H", at_59) is True
        assert manager.tf_just_closed("1H", at_30) is False

    def test_tf_just_closed_unknown_tf_raises(self, manager):
        """Accessing unknown timeframe should raise KeyError."""
        with pytest.raises(KeyError):
            manager.tf_just_closed("2m", pd.Timestamp("2024-01-02 10:00", tz="UTC"))


class TestPOIFiltering:

    def test_get_pois_at_filters(self, manager, df_1m):
        """POIs not yet created should not be returned at an earlier timestamp."""
        # Use a very early timestamp - should get fewer or zero POIs
        early_time = df_1m["time"].iloc[5]
        pois_early = manager.get_pois_at("1m", early_time)

        # Use a late timestamp - should get all POIs
        late_time = df_1m["time"].iloc[-1]
        pois_late = manager.get_pois_at("1m", late_time)

        # Early should have <= POIs than late
        assert len(pois_early) <= len(pois_late)

    def test_get_pois_at_empty_for_early_timestamp(self, manager, df_1m):
        """Before any data, should return empty POIs."""
        early = pd.Timestamp("2020-01-01", tz="UTC")
        pois = manager.get_pois_at("1m", early)
        assert len(pois) == 0

    def test_get_pois_at_unknown_tf_raises(self, manager):
        """Accessing unknown timeframe should raise KeyError."""
        with pytest.raises(KeyError):
            manager.get_pois_at("2m", pd.Timestamp("2024-01-02 10:00", tz="UTC"))

    def test_get_all_active_pois_aggregates(self, manager, df_1m):
        """Aggregated POIs should have timeframe column and come from multiple TFs."""
        late_time = df_1m["time"].iloc[-1]
        all_pois = manager.get_all_active_pois(late_time)

        if len(all_pois) > 0:
            assert "timeframe" in all_pois.columns
            # Should include data from at least one timeframe
            assert len(all_pois["timeframe"].unique()) >= 1

    def test_get_all_active_pois_empty_for_early(self, manager):
        """Before data, should return empty DataFrame with correct columns."""
        early = pd.Timestamp("2020-01-01", tz="UTC")
        result = manager.get_all_active_pois(early)
        assert "timeframe" in result.columns
        assert len(result) == 0


class TestStructureAndFVGFiltering:

    def test_get_structure_at_filters(self, manager, df_1m):
        """Structure events after timestamp should not be returned."""
        early_time = df_1m["time"].iloc[10]
        late_time = df_1m["time"].iloc[-1]

        struct_early = manager.get_structure_at("1m", early_time)
        struct_late = manager.get_structure_at("1m", late_time)

        assert len(struct_early) <= len(struct_late)

    def test_get_fvgs_at_filters(self, manager, df_1m):
        """FVGs created after timestamp should not be returned."""
        early_time = df_1m["time"].iloc[10]
        late_time = df_1m["time"].iloc[-1]

        fvgs_early = manager.get_fvgs_at("1m", early_time)
        fvgs_late = manager.get_fvgs_at("1m", late_time)

        assert len(fvgs_early) <= len(fvgs_late)

    def test_get_structure_at_unknown_tf_raises(self, manager):
        """Accessing unknown timeframe should raise KeyError."""
        with pytest.raises(KeyError):
            manager.get_structure_at("2m", pd.Timestamp("2024-01-02 10:00", tz="UTC"))

    def test_get_fvgs_at_unknown_tf_raises(self, manager):
        """Accessing unknown timeframe should raise KeyError."""
        with pytest.raises(KeyError):
            manager.get_fvgs_at("2m", pd.Timestamp("2024-01-02 10:00", tz="UTC"))


class TestGetTimeframeData:

    def test_get_existing_tf(self, manager):
        """Should return TimeframeData for existing timeframe."""
        td = manager.get_timeframe_data("1m")
        assert isinstance(td, TimeframeData)

    def test_get_unknown_tf_raises(self, manager):
        """Should raise KeyError for unknown timeframe."""
        with pytest.raises(KeyError, match="Timeframe '2m' not found"):
            manager.get_timeframe_data("2m")


class TestEdgeCases:

    def test_small_dataset(self):
        """MTFManager should handle small datasets without crashing."""
        cfg = Config()
        cfg.data.timeframes = ["1m", "5m"]

        rng = np.random.default_rng(99)
        n = 30
        dates = pd.date_range("2024-01-02 09:00", periods=n, freq="1min", tz="UTC")
        prices = 15000.0 + np.cumsum(rng.normal(0, 5, n))
        df = pd.DataFrame({
            "time": dates,
            "open": prices,
            "high": prices + rng.uniform(1, 5, n),
            "low": prices - rng.uniform(1, 5, n),
            "close": prices + rng.uniform(-2, 2, n),
            "tick_volume": rng.integers(10, 100, n),
        })

        mgr = MTFManager(cfg)
        mgr.initialize(df)

        assert "1m" in mgr._data
        assert "5m" in mgr._data
        assert len(mgr._data["1m"].candles) == n

    def test_all_seven_timeframes(self, config_all_tfs):
        """Should handle all 7 default timeframes with enough data."""
        # Need 1440 bars (24 hours) for 1D candle
        df = make_synthetic_1m(n_bars=1500)

        mgr = MTFManager(config_all_tfs)
        mgr.initialize(df)

        for tf in config_all_tfs.data.timeframes:
            assert tf in mgr._data, f"Timeframe {tf} should be present"
            assert len(mgr._data[tf].candles) > 0

    def test_1m_candles_unchanged(self, manager, df_1m):
        """1m candles should be identical to input (just a copy)."""
        candles_1m = manager._data["1m"].candles
        assert len(candles_1m) == len(df_1m)
        pd.testing.assert_frame_equal(
            candles_1m.reset_index(drop=True),
            df_1m.reset_index(drop=True),
        )
