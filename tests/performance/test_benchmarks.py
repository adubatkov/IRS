"""Performance benchmarks: each concept module < 2 sec on 100K rows."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.breakers import detect_breakers
from concepts.fvg import detect_fvg
from concepts.fractals import detect_swings, get_swing_points
from concepts.liquidity import detect_equal_levels, detect_session_levels
from concepts.orderblocks import detect_orderblocks
from concepts.structure import detect_bos_choch, detect_cisd
from concepts.zones import premium_discount_zones, classify_price_zone, zone_percentage
from data.loader import load_instrument

PARQUET_PATH = Path(__file__).parent.parent.parent / "data" / "optimized"
NAS100_FILE = PARQUET_PATH / "NAS100_m1.parquet"

MAX_TIME_SECONDS = 2.0

pytestmark = pytest.mark.skipif(
    not NAS100_FILE.exists(),
    reason=f"NAS100 parquet not found at {NAS100_FILE}",
)


@pytest.fixture(scope="module")
def data_100k():
    """Load 100K rows of NAS100 data."""
    df = load_instrument("NAS100")
    return df.head(100000)


class TestBenchmarks:
    def test_fractals_performance(self, data_100k):
        start = time.perf_counter()
        _swings = detect_swings(data_100k, swing_length=5)  # noqa: F841
        elapsed = time.perf_counter() - start
        print(f"\nFractals (100K rows): {elapsed:.3f}s")
        assert elapsed < MAX_TIME_SECONDS, f"Fractals took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_swing_points_performance(self, data_100k):
        swings = detect_swings(data_100k, swing_length=5)
        start = time.perf_counter()
        _points = get_swing_points(data_100k, swings)  # noqa: F841
        elapsed = time.perf_counter() - start
        print(f"\nSwing points extraction (100K rows): {elapsed:.3f}s")
        assert elapsed < MAX_TIME_SECONDS, f"Swing points took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_structure_performance(self, data_100k):
        start = time.perf_counter()
        events = detect_bos_choch(data_100k, swing_length=5)
        elapsed = time.perf_counter() - start
        print(f"\nBOS/CHoCH (100K rows): {elapsed:.3f}s, {len(events)} events")
        assert elapsed < MAX_TIME_SECONDS, f"BOS/CHoCH took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_cisd_performance(self, data_100k):
        start = time.perf_counter()
        events = detect_cisd(data_100k)
        elapsed = time.perf_counter() - start
        print(f"\nCISD (100K rows): {elapsed:.3f}s, {len(events)} events")
        assert elapsed < MAX_TIME_SECONDS, f"CISD took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_fvg_performance(self, data_100k):
        start = time.perf_counter()
        fvgs = detect_fvg(data_100k, min_gap_pct=0.0005)
        elapsed = time.perf_counter() - start
        print(f"\nFVG (100K rows): {elapsed:.3f}s, {len(fvgs)} FVGs")
        assert elapsed < MAX_TIME_SECONDS, f"FVG took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_orderblocks_performance(self, data_100k):
        events = detect_bos_choch(data_100k, swing_length=5)
        start = time.perf_counter()
        obs = detect_orderblocks(data_100k, events)
        elapsed = time.perf_counter() - start
        print(f"\nOrder Blocks (100K rows): {elapsed:.3f}s, {len(obs)} OBs")
        assert elapsed < MAX_TIME_SECONDS, f"Order Blocks took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_liquidity_equal_levels_performance(self, data_100k):
        start = time.perf_counter()
        levels = detect_equal_levels(data_100k, swing_length=5)
        elapsed = time.perf_counter() - start
        print(f"\nEqual Levels (100K rows): {elapsed:.3f}s, {len(levels)} levels")
        assert elapsed < MAX_TIME_SECONDS, f"Equal Levels took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_liquidity_session_levels_performance(self, data_100k):
        start = time.perf_counter()
        levels = detect_session_levels(data_100k, level_type="daily")
        elapsed = time.perf_counter() - start
        print(f"\nSession Levels (100K rows): {elapsed:.3f}s, {len(levels)} levels")
        assert elapsed < MAX_TIME_SECONDS, f"Session Levels took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_zones_performance(self, data_100k):
        start = time.perf_counter()
        # Run classification on many prices
        high = data_100k["high"].max()
        low = data_100k["low"].min()
        _zones = premium_discount_zones(high, low)  # noqa: F841
        for price in data_100k["close"].values[:10000]:
            classify_price_zone(price, high, low)
            zone_percentage(price, high, low)
        elapsed = time.perf_counter() - start
        print(f"\nZones (10K classifications): {elapsed:.3f}s")
        assert elapsed < MAX_TIME_SECONDS, f"Zones took {elapsed:.3f}s > {MAX_TIME_SECONDS}s"

    def test_full_chain_performance(self, data_100k):
        """Entire concept chain on 100K rows."""
        start = time.perf_counter()

        swings = detect_swings(data_100k, swing_length=5)
        points = get_swing_points(data_100k, swings)
        events = detect_bos_choch(data_100k, swing_length=5)
        cisd = detect_cisd(data_100k)
        fvgs = detect_fvg(data_100k, min_gap_pct=0.0005)
        obs = detect_orderblocks(data_100k, events)
        breakers = detect_breakers(obs)
        eq_levels = detect_equal_levels(data_100k, swing_length=5)
        session_levels = detect_session_levels(data_100k, level_type="daily")

        elapsed = time.perf_counter() - start
        print(f"\n--- Full Chain (100K rows): {elapsed:.3f}s ---")
        print(f"  Swings: {len(points)}, Structure: {len(events)}, CISD: {len(cisd)}")
        print(f"  FVGs: {len(fvgs)}, OBs: {len(obs)}, Breakers: {len(breakers)}")
        print(f"  Equal levels: {len(eq_levels)}, Session levels: {len(session_levels)}")

        # Full chain should be under 10 seconds total
        assert elapsed < 10.0, f"Full chain took {elapsed:.3f}s > 10s"
