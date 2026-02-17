"""Integration test: full concept chain on real NAS100 data.

Loads real data -> resamples -> runs all concepts in sequence.
Validates: no errors, reasonable detection counts, data consistency.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.fvg import detect_fvg
from concepts.fractals import detect_swings, get_swing_points
from concepts.liquidity import detect_equal_levels, detect_session_levels
from concepts.structure import detect_structure, detect_cisd
from concepts.registry import build_poi_registry
from concepts.zones import (
    classify_price_zone,
    consequent_encroachment,
    premium_discount_zones,
)
from data.loader import load_instrument
from data.resampler import resample


PARQUET_PATH = Path(__file__).parent.parent.parent / "data" / "optimized"
NAS100_FILE = PARQUET_PATH / "NAS100_m1.parquet"


pytestmark = pytest.mark.skipif(
    not NAS100_FILE.exists(),
    reason=f"NAS100 parquet not found at {NAS100_FILE}",
)


@pytest.fixture(scope="module")
def nas100_1m():
    """Load NAS100 1-minute data (first 50K rows for speed)."""
    df = load_instrument("NAS100")
    return df.head(50000)


@pytest.fixture(scope="module")
def nas100_15m(nas100_1m):
    return resample(nas100_1m, "15m")


@pytest.fixture(scope="module")
def nas100_1h(nas100_1m):
    return resample(nas100_1m, "1H")


class TestDataLoad:
    def test_data_loads(self, nas100_1m):
        assert len(nas100_1m) > 10000
        assert all(c in nas100_1m.columns for c in ["open", "high", "low", "close"])

    def test_no_nan_ohlc(self, nas100_1m):
        ohlc = nas100_1m[["open", "high", "low", "close"]]
        assert ohlc.isna().sum().sum() == 0

    def test_ohlc_consistency(self, nas100_1m):
        assert (nas100_1m["high"] >= nas100_1m["low"]).all()
        assert (nas100_1m["high"] >= nas100_1m["open"]).all()
        assert (nas100_1m["high"] >= nas100_1m["close"]).all()
        assert (nas100_1m["low"] <= nas100_1m["open"]).all()
        assert (nas100_1m["low"] <= nas100_1m["close"]).all()


class TestResample:
    def test_15m_resampled(self, nas100_15m):
        assert len(nas100_15m) > 100
        assert (nas100_15m["high"] >= nas100_15m["low"]).all()

    def test_1h_resampled(self, nas100_1h):
        assert len(nas100_1h) > 50
        assert (nas100_1h["high"] >= nas100_1h["low"]).all()


class TestFractals:
    def test_detect_on_15m(self, nas100_15m):
        swings = detect_swings(nas100_15m, swing_length=5)
        assert len(swings) == len(nas100_15m)
        sh_count = swings["swing_high"].sum()
        sl_count = swings["swing_low"].sum()
        assert sh_count > 5, f"Expected >5 swing highs, got {sh_count}"
        assert sl_count > 5, f"Expected >5 swing lows, got {sl_count}"
        # No overlap
        assert not (swings["swing_high"] & swings["swing_low"]).any()

    def test_swing_points_extraction(self, nas100_15m):
        swings = detect_swings(nas100_15m, swing_length=5)
        points = get_swing_points(nas100_15m, swings)
        assert len(points) > 10
        assert set(points["direction"].unique()) == {1, -1}

    def test_alternation(self, nas100_15m):
        """Swing points should roughly alternate high/low."""
        swings = detect_swings(nas100_15m, swing_length=5)
        points = get_swing_points(nas100_15m, swings)
        dirs = points["direction"].values
        same_consecutive = sum(
            1 for i in range(1, len(dirs)) if dirs[i] == dirs[i - 1]
        )
        # Allow some non-alternation but it shouldn't be dominant
        total = len(dirs) - 1
        if total > 0:
            alternation_ratio = 1 - (same_consecutive / total)
            assert alternation_ratio > 0.3, (
                f"Alternation ratio too low: {alternation_ratio:.2f}"
            )


class TestStructure:
    def test_structure_on_15m(self, nas100_15m):
        events = detect_structure(nas100_15m, swing_length=5, close_break=True)
        assert len(events) > 0, "Expected at least 1 structure event"
        assert "type" in events.columns
        assert "direction" in events.columns
        assert set(events["direction"].unique()).issubset({1, -1})

    def test_bos_and_cbos_exist(self, nas100_15m):
        events = detect_structure(nas100_15m, swing_length=5)
        types = set(str(t) for t in events["type"])
        # On real data, we should see at least CBOS
        assert "CBOS" in types or "StructureType.CBOS" in types, (
            f"No CBOS found in types: {types}"
        )

    def test_cisd_on_1m(self, nas100_1m):
        # Use first 5K rows for speed
        events = detect_cisd(nas100_1m.head(5000))
        assert len(events) > 0, "Expected CISD events on 1m data"
        assert set(events["direction"].unique()).issubset({1, -1})


class TestFVG:
    def test_detect_on_15m(self, nas100_15m):
        fvgs = detect_fvg(nas100_15m, min_gap_pct=0.0005)
        assert len(fvgs) > 0, "Expected FVGs on 15m data"
        assert all(c in fvgs.columns for c in ["direction", "top", "bottom", "midpoint"])
        assert (fvgs["top"] > fvgs["bottom"]).all(), "FVG top must be > bottom"

    def test_fvg_directions(self, nas100_15m):
        fvgs = detect_fvg(nas100_15m, min_gap_pct=0.0005)
        assert set(fvgs["direction"].unique()).issubset({1, -1})

    def test_fvg_midpoint(self, nas100_15m):
        fvgs = detect_fvg(nas100_15m, min_gap_pct=0.0005)
        for _, fvg in fvgs.iterrows():
            expected = (fvg["top"] + fvg["bottom"]) / 2
            assert abs(fvg["midpoint"] - expected) < 1e-6


class TestLiquidity:
    def test_equal_levels(self, nas100_15m):
        levels = detect_equal_levels(nas100_15m, swing_length=5, range_percent=0.002)
        # May or may not find levels depending on data
        assert isinstance(levels, pd.DataFrame)
        if len(levels) > 0:
            assert "level" in levels.columns
            assert "count" in levels.columns
            assert (levels["count"] >= 2).all()

    def test_session_levels(self, nas100_1m):
        df = nas100_1m.head(10000)
        levels = detect_session_levels(df, level_type="daily")
        assert len(levels) > 0, "Expected daily session levels"
        assert (levels["high"] >= levels["low"]).all()


class TestZones:
    def test_premium_discount(self, nas100_15m):
        # Get actual swing range from data
        high = nas100_15m["high"].max()
        low = nas100_15m["low"].min()
        zones = premium_discount_zones(high, low)
        assert zones["equilibrium"] == (high + low) / 2
        assert zones["premium_zone"][0] == zones["equilibrium"]
        assert zones["discount_zone"][1] == zones["equilibrium"]

    def test_classify_current_price(self, nas100_15m):
        high = nas100_15m["high"].max()
        low = nas100_15m["low"].min()
        last_close = nas100_15m["close"].iloc[-1]
        zone = classify_price_zone(last_close, high, low)
        assert zone in ("premium", "discount", "equilibrium")

    def test_ce_calculation(self, nas100_15m):
        fvgs = detect_fvg(nas100_15m, min_gap_pct=0.0005)
        if len(fvgs) > 0:
            row = fvgs.iloc[0]
            ce = consequent_encroachment(row["top"], row["bottom"])
            assert abs(ce - row["midpoint"]) < 1e-6


class TestFullChain:
    """Run the entire concept chain end-to-end."""

    def test_full_pipeline(self, nas100_15m):
        """Complete pipeline: fractals -> structure -> FVG -> liquidity -> zones."""
        # Step 1: Fractals
        swings = detect_swings(nas100_15m, swing_length=5)
        points = get_swing_points(nas100_15m, swings)
        assert len(points) > 0

        # Step 2: Structure
        events = detect_structure(nas100_15m, swing_length=5)
        assert len(events) > 0

        # Step 3: CISD
        cisd = detect_cisd(nas100_15m)
        assert isinstance(cisd, pd.DataFrame)

        # Step 4: FVG
        fvgs = detect_fvg(nas100_15m, min_gap_pct=0.0005)
        assert isinstance(fvgs, pd.DataFrame)

        # Step 5: Liquidity
        eq_levels = detect_equal_levels(nas100_15m, swing_length=5)
        session_levels = detect_session_levels(nas100_15m, level_type="daily")
        assert isinstance(eq_levels, pd.DataFrame)
        assert isinstance(session_levels, pd.DataFrame)

        # Step 6: Zones
        if len(points) >= 2:
            sh = points[points["direction"] == 1]["level"].max()
            sl = points[points["direction"] == -1]["level"].min()
            if sh > sl:
                zones = premium_discount_zones(sh, sl)
                assert "equilibrium" in zones

        # Summary
        print(f"\n--- Concept Chain Summary (15m, {len(nas100_15m)} bars) ---")
        print(f"  Swing points: {len(points)} ({points['direction'].value_counts().to_dict()})")
        print(f"  Structure events: {len(events)}")
        print(f"  CISD events: {len(cisd)}")
        print(f"  FVGs: {len(fvgs)}")
        print(f"  Equal levels: {len(eq_levels)}")
        print(f"  Session levels: {len(session_levels)}")


class TestPOIRegistry:
    """Test POI registry on real data."""

    def test_registry_on_15m(self, nas100_15m):
        """Build POI registry from all detected concepts."""
        fvgs = detect_fvg(nas100_15m, min_gap_pct=0.0003)
        eq_levels = detect_equal_levels(nas100_15m, swing_length=5)
        session_levels = detect_session_levels(nas100_15m, level_type="daily")

        pois = build_poi_registry(
            fvgs, eq_levels, session_levels,
            timeframe="15m",
        )
        assert len(pois) > 0, "Expected POIs on 15m data"
        assert (pois["score"] > 0).all(), "All POIs must have positive score"
        assert set(pois["direction"].unique()).issubset({1, -1})
        assert (pois["top"] > pois["bottom"]).all()

        print("\n--- POI Registry (15m) ---")
        print(f"  Total POIs: {len(pois)}")
        print(f"  Bullish: {len(pois[pois['direction'] == 1])}")
        print(f"  Bearish: {len(pois[pois['direction'] == -1])}")
        print(f"  Score range: {pois['score'].min():.1f} - {pois['score'].max():.1f}")

    def test_poi_has_confluence(self, nas100_15m):
        """At least some POIs should have multiple components."""
        fvgs = detect_fvg(nas100_15m, min_gap_pct=0.0003)
        eq_levels = detect_equal_levels(nas100_15m, swing_length=5)
        session_levels = detect_session_levels(nas100_15m, level_type="daily")

        pois = build_poi_registry(
            fvgs, eq_levels, session_levels,
            timeframe="15m",
        )
        multi = pois[pois["component_count"] >= 2]
        assert len(multi) > 0, "Expected some confluence POIs on real data"
        print(f"  Confluence POIs (2+): {len(multi)}")
        print(f"  Max components: {pois['component_count'].max()}")
