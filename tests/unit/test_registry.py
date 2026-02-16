"""Tests for POI Registry — concept aggregation and scoring."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.registry import POIStatus, build_poi_registry, update_poi_status


def _empty_df(columns):
    return pd.DataFrame(columns=columns)


def _empty_fvgs():
    return _empty_df(["direction", "top", "bottom", "midpoint", "start_index",
                       "creation_index", "status"])


def _empty_obs():
    return _empty_df(["direction", "top", "bottom", "ob_index", "trigger_index", "status"])


def _empty_breakers():
    return _empty_df(["direction", "top", "bottom", "original_ob_index", "status"])


def _empty_liquidity():
    return _empty_df(["direction", "level", "count", "indices", "status"])


def _empty_sessions():
    return _empty_df(["period_start", "high", "low"])


class TestBuildPOIRegistry:

    def test_single_fvg_becomes_poi(self):
        """One FVG → one POI with LTF FVG base score."""
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "start_index": 0, "creation_index": 2,
            "status": "FRESH",
        }])
        pois = build_poi_registry(
            fvgs, _empty_obs(), _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        assert len(pois) == 1
        poi = pois.iloc[0]
        assert poi["direction"] == 1
        assert poi["top"] == 108.0
        assert poi["bottom"] == 100.0
        assert poi["component_count"] == 1
        # LTF FVG base=1, FRESH multiplier=1.5, no confluence → 1.5
        assert poi["score"] == 1.5

    def test_overlapping_fvg_and_ob_merged(self):
        """FVG + OB at same zone → merged POI with confluence bonus."""
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "start_index": 0, "creation_index": 2,
            "status": "FRESH",
        }])
        obs = pd.DataFrame([{
            "direction": 1, "top": 107.0, "bottom": 102.0,
            "ob_index": 5, "trigger_index": 10, "status": "ACTIVE",
        }])
        pois = build_poi_registry(
            fvgs, obs, _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        assert len(pois) == 1
        poi = pois.iloc[0]
        assert poi["component_count"] == 2
        # FVG LTF: 1*1.5=1.5, OB: 2*1.5=3.0, confluence(2): +2 → 6.5
        assert poi["score"] == 6.5
        assert poi["top"] == 108.0  # Union of zones
        assert poi["bottom"] == 100.0

    def test_no_merge_different_directions(self):
        """Bullish FVG + bearish OB at same level → 2 separate POIs."""
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "start_index": 0, "creation_index": 2,
            "status": "FRESH",
        }])
        obs = pd.DataFrame([{
            "direction": -1, "top": 107.0, "bottom": 102.0,
            "ob_index": 5, "trigger_index": 10, "status": "ACTIVE",
        }])
        pois = build_poi_registry(
            fvgs, obs, _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        assert len(pois) == 2
        directions = set(pois["direction"].values)
        assert directions == {1, -1}

    def test_scoring_freshness_multiplier(self):
        """Fresh POI scores higher than tested POI."""
        fresh_fvg = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "start_index": 0, "creation_index": 2,
            "status": "FRESH",
        }])
        tested_fvg = pd.DataFrame([{
            "direction": 1, "top": 208.0, "bottom": 200.0,
            "midpoint": 204.0, "start_index": 10, "creation_index": 12,
            "status": "TESTED",
        }])

        fresh_pois = build_poi_registry(
            fresh_fvg, _empty_obs(), _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        tested_pois = build_poi_registry(
            tested_fvg, _empty_obs(), _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        assert fresh_pois.iloc[0]["score"] > tested_pois.iloc[0]["score"]

    def test_mitigated_excluded(self):
        """Mitigated components don't produce POIs."""
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "start_index": 0, "creation_index": 2,
            "status": "MITIGATED",
        }])
        pois = build_poi_registry(
            fvgs, _empty_obs(), _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
        )
        assert len(pois) == 0

    def test_non_overlapping_stay_separate(self):
        """FVG at 100-108 and OB at 120-130 → 2 separate POIs."""
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "start_index": 0, "creation_index": 2,
            "status": "FRESH",
        }])
        obs = pd.DataFrame([{
            "direction": 1, "top": 130.0, "bottom": 120.0,
            "ob_index": 50, "trigger_index": 60, "status": "ACTIVE",
        }])
        pois = build_poi_registry(
            fvgs, obs, _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        assert len(pois) == 2
        assert all(p["component_count"] == 1 for _, p in pois.iterrows())

    def test_empty_inputs(self):
        """All empty DataFrames → empty registry."""
        pois = build_poi_registry(
            _empty_fvgs(), _empty_obs(), _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
        )
        assert len(pois) == 0

    def test_htf_fvg_scores_higher(self):
        """HTF FVG (1H) should score higher than LTF FVG (15m)."""
        fvg = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "start_index": 0, "creation_index": 2,
            "status": "FRESH",
        }])
        htf_pois = build_poi_registry(
            fvg, _empty_obs(), _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
            timeframe="1H",
        )
        ltf_pois = build_poi_registry(
            fvg.copy(), _empty_obs(), _empty_breakers(),
            _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        assert htf_pois.iloc[0]["score"] > ltf_pois.iloc[0]["score"]


class TestUpdatePOIStatus:

    def test_tested_on_wick_touch(self):
        pois = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "score": 5.0,
            "components": [{"type": "fvg_ltf", "source_idx": 0, "status": "FRESH"}],
            "component_count": 1, "status": POIStatus.ACTIVE,
        }])
        updated = update_poi_status(pois, candle_high=112, candle_low=106, candle_close=110)
        assert updated.iloc[0]["status"] == POIStatus.TESTED

    def test_mitigated_on_close_through(self):
        pois = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "score": 5.0,
            "components": [{"type": "fvg_ltf", "source_idx": 0, "status": "FRESH"}],
            "component_count": 1, "status": POIStatus.ACTIVE,
        }])
        updated = update_poi_status(pois, candle_high=105, candle_low=95, candle_close=97)
        assert updated.iloc[0]["status"] == POIStatus.MITIGATED

    def test_bearish_poi_tested(self):
        pois = pd.DataFrame([{
            "direction": -1, "top": 120.0, "bottom": 115.0,
            "midpoint": 117.5, "score": 3.0,
            "components": [{"type": "ob", "source_idx": 0, "status": "ACTIVE"}],
            "component_count": 1, "status": POIStatus.ACTIVE,
        }])
        updated = update_poi_status(pois, candle_high=116, candle_low=110, candle_close=114)
        assert updated.iloc[0]["status"] == POIStatus.TESTED

    def test_bearish_poi_mitigated(self):
        pois = pd.DataFrame([{
            "direction": -1, "top": 120.0, "bottom": 115.0,
            "midpoint": 117.5, "score": 3.0,
            "components": [{"type": "ob", "source_idx": 0, "status": "ACTIVE"}],
            "component_count": 1, "status": POIStatus.ACTIVE,
        }])
        updated = update_poi_status(pois, candle_high=122, candle_low=118, candle_close=121)
        assert updated.iloc[0]["status"] == POIStatus.MITIGATED

    def test_mitigated_not_updated(self):
        pois = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "score": 5.0,
            "components": [], "component_count": 0,
            "status": POIStatus.MITIGATED,
        }])
        updated = update_poi_status(pois, candle_high=200, candle_low=50, candle_close=100)
        assert updated.iloc[0]["status"] == POIStatus.MITIGATED


class TestThreeComponentConfluence:
    """Test 3+ component confluence bonus."""

    def test_three_components_get_bonus(self):
        fvgs = pd.DataFrame([{
            "direction": 1, "top": 108.0, "bottom": 100.0,
            "midpoint": 104.0, "start_index": 0, "creation_index": 2,
            "status": "FRESH",
        }])
        obs = pd.DataFrame([{
            "direction": 1, "top": 107.0, "bottom": 101.0,
            "ob_index": 5, "trigger_index": 10, "status": "ACTIVE",
        }])
        breakers = pd.DataFrame([{
            "direction": 1, "top": 106.0, "bottom": 102.0,
            "original_ob_index": 3, "status": "ACTIVE",
        }])
        pois = build_poi_registry(
            fvgs, obs, breakers,
            _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        assert len(pois) == 1
        poi = pois.iloc[0]
        assert poi["component_count"] == 3
        # FVG: 1*1.5=1.5, OB: 2*1.5=3.0, BB: 2*1.5=3.0, confluence(3+): +4 → 11.5
        assert poi["score"] == 11.5
