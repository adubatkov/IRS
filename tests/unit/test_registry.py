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
            fvgs, _empty_liquidity(), _empty_sessions(),
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
            fresh_fvg, _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        tested_pois = build_poi_registry(
            tested_fvg, _empty_liquidity(), _empty_sessions(),
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
            fvgs, _empty_liquidity(), _empty_sessions(),
        )
        assert len(pois) == 0

    def test_non_overlapping_stay_separate(self):
        """Two FVGs at different price levels stay as 2 separate POIs."""
        fvgs = pd.DataFrame([
            {
                "direction": 1, "top": 108.0, "bottom": 100.0,
                "midpoint": 104.0, "start_index": 0, "creation_index": 2,
                "status": "FRESH",
            },
            {
                "direction": 1, "top": 130.0, "bottom": 120.0,
                "midpoint": 125.0, "start_index": 50, "creation_index": 52,
                "status": "FRESH",
            },
        ])
        pois = build_poi_registry(
            fvgs, _empty_liquidity(), _empty_sessions(),
            timeframe="15m",
        )
        assert len(pois) == 2
        assert all(p["component_count"] == 1 for _, p in pois.iterrows())

    def test_empty_inputs(self):
        """All empty DataFrames → empty registry."""
        pois = build_poi_registry(
            _empty_fvgs(), _empty_liquidity(), _empty_sessions(),
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
            fvg, _empty_liquidity(), _empty_sessions(),
            timeframe="1H",
        )
        ltf_pois = build_poi_registry(
            fvg.copy(), _empty_liquidity(), _empty_sessions(),
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
