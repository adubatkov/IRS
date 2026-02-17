"""Tests for position sizing, stop-loss placement, and risk calculations."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RiskConfig
from strategy.risk import (
    calculate_breakeven_level,
    calculate_position_size,
    calculate_stop_loss,
    validate_risk,
)
from strategy.types import SyncMode


def _make_fvgs(rows: list[dict]) -> pd.DataFrame:
    """Build an FVG DataFrame from a list of dicts."""
    cols = ["direction", "top", "bottom", "midpoint", "start_index",
            "creation_index", "status"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def _make_liquidity(rows: list[dict]) -> pd.DataFrame:
    """Build a liquidity DataFrame from a list of dicts."""
    cols = ["direction", "level", "count", "indices", "status"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def _make_poi(direction: int, top: float, bottom: float) -> dict:
    """Build a minimal POI dict for stop-loss tests."""
    return {
        "direction": direction,
        "top": top,
        "bottom": bottom,
        "midpoint": (top + bottom) / 2,
    }


class TestCalculateStopLoss:
    def test_behind_poi_long(self):
        """LONG: SL placed below POI bottom with buffer."""
        poi = _make_poi(direction=1, top=105.0, bottom=100.0)
        sl = calculate_stop_loss(poi, direction=1, nearby_fvgs=_make_fvgs([]),
                                 nearby_liquidity=_make_liquidity([]),
                                 method="behind_poi")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(100.0 - buffer)

    def test_behind_poi_short(self):
        """SHORT: SL placed above POI top with buffer."""
        poi = _make_poi(direction=-1, top=105.0, bottom=100.0)
        sl = calculate_stop_loss(poi, direction=-1, nearby_fvgs=_make_fvgs([]),
                                 nearby_liquidity=_make_liquidity([]),
                                 method="behind_poi")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(105.0 + buffer)

    def test_behind_fvg_long(self):
        """LONG: SL below nearest bullish FVG bottom."""
        poi = _make_poi(direction=1, top=105.0, bottom=100.0)
        fvgs = _make_fvgs([
            {"direction": 1, "top": 99.0, "bottom": 97.0, "midpoint": 98.0,
             "start_index": 0, "creation_index": 2, "status": "FRESH"},
        ])
        sl = calculate_stop_loss(poi, direction=1, nearby_fvgs=fvgs,
                                 nearby_liquidity=_make_liquidity([]),
                                 method="behind_fvg")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(97.0 - buffer)

    def test_behind_fvg_short(self):
        """SHORT: SL above nearest bearish FVG top."""
        poi = _make_poi(direction=-1, top=105.0, bottom=100.0)
        fvgs = _make_fvgs([
            {"direction": -1, "top": 107.0, "bottom": 106.0, "midpoint": 106.5,
             "start_index": 0, "creation_index": 2, "status": "FRESH"},
        ])
        sl = calculate_stop_loss(poi, direction=-1, nearby_fvgs=fvgs,
                                 nearby_liquidity=_make_liquidity([]),
                                 method="behind_fvg")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(107.0 + buffer)

    def test_behind_liquidity_long(self):
        """LONG: SL below nearest sell-side liquidity level."""
        poi = _make_poi(direction=1, top=105.0, bottom=100.0)
        liq = _make_liquidity([
            {"direction": -1, "level": 98.5, "count": 3, "indices": [10, 20, 30],
             "status": "ACTIVE"},
        ])
        sl = calculate_stop_loss(poi, direction=1, nearby_fvgs=_make_fvgs([]),
                                 nearby_liquidity=liq,
                                 method="behind_liquidity")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(98.5 - buffer)

    def test_behind_liquidity_short(self):
        """SHORT: SL above nearest buy-side liquidity level."""
        poi = _make_poi(direction=-1, top=105.0, bottom=100.0)
        liq = _make_liquidity([
            {"direction": 1, "level": 107.0, "count": 2, "indices": [5, 15],
             "status": "ACTIVE"},
        ])
        sl = calculate_stop_loss(poi, direction=-1, nearby_fvgs=_make_fvgs([]),
                                 nearby_liquidity=liq,
                                 method="behind_liquidity")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(107.0 + buffer)

    def test_fallback_to_poi(self):
        """When no FVGs available for behind_fvg, falls back to behind_poi."""
        poi = _make_poi(direction=1, top=105.0, bottom=100.0)
        sl = calculate_stop_loss(poi, direction=1, nearby_fvgs=_make_fvgs([]),
                                 nearby_liquidity=_make_liquidity([]),
                                 method="behind_fvg")
        buffer = 0.0005 * poi["midpoint"]
        # Should fall back to behind_poi
        assert sl == pytest.approx(100.0 - buffer)

    def test_behind_cvb(self):
        """LONG: SL behind FVG midpoint (consequent encroachment)."""
        poi = _make_poi(direction=1, top=105.0, bottom=100.0)
        fvgs = _make_fvgs([
            {"direction": 1, "top": 99.0, "bottom": 97.0, "midpoint": 98.0,
             "start_index": 0, "creation_index": 2, "status": "FRESH"},
        ])
        sl = calculate_stop_loss(poi, direction=1, nearby_fvgs=fvgs,
                                 nearby_liquidity=_make_liquidity([]),
                                 method="behind_cvb")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(98.0 - buffer)

    def test_behind_cvb_short(self):
        """SHORT: SL behind FVG midpoint."""
        poi = _make_poi(direction=-1, top=105.0, bottom=100.0)
        fvgs = _make_fvgs([
            {"direction": -1, "top": 107.0, "bottom": 105.0, "midpoint": 106.0,
             "start_index": 0, "creation_index": 2, "status": "FRESH"},
        ])
        sl = calculate_stop_loss(poi, direction=-1, nearby_fvgs=fvgs,
                                 nearby_liquidity=_make_liquidity([]),
                                 method="behind_cvb")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(106.0 + buffer)

    def test_unknown_method_falls_back(self):
        """Unknown method name falls back to behind_poi."""
        poi = _make_poi(direction=1, top=105.0, bottom=100.0)
        sl = calculate_stop_loss(poi, direction=1, nearby_fvgs=_make_fvgs([]),
                                 nearby_liquidity=_make_liquidity([]),
                                 method="unknown_method")
        buffer = 0.0005 * poi["midpoint"]
        assert sl == pytest.approx(100.0 - buffer)


class TestCalculatePositionSize:
    def test_basic_calculation(self):
        """Basic position sizing: 10000 equity, 2% risk, known distance."""
        config = RiskConfig()  # max_risk_per_trade=0.02
        size = calculate_position_size(
            account_equity=10000.0,
            entry_price=100.0,
            stop_loss=99.0,
            sync_mode=SyncMode.SYNC,
            risk_config=config,
        )
        # risk = 10000 * 0.02 = 200, distance = 1.0, sync=1.0 -> 200/1 * 1 = 200
        assert size == pytest.approx(200.0)

    def test_sync_full_size(self):
        """SYNC mode gives full position size."""
        config = RiskConfig()
        size = calculate_position_size(
            account_equity=10000.0,
            entry_price=100.0,
            stop_loss=98.0,
            sync_mode=SyncMode.SYNC,
            risk_config=config,
        )
        # risk=200, distance=2.0, sync=1.0 -> 100.0
        assert size == pytest.approx(100.0)

    def test_desync_half_size(self):
        """DESYNC mode halves the position size."""
        config = RiskConfig()
        size = calculate_position_size(
            account_equity=10000.0,
            entry_price=100.0,
            stop_loss=98.0,
            sync_mode=SyncMode.DESYNC,
            risk_config=config,
        )
        # risk=200, distance=2.0, desync=0.5 -> 100 * 0.5 = 50.0
        assert size == pytest.approx(50.0)

    def test_undefined_zero(self):
        """UNDEFINED mode produces zero position size."""
        config = RiskConfig()
        size = calculate_position_size(
            account_equity=10000.0,
            entry_price=100.0,
            stop_loss=98.0,
            sync_mode=SyncMode.UNDEFINED,
            risk_config=config,
        )
        assert size == 0.0

    def test_zero_distance_returns_zero(self):
        """Zero distance between entry and SL returns zero (avoid division by zero)."""
        config = RiskConfig()
        size = calculate_position_size(
            account_equity=10000.0,
            entry_price=100.0,
            stop_loss=100.0,
            sync_mode=SyncMode.SYNC,
            risk_config=config,
        )
        assert size == 0.0

    def test_custom_risk_config(self):
        """Custom risk config values are respected."""
        config = RiskConfig(
            max_risk_per_trade=0.01,
            position_size_sync=0.8,
            position_size_desync=0.4,
        )
        size = calculate_position_size(
            account_equity=20000.0,
            entry_price=100.0,
            stop_loss=99.0,
            sync_mode=SyncMode.SYNC,
            risk_config=config,
        )
        # risk=20000*0.01=200, distance=1.0, sync=0.8 -> 200*0.8=160
        assert size == pytest.approx(160.0)


class TestValidateRisk:
    def test_valid_long_3rr(self):
        """Long with 3:1 RR passes minimum 2.0 check."""
        is_valid, rr = validate_risk(
            entry_price=100.0, stop_loss=99.0, target=103.0, direction=1,
        )
        assert is_valid is True
        assert rr == pytest.approx(3.0)

    def test_invalid_long_low_rr(self):
        """Long with 1:1 RR fails minimum 2.0 check."""
        is_valid, rr = validate_risk(
            entry_price=100.0, stop_loss=99.0, target=101.0, direction=1,
        )
        assert is_valid is False
        assert rr == pytest.approx(1.0)

    def test_valid_short(self):
        """Short with 3:1 RR passes minimum 2.0 check."""
        is_valid, rr = validate_risk(
            entry_price=100.0, stop_loss=101.0, target=97.0, direction=-1,
        )
        assert is_valid is True
        assert rr == pytest.approx(3.0)

    def test_invalid_short_low_rr(self):
        """Short with 1.5:1 RR fails minimum 2.0 check."""
        is_valid, rr = validate_risk(
            entry_price=100.0, stop_loss=102.0, target=97.0, direction=-1,
        )
        assert is_valid is False
        assert rr == pytest.approx(1.5)

    def test_exact_minimum_rr(self):
        """RR exactly at minimum threshold passes."""
        is_valid, rr = validate_risk(
            entry_price=100.0, stop_loss=99.0, target=102.0, direction=1,
            min_rr=2.0,
        )
        assert is_valid is True
        assert rr == pytest.approx(2.0)

    def test_zero_risk_returns_invalid(self):
        """Zero risk distance (entry == SL) returns invalid."""
        is_valid, rr = validate_risk(
            entry_price=100.0, stop_loss=100.0, target=110.0, direction=1,
        )
        assert is_valid is False
        assert rr == 0.0

    def test_negative_risk_returns_invalid(self):
        """SL on wrong side (negative risk) returns invalid."""
        # Long with SL above entry
        is_valid, rr = validate_risk(
            entry_price=100.0, stop_loss=101.0, target=110.0, direction=1,
        )
        assert is_valid is False
        assert rr == 0.0

    def test_custom_min_rr(self):
        """Custom minimum RR threshold is respected."""
        is_valid, rr = validate_risk(
            entry_price=100.0, stop_loss=99.0, target=102.5, direction=1,
            min_rr=3.0,
        )
        assert is_valid is False
        assert rr == pytest.approx(2.5)


class TestCalculateBreakevenLevel:
    def test_long_breakeven(self):
        """Long: breakeven is above entry by 2x commission."""
        be = calculate_breakeven_level(entry_price=100.0, direction=1,
                                       commission_pct=0.0006)
        # 100 * (1 + 2*0.0006) = 100 * 1.0012 = 100.12
        assert be == pytest.approx(100.12)

    def test_short_breakeven(self):
        """Short: breakeven is below entry by 2x commission."""
        be = calculate_breakeven_level(entry_price=100.0, direction=-1,
                                       commission_pct=0.0006)
        # 100 * (1 - 2*0.0006) = 100 * 0.9988 = 99.88
        assert be == pytest.approx(99.88)

    def test_custom_commission(self):
        """Custom commission rate is applied correctly."""
        be = calculate_breakeven_level(entry_price=1000.0, direction=1,
                                       commission_pct=0.001)
        # 1000 * (1 + 2*0.001) = 1000 * 1.002 = 1002.0
        assert be == pytest.approx(1002.0)

    def test_zero_commission(self):
        """Zero commission means breakeven equals entry."""
        be_long = calculate_breakeven_level(entry_price=100.0, direction=1,
                                            commission_pct=0.0)
        be_short = calculate_breakeven_level(entry_price=100.0, direction=-1,
                                             commission_pct=0.0)
        assert be_long == pytest.approx(100.0)
        assert be_short == pytest.approx(100.0)
