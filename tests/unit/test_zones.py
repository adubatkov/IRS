"""Tests for Premium/Discount zones and CE/CVB calculation."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.zones import (
    classify_price_zone,
    consequent_encroachment,
    premium_discount_zones,
    zone_percentage,
)


class TestPremiumDiscountZones:
    def test_basic_zones(self):
        zones = premium_discount_zones(200, 100)
        assert zones["equilibrium"] == 150
        assert zones["premium_zone"] == (150, 200)
        assert zones["discount_zone"] == (100, 150)

    def test_quarter_levels(self):
        zones = premium_discount_zones(200, 100)
        assert zones["quarter_75"] == 175
        assert zones["quarter_25"] == 125


class TestClassifyPriceZone:
    def test_premium(self):
        assert classify_price_zone(180, 200, 100) == "premium"

    def test_discount(self):
        assert classify_price_zone(120, 200, 100) == "discount"

    def test_equilibrium(self):
        assert classify_price_zone(150, 200, 100) == "equilibrium"


class TestConsequentEncroachment:
    def test_midpoint(self):
        assert consequent_encroachment(110, 100) == 105

    def test_fvg_ce(self):
        assert consequent_encroachment(108, 100) == 104


class TestZonePercentage:
    def test_at_high(self):
        assert zone_percentage(200, 200, 100) == 100.0

    def test_at_low(self):
        assert zone_percentage(100, 200, 100) == 0.0

    def test_at_midpoint(self):
        assert zone_percentage(150, 200, 100) == 50.0
