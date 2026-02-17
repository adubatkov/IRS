"""Tests for HTF/LTF sync checking and position sizing."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RiskConfig
from context.sync_checker import check_sync, get_position_size_multiplier, get_target_mode
from strategy.types import Bias, SyncMode


class TestCheckSync:
    def test_both_bullish_is_sync(self):
        assert check_sync(Bias.BULLISH, Bias.BULLISH) == SyncMode.SYNC

    def test_both_bearish_is_sync(self):
        assert check_sync(Bias.BEARISH, Bias.BEARISH) == SyncMode.SYNC

    def test_bullish_bearish_is_desync(self):
        assert check_sync(Bias.BULLISH, Bias.BEARISH) == SyncMode.DESYNC

    def test_bearish_bullish_is_desync(self):
        assert check_sync(Bias.BEARISH, Bias.BULLISH) == SyncMode.DESYNC

    def test_undefined_htf_is_undefined(self):
        assert check_sync(Bias.UNDEFINED, Bias.BULLISH) == SyncMode.UNDEFINED

    def test_undefined_ltf_is_undefined(self):
        assert check_sync(Bias.BULLISH, Bias.UNDEFINED) == SyncMode.UNDEFINED

    def test_both_undefined_is_undefined(self):
        assert check_sync(Bias.UNDEFINED, Bias.UNDEFINED) == SyncMode.UNDEFINED


class TestGetPositionSizeMultiplier:
    def test_sync_multiplier(self):
        config = RiskConfig()
        assert get_position_size_multiplier(SyncMode.SYNC, config) == 1.0

    def test_desync_multiplier(self):
        config = RiskConfig()
        assert get_position_size_multiplier(SyncMode.DESYNC, config) == 0.5

    def test_undefined_multiplier(self):
        config = RiskConfig()
        assert get_position_size_multiplier(SyncMode.UNDEFINED, config) == 0.0

    def test_custom_risk_config(self):
        """Non-default multipliers from config are respected."""
        config = RiskConfig(
            position_size_sync=0.8,
            position_size_desync=0.3,
        )
        assert get_position_size_multiplier(SyncMode.SYNC, config) == 0.8
        assert get_position_size_multiplier(SyncMode.DESYNC, config) == 0.3
        # UNDEFINED always returns 0.0 regardless of config
        assert get_position_size_multiplier(SyncMode.UNDEFINED, config) == 0.0


class TestGetTargetMode:
    def test_target_mode_sync(self):
        assert get_target_mode(SyncMode.SYNC) == "distant"

    def test_target_mode_desync(self):
        assert get_target_mode(SyncMode.DESYNC) == "local"

    def test_target_mode_undefined(self):
        assert get_target_mode(SyncMode.UNDEFINED) == "none"
