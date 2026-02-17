"""Check alignment between HTF and LTF directional bias."""

from strategy.types import Bias, SyncMode
from config import RiskConfig


def check_sync(
    htf_bias: Bias,
    ltf_bias: Bias,
) -> SyncMode:
    """Determine if HTF and LTF are synchronized.

    Rules (from strategy spec section 13.2):
    - Both BULLISH -> SYNC
    - Both BEARISH -> SYNC
    - One BULLISH, other BEARISH -> DESYNC
    - Either UNDEFINED -> UNDEFINED

    Args:
        htf_bias: Higher time-frame directional bias.
        ltf_bias: Lower time-frame directional bias.

    Returns:
        SyncMode enum value.
    """
    if htf_bias == Bias.UNDEFINED or ltf_bias == Bias.UNDEFINED:
        return SyncMode.UNDEFINED

    if htf_bias == ltf_bias:
        return SyncMode.SYNC

    return SyncMode.DESYNC


def get_position_size_multiplier(
    sync_mode: SyncMode,
    risk_config: RiskConfig,
) -> float:
    """Get position size multiplier based on sync mode.

    Args:
        sync_mode: Current synchronization state.
        risk_config: Risk configuration with size multipliers.

    Returns:
        ``risk_config.position_size_sync`` (default 1.0) for SYNC,
        ``risk_config.position_size_desync`` (default 0.5) for DESYNC,
        0.0 for UNDEFINED (no trading).
    """
    if sync_mode == SyncMode.SYNC:
        return risk_config.position_size_sync
    if sync_mode == SyncMode.DESYNC:
        return risk_config.position_size_desync
    return 0.0


def get_target_mode(
    sync_mode: SyncMode,
) -> str:
    """Determine target selection mode based on sync.

    Args:
        sync_mode: Current synchronization state.

    Returns:
        ``"distant"`` for SYNC -- use primary 4H/1H targets.
        ``"local"`` for DESYNC -- use local 15m/30m fractals only.
        ``"none"`` for UNDEFINED -- no target.
    """
    if sync_mode == SyncMode.SYNC:
        return "distant"
    if sync_mode == SyncMode.DESYNC:
        return "local"
    return "none"
