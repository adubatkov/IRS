"""Shared types for the strategy and context modules.

Defines all enums, dataclasses, and type aliases used across Phase 3.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class Bias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    UNDEFINED = "UNDEFINED"


class SyncMode(str, Enum):
    SYNC = "SYNC"
    DESYNC = "DESYNC"
    UNDEFINED = "UNDEFINED"


class POIPhase(str, Enum):
    IDLE = "IDLE"
    POI_TAPPED = "POI_TAPPED"
    COLLECTING = "COLLECTING"
    READY = "READY"
    POSITIONED = "POSITIONED"
    MANAGING = "MANAGING"
    CLOSED = "CLOSED"


class ConfirmationType(str, Enum):
    POI_TAP = "POI_TAP"
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    FVG_INVERSION = "FVG_INVERSION"
    INVERSION_TEST = "INVERSION_TEST"
    STRUCTURE_BREAK = "STRUCTURE_BREAK"
    FVG_WICK_REACTION = "FVG_WICK_REACTION"
    CVB_TEST = "CVB_TEST"
    ADDITIONAL_CBOS = "ADDITIONAL_CBOS"


class SignalType(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    MODIFY_SL = "MODIFY_SL"
    ADD_ON = "ADD_ON"
    MOVE_TO_BE = "MOVE_TO_BE"


class ExitReason(str, Enum):
    TARGET_HIT = "TARGET_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    BREAKEVEN_HIT = "BREAKEVEN_HIT"
    FTA_VALIDATED = "FTA_VALIDATED"
    POI_INVALIDATED = "POI_INVALIDATED"
    FLIP = "FLIP"
    MAX_AGE = "MAX_AGE"


@dataclass
class Confirmation:
    """A single confirmation event detected inside a POI interaction."""

    type: ConfirmationType
    timestamp: pd.Timestamp
    bar_index: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """A trading signal produced by the strategy layer.

    Phase 4 (Backtest Engine) consumes these to execute trades.
    """

    type: SignalType
    poi_id: str
    direction: int  # +1 long, -1 short
    timestamp: pd.Timestamp
    price: float
    stop_loss: float = 0.0
    target: float = 0.0
    position_size_mult: float = 1.0  # 1.0 sync, 0.5 desync
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class POIState:
    """State tracking for a single POI through its lifecycle.

    Managed by the StateMachineManager in context/state_machine.py.
    """

    poi_id: str
    poi_data: dict[str, Any]  # direction, top, bottom, midpoint, score, timeframe, components
    phase: POIPhase = POIPhase.IDLE
    confirmations: list[Confirmation] = field(default_factory=list)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    breakeven_level: Optional[float] = None
    fta: Optional[dict[str, Any]] = None
    addons: list[dict[str, Any]] = field(default_factory=list)
    created_at: Optional[pd.Timestamp] = None
    last_updated: Optional[pd.Timestamp] = None
