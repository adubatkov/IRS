"""Lightweight event log for backtest audit trail."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class EventType(str, Enum):
    POI_REGISTERED = "POI_REGISTERED"
    POI_TAPPED = "POI_TAPPED"
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    BE_MOVED = "BE_MOVED"
    SL_MODIFIED = "SL_MODIFIED"
    ADDON = "ADDON"
    BIAS_UPDATED = "BIAS_UPDATED"
    SYNC_UPDATED = "SYNC_UPDATED"
    POSITION_REJECTED = "POSITION_REJECTED"


@dataclass
class Event:
    type: EventType
    timestamp: pd.Timestamp
    poi_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class EventLog:
    """Append-only event log for backtest audit trail."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def emit(
        self,
        event_type: EventType,
        timestamp: pd.Timestamp,
        poi_id: str = "",
        **details: Any,
    ) -> None:
        """Record an event."""
        self._events.append(Event(
            type=event_type,
            timestamp=timestamp,
            poi_id=poi_id,
            details=details,
        ))

    def get_events(
        self,
        event_type: Optional[EventType] = None,
    ) -> list[Event]:
        """Return events, optionally filtered by type."""
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.type == event_type]

    def to_dataframe(self) -> pd.DataFrame:
        """Export all events as DataFrame."""
        if not self._events:
            return pd.DataFrame(columns=["type", "timestamp", "poi_id", "details"])
        return pd.DataFrame([
            {
                "type": e.type.value,
                "timestamp": e.timestamp,
                "poi_id": e.poi_id,
                "details": e.details,
            }
            for e in self._events
        ])

    def __len__(self) -> int:
        return len(self._events)
