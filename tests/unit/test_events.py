"""Tests for engine.events module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from engine.events import EventLog, EventType


def test_emit_and_retrieve():
    log = EventLog()
    ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")

    log.emit(EventType.ENTRY, ts, poi_id="z1")
    log.emit(EventType.EXIT, ts, poi_id="z2")
    log.emit(EventType.ADDON, ts, poi_id="z1")

    events = log.get_events()
    assert len(events) == 3
    assert len(log) == 3
    assert events[0].type == EventType.ENTRY
    assert events[1].type == EventType.EXIT
    assert events[2].type == EventType.ADDON


def test_filter_by_type():
    log = EventLog()
    ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")

    log.emit(EventType.ENTRY, ts, poi_id="z1")
    log.emit(EventType.EXIT, ts, poi_id="z2")
    log.emit(EventType.ENTRY, ts, poi_id="z3")
    log.emit(EventType.BE_MOVED, ts, poi_id="z1")

    entries = log.get_events(EventType.ENTRY)
    assert len(entries) == 2
    assert all(e.type == EventType.ENTRY for e in entries)

    exits = log.get_events(EventType.EXIT)
    assert len(exits) == 1
    assert exits[0].poi_id == "z2"


def test_to_dataframe():
    log = EventLog()
    ts1 = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    ts2 = pd.Timestamp("2024-01-01 11:00", tz="UTC")

    log.emit(EventType.POI_REGISTERED, ts1, poi_id="z1", price=1.05)
    log.emit(EventType.POI_TAPPED, ts2, poi_id="z1", price=1.06)

    df = log.to_dataframe()
    assert list(df.columns) == ["type", "timestamp", "poi_id", "details"]
    assert len(df) == 2
    assert df.iloc[0]["type"] == "POI_REGISTERED"
    assert df.iloc[1]["timestamp"] == ts2


def test_empty_log():
    log = EventLog()

    assert len(log) == 0
    assert log.get_events() == []

    df = log.to_dataframe()
    assert list(df.columns) == ["type", "timestamp", "poi_id", "details"]
    assert len(df) == 0


def test_event_details():
    log = EventLog()
    ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")

    log.emit(
        EventType.SL_MODIFIED,
        ts,
        poi_id="z5",
        old_sl=1.0800,
        new_sl=1.0850,
        reason="trailing",
    )

    event = log.get_events()[0]
    assert event.details["old_sl"] == 1.0800
    assert event.details["new_sl"] == 1.0850
    assert event.details["reason"] == "trailing"
    assert len(event.details) == 3
