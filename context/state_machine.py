"""POI state machine for tracking POI lifecycle through phases."""

import pandas as pd
from typing import Any
from dataclasses import dataclass

from config import ConfirmationsConfig
from strategy.types import POIPhase, POIState, Signal
from strategy.confirmations import (
    check_poi_tap,
    collect_confirmations,
    is_ready,
)


def make_poi_id(timeframe: str, direction: int, creation_index: int) -> str:
    """Create a unique POI identifier.

    Format: "{timeframe}_{direction}_{creation_index}"
    Example: "4H_1_100", "15m_-1_250"
    """
    return f"{timeframe}_{direction}_{creation_index}"


@dataclass
class ConceptData:
    """Concept data needed for state machine updates.

    A subset of per-TF concept data relevant for confirmation counting.
    """

    nearby_fvgs: pd.DataFrame
    fvg_lifecycle: list[dict]
    nearby_liquidity: pd.DataFrame
    structure_events: pd.DataFrame


def transition(
    state: POIState,
    candle: pd.Series,
    bar_index: int,
    timestamp: pd.Timestamp,
    concept_data: ConceptData,
    config: ConfirmationsConfig,
) -> tuple[POIState, list[Signal]]:
    """Core state transition function.

    Phases:
    - IDLE: Check if candle taps POI -> POI_TAPPED
    - POI_TAPPED: Confirmation collection begins -> COLLECTING
    - COLLECTING: Accumulate confirmations. When is_ready() -> READY
    - READY: Strategy layer decides entry (external). Entry sets -> POSITIONED
    - POSITIONED: External management. SL/BE moves set -> MANAGING
    - MANAGING: External management continues. Exit closes -> CLOSED
    - CLOSED: Terminal state, no transitions

    Args:
        state: Current POI state.
        candle: Current candle data (open, high, low, close).
        bar_index: Current bar index.
        timestamp: Current bar timestamp.
        concept_data: Nearby concept data for confirmation checking.
        config: Confirmations config.

    Returns:
        (updated_state, signals) - updated state and any signals emitted.
    """
    signals: list[Signal] = []

    state.last_updated = timestamp

    if state.phase == POIPhase.CLOSED:
        return state, signals

    if state.phase == POIPhase.IDLE:
        # Check if candle taps the POI
        poi_top = state.poi_data["top"]
        poi_bottom = state.poi_data["bottom"]
        direction = state.poi_data["direction"]

        if check_poi_tap(candle["high"], candle["low"], poi_top, poi_bottom, direction):
            state.phase = POIPhase.POI_TAPPED
            # Fall through to start collecting on same bar

    if state.phase == POIPhase.POI_TAPPED:
        # Start collecting confirmations
        state.phase = POIPhase.COLLECTING
        # Collect on first tap bar
        state.confirmations = collect_confirmations(
            candle=candle,
            bar_index=bar_index,
            timestamp=timestamp,
            poi_data=state.poi_data,
            existing_confirms=state.confirmations,
            nearby_fvgs=concept_data.nearby_fvgs,
            fvg_lifecycle=concept_data.fvg_lifecycle,
            nearby_liquidity=concept_data.nearby_liquidity,
            structure_events=concept_data.structure_events,
            config=config,
        )

        if is_ready(state.confirmations, config):
            state.phase = POIPhase.READY

        return state, signals

    if state.phase == POIPhase.COLLECTING:
        # Continue collecting
        state.confirmations = collect_confirmations(
            candle=candle,
            bar_index=bar_index,
            timestamp=timestamp,
            poi_data=state.poi_data,
            existing_confirms=state.confirmations,
            nearby_fvgs=concept_data.nearby_fvgs,
            fvg_lifecycle=concept_data.fvg_lifecycle,
            nearby_liquidity=concept_data.nearby_liquidity,
            structure_events=concept_data.structure_events,
            config=config,
        )

        if is_ready(state.confirmations, config):
            state.phase = POIPhase.READY

        return state, signals

    # READY, POSITIONED, MANAGING phases are handled externally
    # by entries.py and exits.py -- the state machine just tracks state

    return state, signals


class StateMachineManager:
    """Manages multiple POI state machines concurrently.

    This is the only stateful class in Phase 3.
    """

    def __init__(self, config: ConfirmationsConfig):
        self.config = config
        self._states: dict[str, POIState] = {}
        self._next_index: int = 0

    def register_poi(
        self,
        poi_data: dict[str, Any],
        timeframe: str,
        timestamp: pd.Timestamp,
    ) -> str:
        """Register a new POI for tracking.

        Args:
            poi_data: POI data dict with direction, top, bottom, midpoint, etc.
            timeframe: Source timeframe.
            timestamp: Registration timestamp.

        Returns:
            The generated poi_id.
        """
        direction = poi_data["direction"]
        poi_id = make_poi_id(timeframe, direction, self._next_index)
        self._next_index += 1

        state = POIState(
            poi_id=poi_id,
            poi_data=poi_data,
            phase=POIPhase.IDLE,
            created_at=timestamp,
            last_updated=timestamp,
        )
        self._states[poi_id] = state
        return poi_id

    def update(
        self,
        candle: pd.Series,
        bar_index: int,
        timestamp: pd.Timestamp,
        concept_data: ConceptData,
    ) -> list[Signal]:
        """Update all active state machines with current bar.

        Only processes POIs in IDLE, POI_TAPPED, or COLLECTING phase.
        READY, POSITIONED, MANAGING are handled externally.

        Args:
            candle: Current candle data.
            bar_index: Current bar index.
            timestamp: Current timestamp.
            concept_data: Concept data for confirmation checking.

        Returns:
            List of signals emitted during this update.
        """
        all_signals: list[Signal] = []

        for poi_id, state in self._states.items():
            if state.phase in (POIPhase.IDLE, POIPhase.POI_TAPPED, POIPhase.COLLECTING):
                updated_state, signals = transition(
                    state=state,
                    candle=candle,
                    bar_index=bar_index,
                    timestamp=timestamp,
                    concept_data=concept_data,
                    config=self.config,
                )
                self._states[poi_id] = updated_state
                all_signals.extend(signals)

        return all_signals

    def get_state(self, poi_id: str) -> POIState:
        """Get state for a specific POI."""
        if poi_id not in self._states:
            raise KeyError(f"POI '{poi_id}' not found")
        return self._states[poi_id]

    def get_active_states(self) -> list[POIState]:
        """Get all POI states that are not CLOSED."""
        return [s for s in self._states.values() if s.phase != POIPhase.CLOSED]

    def get_positioned_states(self) -> list[POIState]:
        """Get POI states in POSITIONED or MANAGING phase."""
        return [
            s for s in self._states.values()
            if s.phase in (POIPhase.POSITIONED, POIPhase.MANAGING)
        ]

    def get_ready_states(self) -> list[POIState]:
        """Get POI states in READY phase."""
        return [s for s in self._states.values() if s.phase == POIPhase.READY]

    def set_positioned(
        self,
        poi_id: str,
        entry_price: float,
        stop_loss: float,
        target: float,
    ) -> None:
        """Mark a POI as positioned (entry taken).

        Called by entries.py after a trade is entered.
        """
        state = self.get_state(poi_id)
        state.phase = POIPhase.POSITIONED
        state.entry_price = entry_price
        state.stop_loss = stop_loss
        state.target = target

    def set_managing(self, poi_id: str) -> None:
        """Transition from POSITIONED to MANAGING (BE moved)."""
        state = self.get_state(poi_id)
        state.phase = POIPhase.MANAGING

    def invalidate_poi(self, poi_id: str, reason: str = "") -> None:
        """Force a POI to CLOSED state.

        Used when a POI is invalidated (price breaks through, or
        opposing POI activated, etc).
        """
        state = self.get_state(poi_id)
        state.phase = POIPhase.CLOSED

    def close_poi(self, poi_id: str) -> None:
        """Close a POI (trade exited or POI expired)."""
        state = self.get_state(poi_id)
        state.phase = POIPhase.CLOSED
