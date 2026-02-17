"""Multi-timeframe data manager for batch pre-computation and time-gated access.

Pre-computes all concept detection across all configured timeframes, then provides
time-gated query methods to prevent look-ahead bias during backtesting.
"""

from dataclasses import dataclass

import pandas as pd

from config import Config
from data.resampler import resample
from concepts.fractals import detect_swings, get_swing_points
from concepts.structure import detect_structure, detect_cisd
from concepts.fvg import detect_fvg, track_fvg_lifecycle
from concepts.liquidity import detect_equal_levels, detect_session_levels
from concepts.registry import build_poi_registry


@dataclass
class TimeframeData:
    """All pre-computed concept data for one timeframe."""

    candles: pd.DataFrame
    swings: pd.DataFrame
    swing_points: pd.DataFrame
    structure: pd.DataFrame
    cisd: pd.DataFrame
    fvgs: pd.DataFrame
    fvg_lifecycle: list[dict]
    liquidity: pd.DataFrame
    session_levels: pd.DataFrame
    pois: pd.DataFrame


class MTFManager:
    """Pre-computes and provides time-gated access to multi-TF concept data."""

    def __init__(self, config: Config):
        self.config = config
        self._data: dict[str, TimeframeData] = {}
        # Set of TF candle open times for tf_just_closed checks
        self._tf_boundary_times: dict[str, set[pd.Timestamp]] = {}

    def initialize(self, df_1m: pd.DataFrame) -> None:
        """Resample to all TFs and pre-compute all concepts.

        Uses config to determine which timeframes to process
        and what parameters to use for each concept detector.
        For 1m, skip resampling (use as-is).
        """
        timeframes = self.config.data.timeframes

        for tf in timeframes:
            if tf == "1m":
                candles = df_1m.copy()
            else:
                candles = resample(df_1m, tf)

            tf_data = self._compute_tf(tf, candles)
            self._data[tf] = tf_data

            # Build boundary time set for tf_just_closed
            if tf == "1m":
                # For 1m, every bar is a boundary (always returns True)
                self._tf_boundary_times[tf] = set(candles["time"])
            else:
                # Store TF candle open times as boundaries
                self._tf_boundary_times[tf] = set(candles["time"])

    def _compute_tf(self, tf: str, candles: pd.DataFrame) -> TimeframeData:
        """Run full concept pipeline for one timeframe."""
        swing_length = self.config.concepts.fractals.swing_length.get(tf, 5)
        close_break = self.config.concepts.structure.break_mode == "close"

        # Fractals
        swings = detect_swings(candles, swing_length=swing_length)
        swing_points = get_swing_points(candles, swings)

        # Structure
        structure = detect_structure(
            candles, swing_length=swing_length, close_break=close_break
        )
        cisd = detect_cisd(candles)

        # FVG
        fvgs = detect_fvg(
            candles,
            min_gap_pct=self.config.concepts.fvg.min_gap_pct,
            join_consecutive=self.config.concepts.fvg.join_consecutive,
        )
        fvg_lifecycle = track_fvg_lifecycle(
            candles,
            fvgs,
            mitigation_mode=self.config.concepts.fvg.mitigation_mode,
        )

        # Liquidity
        liquidity = detect_equal_levels(
            candles,
            swing_length=swing_length,
            range_percent=self.config.concepts.liquidity.range_percent,
            min_touches=self.config.concepts.liquidity.min_touches,
        )

        # Session levels (only if time column present)
        if "time" in candles.columns:
            session_levels = detect_session_levels(candles)
        else:
            session_levels = pd.DataFrame(columns=["period_start", "high", "low"])

        # POI registry
        pois = build_poi_registry(
            fvgs,
            liquidity,
            session_levels,
            fvg_lifecycle=fvg_lifecycle,
            timeframe=tf,
        )

        # Add creation_time to POIs for time-gating
        if len(pois) > 0 and "time" in candles.columns:
            pois = self._add_poi_creation_times(pois, fvgs, candles)

        return TimeframeData(
            candles=candles,
            swings=swings,
            swing_points=swing_points,
            structure=structure,
            cisd=cisd,
            fvgs=fvgs,
            fvg_lifecycle=fvg_lifecycle,
            liquidity=liquidity,
            session_levels=session_levels,
            pois=pois,
        )

    def _add_poi_creation_times(
        self,
        pois: pd.DataFrame,
        fvgs: pd.DataFrame,
        candles: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add a creation_time column to POIs based on FVG creation indices."""
        creation_times = []
        earliest_time = candles["time"].iloc[0]

        for _, poi in pois.iterrows():
            components = poi.get("components", [])
            max_time = earliest_time

            for comp in components:
                comp_type = comp.get("type", "")
                source_idx = comp.get("source_idx")

                if comp_type in ("fvg_htf", "fvg_ltf", "ifvg") and source_idx is not None:
                    # Look up the FVG creation_index
                    if source_idx < len(fvgs):
                        creation_idx = fvgs.iloc[source_idx]["creation_index"]
                        if creation_idx in candles.index:
                            fvg_time = candles.loc[creation_idx, "time"]
                            if fvg_time > max_time:
                                max_time = fvg_time

            creation_times.append(max_time)

        result = pois.copy()
        result["creation_time"] = creation_times
        return result

    def get_timeframe_data(self, tf: str) -> TimeframeData:
        """Get full pre-computed data for a timeframe."""
        if tf not in self._data:
            raise KeyError(f"Timeframe '{tf}' not found. Available: {list(self._data.keys())}")
        return self._data[tf]

    def get_candle_at(self, tf: str, timestamp: pd.Timestamp) -> pd.Series | None:
        """Get the most recently CLOSED candle for tf at given timestamp.

        Time-gated: only returns candles with time <= timestamp.
        """
        if tf not in self._data:
            raise KeyError(f"Timeframe '{tf}' not found. Available: {list(self._data.keys())}")

        candles = self._data[tf].candles
        if "time" not in candles.columns:
            return None

        mask = candles["time"] <= timestamp
        if not mask.any():
            return None

        return candles.loc[mask].iloc[-1]

    def get_pois_at(self, tf: str, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Get POIs that were created before timestamp.

        Filter POI registry: only include POIs whose creation_time
        is <= timestamp.
        """
        if tf not in self._data:
            raise KeyError(f"Timeframe '{tf}' not found. Available: {list(self._data.keys())}")

        pois = self._data[tf].pois
        if len(pois) == 0:
            return pois

        if "creation_time" not in pois.columns:
            return pois

        mask = pois["creation_time"] <= timestamp
        return pois.loc[mask].reset_index(drop=True)

    def get_structure_at(self, tf: str, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Get structure events confirmed before timestamp."""
        if tf not in self._data:
            raise KeyError(f"Timeframe '{tf}' not found. Available: {list(self._data.keys())}")

        structure = self._data[tf].structure
        if len(structure) == 0:
            return structure

        candles = self._data[tf].candles
        if "time" not in candles.columns:
            return structure

        # Filter by broken_index mapped to time
        valid_indices = []
        for idx, row in structure.iterrows():
            broken_idx = row["broken_index"]
            if broken_idx in candles.index:
                event_time = candles.loc[broken_idx, "time"]
                if event_time <= timestamp:
                    valid_indices.append(idx)

        if not valid_indices:
            return structure.iloc[:0]

        return structure.loc[valid_indices].reset_index(drop=True)

    def get_fvgs_at(self, tf: str, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Get FVGs created before timestamp."""
        if tf not in self._data:
            raise KeyError(f"Timeframe '{tf}' not found. Available: {list(self._data.keys())}")

        fvgs = self._data[tf].fvgs
        if len(fvgs) == 0:
            return fvgs

        candles = self._data[tf].candles
        if "time" not in candles.columns:
            return fvgs

        # Filter by creation_index mapped to time
        valid_indices = []
        for idx, row in fvgs.iterrows():
            creation_idx = row["creation_index"]
            if creation_idx in candles.index:
                creation_time = candles.loc[creation_idx, "time"]
                if creation_time <= timestamp:
                    valid_indices.append(idx)

        if not valid_indices:
            return fvgs.iloc[:0]

        return fvgs.loc[valid_indices].reset_index(drop=True)

    def get_all_active_pois(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Aggregate active POIs across all timeframes at timestamp.

        Adds a 'timeframe' column to distinguish origin.
        """
        frames = []

        for tf in self._data:
            pois = self.get_pois_at(tf, timestamp)
            if len(pois) > 0:
                pois_copy = pois.copy()
                pois_copy["timeframe"] = tf
                frames.append(pois_copy)

        if not frames:
            return pd.DataFrame(
                columns=["direction", "top", "bottom", "midpoint", "score",
                         "components", "component_count", "status", "timeframe"]
            )

        result = pd.concat(frames, ignore_index=True)
        return result.sort_values("score", ascending=False).reset_index(drop=True)

    def tf_just_closed(self, tf: str, timestamp_1m: pd.Timestamp) -> bool:
        """Check if a new candle just closed on the given timeframe.

        A candle 'just closed' if the 1m timestamp is the last bar
        that falls within that TF candle.

        For 1m, always return True.
        For others, check if the NEXT 1m bar would start a new TF candle.
        """
        if tf == "1m":
            return True

        if tf not in self._tf_boundary_times:
            raise KeyError(f"Timeframe '{tf}' not found. Available: {list(self._data.keys())}")

        # A TF candle just closed if the next minute is a TF boundary
        next_minute = timestamp_1m + pd.Timedelta(minutes=1)
        return next_minute in self._tf_boundary_times[tf]
