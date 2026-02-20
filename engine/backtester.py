"""Main backtest orchestrator: bar-by-bar loop from data to results."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from context.mtf_manager import MTFManager
from context.bias import determine_bias_at
from context.sync_checker import check_sync
from context.state_machine import StateMachineManager, ConceptData
from strategy.types import Signal, SignalType, SyncMode, Bias
from strategy.entries import evaluate_entry
from strategy.exits import evaluate_exit, select_target
from strategy.addons import find_addon_candidates, evaluate_addon
from strategy.fta_handler import detect_fta, classify_fta_distance
from engine.portfolio import Portfolio
from engine.trade_log import TradeLog
from engine.metrics import compute_metrics, MetricsResult
from engine.events import EventLog, EventType


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""
    trade_log: pd.DataFrame
    equity_curve: np.ndarray
    metrics: MetricsResult
    signals: list[Signal]
    events: pd.DataFrame
    config: Config
    timestamps: pd.DatetimeIndex


class Backtester:
    """Main backtest orchestrator."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._manager: Optional[MTFManager] = None
        self._sm: Optional[StateMachineManager] = None
        self._portfolio: Optional[Portfolio] = None
        self._trade_log: Optional[TradeLog] = None
        self._event_log: Optional[EventLog] = None
        self._htf_bias: Bias = Bias.UNDEFINED
        self._ltf_bias: Bias = Bias.UNDEFINED
        self._sync_mode: SyncMode = SyncMode.UNDEFINED
        self._registered_poi_keys: set[str] = set()
        self._signals: list[Signal] = []

    def run(self, df_1m: pd.DataFrame) -> BacktestResult:
        """Execute the full backtest."""
        # 1. Filter date range
        df = self._filter_date_range(df_1m)
        n_bars = len(df)
        if n_bars == 0:
            raise ValueError("No data in the specified date range")

        # 2. Initialize MTFManager (pre-compute all concepts)
        self._manager = MTFManager(self._config)
        self._manager.initialize(df)

        # 3. Initialize components
        self._trade_log = TradeLog()
        self._event_log = EventLog()
        self._sm = StateMachineManager(self._config.strategy.confirmations)
        self._portfolio = Portfolio(
            backtest_config=self._config.backtest,
            risk_config=self._config.strategy.risk,
            n_bars=n_bars,
            trade_log=self._trade_log,
            event_log=self._event_log,
        )
        self._signals = []
        self._registered_poi_keys = set()

        # 4. Register initial POIs from first timestamp
        first_ts = df["time"].iloc[0]
        self._register_new_pois(first_ts)

        # 5. Compute initial bias
        self._update_bias_sync(first_ts)

        # 6. Main loop
        for bar_idx in range(n_bars):
            candle = df.iloc[bar_idx]
            ts = candle["time"]
            self._process_bar(candle, bar_idx, ts)

        # 7. Close remaining open positions at last price
        if n_bars > 0:
            last_candle = df.iloc[-1]
            last_ts = last_candle["time"]
            for poi_id in self._portfolio.get_open_poi_ids():
                self._portfolio.close_position(
                    poi_id=poi_id,
                    exit_signal_price=last_candle["close"],
                    exit_reason="END_OF_DATA",
                    timestamp=last_ts,
                    bar_index=n_bars - 1,
                )

        # 8. Compute metrics
        trade_df = self._trade_log.to_dataframe()
        equity_curve = self._portfolio.get_equity_curve()
        ts_index = pd.DatetimeIndex(df["time"].values)

        metrics = compute_metrics(
            trade_df=trade_df,
            equity_curve=equity_curve,
            initial_capital=self._config.backtest.initial_capital,
            timestamps=ts_index,
        )

        # 9. Return result
        return BacktestResult(
            trade_log=trade_df,
            equity_curve=equity_curve,
            metrics=metrics,
            signals=self._signals,
            events=self._event_log.to_dataframe(),
            config=self._config,
            timestamps=ts_index,
        )

    def _filter_date_range(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """Filter to configured date range."""
        df = df_1m.copy()
        start = pd.Timestamp(self._config.backtest.start_date, tz="UTC")
        end = pd.Timestamp(self._config.backtest.end_date, tz="UTC")

        if "time" in df.columns:
            mask = (df["time"] >= start) & (df["time"] <= end)
            df = df[mask].reset_index(drop=True)
        return df

    def _register_new_pois(self, timestamp: pd.Timestamp) -> None:
        """Register new POIs from all timeframes."""
        for tf in self._config.data.timeframes:
            pois = self._manager.get_pois_at(tf, timestamp)
            if len(pois) == 0:
                continue

            for _, poi in pois.iterrows():
                # Create fingerprint to avoid duplicates
                direction = poi.get("direction", 0)
                top = poi.get("top", 0.0)
                bottom = poi.get("bottom", 0.0)
                key = f"{tf}_{direction}_{top:.6f}_{bottom:.6f}"

                if key in self._registered_poi_keys:
                    continue
                self._registered_poi_keys.add(key)

                poi_dict = poi.to_dict()
                poi_dict["timeframe"] = tf
                poi_id = self._sm.register_poi(poi_dict, tf, timestamp)

                self._event_log.emit(
                    EventType.POI_REGISTERED, timestamp, poi_id,
                    timeframe=tf, direction=direction,
                )

    def _update_bias_sync(self, timestamp: pd.Timestamp) -> None:
        """Recompute HTF bias (1H) and LTF bias (5m), derive sync mode."""
        tfs = self._config.data.timeframes

        # HTF bias from 1H (or highest available)
        htf_tf = "1H" if "1H" in tfs else (tfs[-1] if tfs else "1m")
        td_htf = self._manager.get_timeframe_data(htf_tf)
        struct_htf = self._manager.get_structure_at(htf_tf, timestamp)
        self._htf_bias = determine_bias_at(td_htf.candles, struct_htf, timestamp)

        # LTF bias from 5m (or lowest non-1m)
        ltf_tf = "5m" if "5m" in tfs else (tfs[1] if len(tfs) > 1 else "1m")
        td_ltf = self._manager.get_timeframe_data(ltf_tf)
        struct_ltf = self._manager.get_structure_at(ltf_tf, timestamp)
        self._ltf_bias = determine_bias_at(td_ltf.candles, struct_ltf, timestamp)

        self._sync_mode = check_sync(self._htf_bias, self._ltf_bias)

        self._event_log.emit(
            EventType.BIAS_UPDATED, timestamp,
            htf_bias=self._htf_bias.value, ltf_bias=self._ltf_bias.value,
        )
        self._event_log.emit(
            EventType.SYNC_UPDATED, timestamp,
            sync_mode=self._sync_mode.value,
        )

    def _process_bar(
        self,
        candle: pd.Series,
        bar_index: int,
        timestamp: pd.Timestamp,
    ) -> None:
        """Process a single 1m bar through the full pipeline."""
        # a. Check HTF boundary closures
        for tf in self._config.data.timeframes:
            if tf == "1m":
                continue
            if self._manager.tf_just_closed(tf, timestamp):
                self._register_new_pois(timestamp)
                self._update_bias_sync(timestamp)
                break  # One update per bar is sufficient

        # b. Build concept data for 1m
        td_1m = self._manager.get_timeframe_data("1m")
        concept_data = ConceptData(
            nearby_fvgs=td_1m.fvgs,
            fvg_lifecycle=td_1m.fvg_lifecycle,
            nearby_liquidity=td_1m.liquidity,
            structure_events=td_1m.structure,
        )

        # c. State machine update
        sm_signals = self._sm.update(candle, bar_index, timestamp, concept_data)
        self._signals.extend(sm_signals)

        # d. Exits FIRST (to free position slots)
        self._handle_exits(candle, bar_index, timestamp)

        # e. Entries
        self._handle_entries(candle, bar_index, timestamp)

        # f. Add-ons
        self._handle_addons(candle, bar_index, timestamp)

        # g. Mark to market
        self._portfolio.update_mark_to_market(
            bar_index, candle["high"], candle["low"], candle["close"]
        )

    def _handle_entries(
        self,
        candle: pd.Series,
        bar_index: int,
        timestamp: pd.Timestamp,
    ) -> None:
        """For each READY POI state, evaluate entry."""
        for state in self._sm.get_ready_states():
            if self._portfolio.has_position_for_poi(state.poi_id):
                continue
            if not self._portfolio.can_open_position():
                self._event_log.emit(
                    EventType.POSITION_REJECTED, timestamp,
                    state.poi_id, reason="max_positions_reached"
                )
                continue

            # Get target estimate
            active_pois = self._manager.get_all_active_pois(timestamp)
            td_1m = self._manager.get_timeframe_data("1m")

            target_est = select_target(
                direction=state.poi_data["direction"],
                current_price=candle["close"],
                active_pois=active_pois,
                swing_points=td_1m.swing_points,
                sync_mode=self._sync_mode,
                config=self._config.strategy,
            )

            # FTA check
            fta = None
            fta_class = "none"
            if len(active_pois) > 0:
                fta = detect_fta(
                    candle["close"], target_est,
                    state.poi_data["direction"], active_pois
                )
                if fta is not None:
                    fta_class = classify_fta_distance(
                        fta, candle["close"], target_est
                    )

            entry_signal = evaluate_entry(
                poi_state=state,
                candle=candle,
                bar_index=bar_index,
                timestamp=timestamp,
                fta=fta,
                fta_classification=fta_class,
                sync_mode=self._sync_mode,
                nearby_fvgs=td_1m.fvgs,
                nearby_liquidity=td_1m.liquidity,
                config=self._config.strategy,
            )

            if entry_signal is not None:
                # Override target with proper selection
                entry_signal.target = target_est
                self._signals.append(entry_signal)

                trade_id = self._portfolio.open_position(
                    signal=entry_signal,
                    sync_mode=self._sync_mode,
                    bar_index=bar_index,
                    timeframe=state.poi_data.get("timeframe", ""),
                    confirmation_count=len(state.confirmations),
                )
                if trade_id is not None:
                    # Use actual fill price (post-slippage) from trade log
                    fill_price = self._trade_log.get_trade(trade_id).entry_price
                    self._sm.set_positioned(
                        state.poi_id,
                        fill_price,
                        entry_signal.stop_loss,
                        entry_signal.target,
                    )

    def _handle_exits(
        self,
        candle: pd.Series,
        bar_index: int,
        timestamp: pd.Timestamp,
    ) -> None:
        """For each POSITIONED/MANAGING state, evaluate exit."""
        td_1m = self._manager.get_timeframe_data("1m")

        for state in self._sm.get_positioned_states():
            # Compute FTA for this position's target
            fta = None
            if state.target is not None:
                active_pois = self._manager.get_all_active_pois(timestamp)
                if len(active_pois) > 0:
                    fta = detect_fta(
                        candle["close"], state.target,
                        state.poi_data["direction"], active_pois
                    )

            exit_signal = evaluate_exit(
                poi_state=state,
                candle=candle,
                bar_index=bar_index,
                timestamp=timestamp,
                fta=fta,
                structure_events=td_1m.structure,
                config=self._config.strategy,
            )

            if exit_signal is None:
                continue

            self._signals.append(exit_signal)

            if exit_signal.type == SignalType.EXIT:
                self._portfolio.close_position(
                    poi_id=state.poi_id,
                    exit_signal_price=exit_signal.price,
                    exit_reason=exit_signal.reason,
                    timestamp=timestamp,
                    bar_index=bar_index,
                )
                self._sm.close_poi(state.poi_id)

            elif exit_signal.type == SignalType.MOVE_TO_BE:
                state.breakeven_level = exit_signal.price
                state.stop_loss = exit_signal.price
                self._portfolio.modify_stop_loss(state.poi_id, exit_signal.price)
                self._sm.set_managing(state.poi_id)

    def _handle_addons(
        self,
        candle: pd.Series,
        bar_index: int,
        timestamp: pd.Timestamp,
    ) -> None:
        """Check for add-on entry opportunities."""
        td_1m = self._manager.get_timeframe_data("1m")

        for state in self._sm.get_positioned_states():
            if state.target is None:
                continue

            # Look for local TF POIs for add-ons
            ltf = "15m" if "15m" in self._config.data.timeframes else "5m"
            local_pois = self._manager.get_pois_at(ltf, timestamp)

            candidates = find_addon_candidates(
                state.poi_data["direction"],
                candle["close"],
                state.target,
                local_pois,
                timestamp,
            )

            for _, cand in candidates.head(1).iterrows():
                addon_signal = evaluate_addon(
                    main_state=state,
                    candidate_poi=cand,
                    candle=candle,
                    bar_index=bar_index,
                    timestamp=timestamp,
                    structure_events=td_1m.structure,
                    config=self._config.strategy,
                )
                if addon_signal is not None:
                    self._signals.append(addon_signal)
                    parent_positions = self._portfolio.get_positions_for_poi(state.poi_id)
                    parent_id = parent_positions[0].trade_id if parent_positions else None

                    self._portfolio.open_position(
                        signal=addon_signal,
                        sync_mode=self._sync_mode,
                        bar_index=bar_index,
                        is_addon=True,
                        parent_trade_id=parent_id,
                    )


def run_backtest(config: Config, df_1m: pd.DataFrame) -> BacktestResult:
    """Convenience entry point.

    Usage:
        from config import load_config
        from data.loader import load_instrument
        from engine.backtester import run_backtest

        config = load_config()
        df = load_instrument("NAS100")
        result = run_backtest(config, df)
    """
    return Backtester(config).run(df_1m)
