# Phase 3: Strategy Logic

## Context

Phase 1 (Data) and Phase 2 (Concepts) are complete. 136 tests pass. All concept modules are built and tested:
- `concepts/fractals.py`, `concepts/structure.py`, `concepts/fvg.py`
- `concepts/liquidity.py`, `concepts/zones.py`, `concepts/registry.py`

Phase 3 builds the **strategy logic layer** on top of concepts: multi-timeframe orchestration, bias determination, confirmation counting, entry/exit decisions, and position management. The output is `Signal` objects that Phase 4 (Backtest Engine) will consume.

**Housekeeping:** `config.py` still has `OrderBlocksConfig` from deleted OB module -- remove it in Step 0.

## Architecture

```
Phase 2 (concepts/)          Phase 3 (context/ + strategy/)          Phase 4 (engine/)

fractals ─┐                  ┌─ mtf_manager ──── bias ──── sync ─┐
structure ─┤  pre-computed   │                                     │
fvg ───────┤ ═══════════════>│  confirmations ─── state_machine ──>│ Signal objects
liquidity ─┤  per timeframe  │  fta_handler                        │ ═══════════>  backtester
zones ─────┤                 │  entries / exits / addons ──────────┘
registry ──┘                 │  risk
                             └─ types.py (shared enums/dataclasses)
```

**Key design decisions:**
1. **Batch pre-compute + time-gated access**: All concepts computed once across 7 TFs before backtest. MTF manager prevents look-ahead by filtering results by timestamp.
2. **Signals as boundary contract**: Phase 3 produces `Signal` dataclasses; Phase 4 consumes them. No portfolio/execution coupling.
3. **Pure functions except state machine**: `StateMachineManager` is the only stateful class. Everything else is pure and independently testable.
4. **Event list per POI**: Confirmations are `list[Confirmation]` (5-8 items), not DataFrames.
5. **Config injection everywhere**: No global state. Every function receives its config as a parameter.

## Build Order (3 Tiers)

```
Tier 1 (no Phase-3 internal deps):
  Step 0: config.py cleanup (remove OrderBlocksConfig)
  Step 1: strategy/types.py
  Step 2: context/mtf_manager.py
  Step 3: context/bias.py

Tier 2 (depends on Tier 1):
  Step 4: context/sync_checker.py
  Step 5: strategy/confirmations.py        <-- core strategy logic
  Step 6: strategy/fta_handler.py
  Step 7: strategy/risk.py

Tier 3 (depends on Tier 2):
  Step 8:  context/state_machine.py        <-- orchestrator
  Step 9:  strategy/entries.py
  Step 10: strategy/exits.py + strategy/addons.py
  Step 11: Integration test + notebook
```

---

## Step 0: Config Cleanup

**File:** `config.py`
- Remove `OrderBlocksConfig` dataclass
- Remove `orderblocks` field from `ConceptsConfig`
- Add `FTAConfig` to `StrategyConfig`:
  ```python
  @dataclass
  class FTAConfig:
      close_threshold_pct: float = 0.3
      invalidation_mode: str = "close"
  ```

---

## Step 1: `strategy/types.py` -- Shared Types

All Phase 3 modules import from here. No external Phase 3 dependencies.

**Enums:**
- `Bias` (BULLISH, BEARISH, UNDEFINED)
- `SyncMode` (SYNC, DESYNC, UNDEFINED)
- `POIPhase` (IDLE, POI_TAPPED, COLLECTING, READY, POSITIONED, MANAGING, CLOSED)
- `ConfirmationType` (POI_TAP, LIQUIDITY_SWEEP, FVG_INVERSION, INVERSION_TEST, STRUCTURE_BREAK, FVG_WICK_REACTION, CVB_TEST, ADDITIONAL_CBOS)
- `SignalType` (ENTER, EXIT, MODIFY_SL, ADD_ON, MOVE_TO_BE)
- `ExitReason` (TARGET_HIT, STOP_LOSS_HIT, BREAKEVEN_HIT, FTA_VALIDATED, POI_INVALIDATED, FLIP)

**Dataclasses:**
- `Confirmation(type, timestamp, bar_index, details: dict)`
- `Signal(type, poi_id, direction, timestamp, price, stop_loss, target, position_size_mult, reason, metadata)`
- `POIState(poi_id, poi_data, phase, confirmations, entry_price, stop_loss, target, breakeven_level, fta, addons, created_at, last_updated)`

**Test:** `tests/unit/test_types.py` -- enum values, dataclass defaults, serialization

---

## Step 2: `context/mtf_manager.py` -- Multi-Timeframe Manager

**Dependencies:** `data/resampler.py`, all `concepts/` modules, `config.py`

```python
class TimeframeData:
    """Pre-computed concept data for one TF."""
    candles, swings, swing_points, structure, cisd,
    fvgs, fvg_lifecycle, liquidity, session_levels, pois

class MTFManager:
    def initialize(self, df_1m: pd.DataFrame) -> None
        # Resample to all TFs, run full concept pipeline per TF

    def get_candle_at(self, tf, timestamp) -> pd.Series | None
        # Most recently CLOSED candle (no look-ahead)

    def get_pois_at(self, tf, timestamp) -> pd.DataFrame
    def get_structure_at(self, tf, timestamp) -> pd.DataFrame
    def get_fvgs_at(self, tf, timestamp) -> pd.DataFrame

    def get_all_active_pois(self, timestamp) -> pd.DataFrame
        # Aggregate across all TFs

    def tf_just_closed(self, tf, timestamp_1m) -> bool
        # Did a new candle just close on this TF?
```

**Test:** `tests/unit/test_mtf_manager.py`
- Initialize with synthetic 1m data, verify TF candle counts
- `get_candle_at()` returns correct candle, no look-ahead
- `tf_just_closed()` fires at correct boundaries
- `get_pois_at()` filters correctly by timestamp

**Integration:** `tests/integration/test_mtf_pipeline.py`
- Real NAS100 data (10K bars), verify concept counts per TF

---

## Step 3: `context/bias.py` -- HTF Bias

**Dependencies:** `strategy/types.py`

```python
def determine_bias(candles, structure_events, lookback=10) -> Bias
    # From recent structure: predominantly bullish cBOS -> BULLISH, etc.

def determine_bias_at(candles, structure_events, timestamp, lookback=10) -> Bias
    # Time-filtered version

def get_trend_from_structure(structure_events, n_recent=3) -> Bias
    # Direction of last N structure events
```

**Test:** `tests/unit/test_bias.py` -- uptrend=BULLISH, downtrend=BEARISH, mixed=UNDEFINED

---

## Step 4: `context/sync_checker.py` -- Sync/Desync

**Dependencies:** `strategy/types.py`, `context/bias.py`

```python
def check_sync(htf_bias, ltf_bias) -> SyncMode
    # Both same -> SYNC, different -> DESYNC

def get_position_size_multiplier(sync_mode, risk_config) -> float
    # SYNC -> 1.0, DESYNC -> 0.5

def get_target_mode(sync_mode) -> str
    # SYNC -> "distant", DESYNC -> "local"
```

**Test:** `tests/unit/test_sync_checker.py` -- all 9 Bias combinations, multipliers, target modes

---

## Step 5: `strategy/confirmations.py` -- Confirmation Counting

**Dependencies:** `strategy/types.py`, concept type signatures

This is the **core strategy module**. 8 individual checkers + master collector.

```python
# Individual checkers (each returns dict with details or None):
def check_poi_tap(candle_high, candle_low, poi_top, poi_bottom, poi_direction) -> bool
def check_liquidity_sweep(candle_h/l/c, nearby_liquidity, poi_direction) -> dict | None
def check_fvg_inversion(fvg_lifecycle, bar_index, poi_direction) -> dict | None
def check_inversion_test(candle_h/l, inverted_fvgs, poi_direction) -> dict | None
def check_structure_break(structure_events, bar_index, poi_direction) -> dict | None
def check_fvg_wick_reaction(candle, nearby_fvgs, poi_direction) -> dict | None
def check_cvb_test(candle_h/l/c, nearby_fvgs, poi_direction) -> dict | None
def check_additional_cbos(structure_events, bar_index, poi_direction, existing) -> dict | None

# Master function:
def collect_confirmations(candle, bar_index, timestamp, poi_data,
    existing_confirms, nearby_fvgs, fvg_lifecycle, nearby_liquidity,
    structure_events, config) -> list[Confirmation]

# Helpers:
def confirmation_count(confirms) -> int
def is_ready(confirms, config) -> bool
def has_fifth_confirm_trap(confirms) -> bool
    # True if exactly 5 confirms but no FVG test/inversion test
```

**Rules encoded:**
- RULE 2: Dedup by type per occurrence
- RULE 3: FVG wick reaction only valid after 5+ confirms exist
- RULE 4: Per-POI counting (caller responsibility)
- 5th-confirm trap: 5 confirms but no FVG_INVERSION/INVERSION_TEST/FVG_WICK_REACTION -> wait for RTO

**Test:** `tests/unit/test_confirmations.py` -- most important test file
- Each checker individually
- Incremental collection across bars
- Dedup, max cap, 5th-confirm trap detection
- FVG wick blocked under 5 confirms

---

## Step 6: `strategy/fta_handler.py` -- First Trouble Area

**Dependencies:** `strategy/types.py`

```python
def detect_fta(current_price, target, direction, active_pois) -> dict | None
    # For LONG: first bearish POI between price and target
    # For SHORT: first bullish POI between price and target

def classify_fta_distance(fta, current_price, target, threshold=0.3) -> str
    # "far" | "close" | "none"

def check_fta_invalidation(fta, candle_close) -> bool
    # Price closed through FTA

def check_fta_validation(fta, candle_h/l/c, direction) -> bool
    # Price rejected at FTA (bounced back)

def should_enter_with_fta(fta, classification) -> tuple[bool, str]
    # Decision matrix: far=enter, close=wait, none=enter
```

**Test:** `tests/unit/test_fta_handler.py` -- detection, distance, invalidation/validation, decision matrix

---

## Step 7: `strategy/risk.py` -- Position Sizing & SL

**Dependencies:** `strategy/types.py`, `context/sync_checker.py`

```python
def calculate_stop_loss(poi_data, direction, nearby_fvgs, nearby_liquidity,
    method="behind_liquidity") -> float
    # Behind FVG/CVB/liquidity/POI zone

def calculate_position_size(equity, entry, sl, sync_mode, risk_config) -> float
    # risk_amount = equity * max_risk * sync_mult / distance

def validate_risk(entry, sl, target, direction, min_rr=2.0) -> tuple[bool, float]
    # Check minimum RR

def calculate_breakeven_level(entry, direction, commission_pct=0.0006) -> float
```

**Test:** `tests/unit/test_risk.py` -- SL placement, position sizing, RR validation, BE calculation

---

## Step 8: `context/state_machine.py` -- POI State Orchestrator

**Dependencies:** `strategy/types.py`, `strategy/confirmations.py`

```python
def make_poi_id(timeframe, direction, creation_index) -> str

def transition(state, candle, bar_index, timestamp, concept_data, config)
    -> tuple[POIState, list[Signal]]
    # Core transition: IDLE -> TAPPED -> COLLECTING -> READY -> POSITIONED -> MANAGING -> CLOSED

class StateMachineManager:
    def register_poi(poi_data, timeframe, timestamp) -> str
    def update(candle, bar_index, timestamp, concept_data) -> list[Signal]
    def get_active_states() -> list[POIState]
    def get_positioned_states() -> list[POIState]
    def invalidate_poi(poi_id, reason) -> None
```

**Test:** `tests/unit/test_state_machine.py`
- Full lifecycle (IDLE through CLOSED)
- Multiple simultaneous POIs
- Invalidation forces CLOSED
- Signals emitted at correct transitions

---

## Step 9: `strategy/entries.py` -- Entry Decisions

**Dependencies:** `strategy/types.py`, `confirmations.py`, `fta_handler.py`, `risk.py`

```python
def evaluate_entry(poi_state, candle, bar_index, timestamp, fta, fta_class,
    sync_mode, config) -> Signal | None
    # Decision tree:
    # 1. Phase == READY?
    # 2. FTA close? -> wait
    # 3. 5th-confirm trap? -> wait for RTO
    # 4. Conservative: structural exit + SL behind liquidity
    # 5. Aggressive: immediate BU plan or wait RTO

def check_conservative_entry(poi_state, candle, config) -> bool
def check_aggressive_entry(poi_state, candle, config) -> bool
def check_rto_entry(poi_state, candle, nearby_fvgs) -> bool
def build_entry_signal(...) -> Signal
```

**Test:** `tests/unit/test_entries.py` -- conservative/aggressive/RTO, FTA blocking, trap prevention

---

## Step 10: `strategy/exits.py` + `strategy/addons.py`

### exits.py

```python
def check_target_hit(candle_h/l, target, direction) -> bool
def check_stop_loss_hit(candle_h/l, sl, direction) -> bool
def check_structural_breakeven(poi_state, structure, bar_index, config) -> float | None
def check_fta_breakeven(poi_state, fta, current_price, config) -> float | None
def select_target(direction, price, pois, swings, sync_mode, config) -> float
def evaluate_exit(poi_state, candle, bar_index, timestamp, fta, structure, config) -> Signal | None
```

### addons.py

```python
def find_addon_candidates(direction, price, target, local_pois, timestamp) -> pd.DataFrame
def evaluate_addon(main_state, candidate, candle, bar_index, timestamp, structure, config) -> Signal | None
def should_addon_bu(addon_state, structure, bar_index) -> bool
```

**Tests:** `tests/unit/test_exits.py`, `tests/unit/test_addons.py`

---

## Step 11: Integration Test + Notebook

### `tests/integration/test_strategy_pipeline.py`
Full pipeline test with synthetic data containing a known trade pattern:
1. Initialize MTF manager
2. Step bar-by-bar through state machine
3. Verify: bias correct, sync correct, confirmations counted, entry signal at right bar, exit at target

### `notebooks/07_strategy_viewer.ipynb`
Visualize on real NAS100 data:
- POI detection across timeframes
- Confirmation events marked on chart
- Entry/exit signals overlaid
- State machine phase timeline

---

## File Summary

| Step | Source File | Test File | Depends On |
|------|------------|-----------|------------|
| 0 | `config.py` (edit) | existing tests | -- |
| 1 | `strategy/types.py` | `tests/unit/test_types.py` | -- |
| 2 | `context/mtf_manager.py` | `tests/unit/test_mtf_manager.py` | data/*, concepts/* |
| 3 | `context/bias.py` | `tests/unit/test_bias.py` | types |
| 4 | `context/sync_checker.py` | `tests/unit/test_sync_checker.py` | types, bias |
| 5 | `strategy/confirmations.py` | `tests/unit/test_confirmations.py` | types |
| 6 | `strategy/fta_handler.py` | `tests/unit/test_fta_handler.py` | types |
| 7 | `strategy/risk.py` | `tests/unit/test_risk.py` | types, sync_checker |
| 8 | `context/state_machine.py` | `tests/unit/test_state_machine.py` | types, confirmations |
| 9 | `strategy/entries.py` | `tests/unit/test_entries.py` | types, confirms, fta, risk |
| 10a | `strategy/exits.py` | `tests/unit/test_exits.py` | types, fta, risk |
| 10b | `strategy/addons.py` | `tests/unit/test_addons.py` | types, entries, risk |
| 11 | `notebooks/07_strategy_viewer.ipynb` | `tests/integration/test_strategy_pipeline.py` | all |

**Total: 12 source files (1 edit + 11 new) + 12 test files = ~24 files**

## Verification

After each step:
1. `pytest tests/unit/test_{module}.py -v` -- step tests pass
2. `ruff check {files}` -- clean
3. `mypy {files} --ignore-missing-imports` -- clean

After all steps:
1. `pytest tests/ -v` -- all tests pass (136 existing + ~80 new)
2. Full integration pipeline test passes
3. Notebook runs without errors on real NAS100 data
