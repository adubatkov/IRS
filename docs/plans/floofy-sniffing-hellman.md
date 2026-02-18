# Phase 4: Backtest Engine

## Context

Phases 1-3 are complete (404 tests pass). Phase 3 produces `Signal` objects from the strategy logic layer. Phase 4 builds the **backtest engine** that consumes those signals: executing trades with commission/slippage, tracking positions and equity, and computing performance metrics.

The engine sits above Phase 3 and reuses all existing modules as-is. No modifications to Phases 1-3 are required.

## Architecture

```
Phase 3 (strategy/)                  Phase 4 (engine/)

evaluate_entry() ──> Signal ──┐     ┌── backtester.py (orchestrator)
evaluate_exit()  ──> Signal ──┤     │     _process_bar() loop
evaluate_addon() ──> Signal ──┘     │        │
                                    ├── portfolio.py (positions, equity, execution)
StateMachineManager ───────────────>│     open_position(), close_position()
MTFManager ────────────────────────>│     apply_slippage(), mark_to_market()
bias / sync_checker ───────────────>├── trade_log.py (trade journal)
                                    │     TradeRecord, MFE/MAE, R-multiple
                                    ├── metrics.py (performance analytics)
                                    │     Sharpe, drawdown, win rate, profit factor
                                    └── events.py (audit trail)
                                          EventLog for debugging

Output: BacktestResult(trade_log, equity_curve, metrics, signals, events)
```

**Entry point**: `result = run_backtest(config, df_1m)`

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Same-bar SL+TP | Check SL first | Conservative; avoids phantom wins |
| Exit before entry per bar | Exits processed first | Frees position slots for same-bar replacement |
| Equity storage | Pre-allocated numpy array | Performance: zero allocation in hot loop |
| Bias/sync updates | Only on TF boundary close | HTF bias can't change intra-candle |
| POI registration | Dynamic per HTF close | Prevents registering POIs before visible |
| Cash accounting | Two-step (commission at entry, proceeds at exit) | Matches real brokerage; equity always accurate |
| MFE/MAE | Price distance from entry | Simplest in hot loop; convert to R at close |

## Build Order

```
Tier 1 (no internal deps, parallel):
  Step 1a: engine/events.py        (~60 LOC)
  Step 1b: engine/trade_log.py     (~150 LOC)

Tier 2 (depends on Tier 1):
  Step 2:  engine/portfolio.py     (~200 LOC)

Tier 3 (no deps on portfolio, parallel with Tier 2):
  Step 3:  engine/metrics.py       (~200 LOC)

Tier 4 (depends on all above):
  Step 4:  engine/backtester.py    (~300 LOC)

Step 5:  Integration test + engine/__init__.py exports
```

---

## Step 1a: `engine/events.py` -- Event Log

Lightweight append-only audit trail for backtest debugging.

```python
class EventType(str, Enum):
    POI_REGISTERED, POI_TAPPED, ENTRY, EXIT, BE_MOVED,
    SL_MODIFIED, ADDON, BIAS_UPDATED, SYNC_UPDATED, POSITION_REJECTED

@dataclass
class Event:
    type: EventType
    timestamp: pd.Timestamp
    poi_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)

class EventLog:
    def emit(self, event_type, timestamp, poi_id="", **details) -> None
    def get_events(self, event_type=None) -> list[Event]
    def to_dataframe(self) -> pd.DataFrame
```

**Test**: `tests/unit/test_events.py` (~5 tests) -- emit/retrieve, filtering, to_dataframe, empty log

---

## Step 1b: `engine/trade_log.py` -- Trade Journal

Records complete trade lifecycle with MFE/MAE tracking.

```python
@dataclass
class TradeRecord:
    trade_id: int
    poi_id: str
    direction: int                    # +1/-1
    entry_time: pd.Timestamp
    entry_price: float                # After slippage
    entry_signal_price: float         # Before slippage
    position_size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_signal_price: Optional[float] = None
    exit_reason: str = ""
    realized_pnl: float = 0.0
    commission_entry: float = 0.0
    commission_exit: float = 0.0
    gross_pnl: float = 0.0
    max_favorable_excursion: float = 0.0  # MFE in price units
    max_adverse_excursion: float = 0.0    # MAE in price units
    sync_mode: str = ""
    timeframe: str = ""
    confirmation_count: int = 0
    stop_loss: float = 0.0
    target: float = 0.0
    is_addon: bool = False
    parent_trade_id: Optional[int] = None
    outcome: str = ""                 # WIN/LOSS/BREAKEVEN
    r_multiple: float = 0.0
    duration_bars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

def classify_outcome(realized_pnl, commission_total) -> str
def compute_r_multiple(entry_price, exit_price, stop_loss, direction) -> float

class TradeLog:
    def open_trade(...) -> int           # Returns trade_id
    def close_trade(trade_id, ...) -> TradeRecord
    def update_excursion(trade_id, candle_high, candle_low) -> None  # Hot path
    def get_open_trades() -> list[TradeRecord]
    def get_trade(trade_id) -> TradeRecord
    def to_dataframe() -> pd.DataFrame
    def to_csv(path) -> None
```

**Test**: `tests/unit/test_trade_log.py` (~10 tests) -- open/close win/loss/BE, MFE/MAE tracking, R-multiple, to_dataframe, to_csv

---

## Step 2: `engine/portfolio.py` -- Position & Equity Management

```python
@dataclass
class PositionInfo:
    trade_id: int
    poi_id: str
    direction: int
    entry_price: float
    position_size: float
    stop_loss: float
    target: float
    entry_bar_index: int
    is_addon: bool = False
    parent_trade_id: Optional[int] = None

class Portfolio:
    def __init__(self, backtest_config, risk_config, n_bars, trade_log, event_log=None)

    @property equity -> float           # cash + unrealized P&L
    @property cash -> float
    @property open_position_count -> int  # Distinct poi_ids

    def can_open_position() -> bool     # < max_concurrent_positions
    def open_position(signal, sync_mode, bar_index, ...) -> Optional[int]
    def close_position(poi_id, exit_signal_price, exit_reason, timestamp, bar_index, trade_id=None) -> list[TradeRecord]
    def modify_stop_loss(poi_id, new_sl) -> None
    def update_mark_to_market(bar_index, candle_high, candle_low, candle_close) -> None
    def get_equity_curve() -> np.ndarray
    def get_positions_for_poi(poi_id) -> list[PositionInfo]
    def has_position_for_poi(poi_id) -> bool

def apply_slippage(price, direction, is_entry, slippage_pct) -> float
    # Always against trader: LONG entry up, SHORT entry down, etc.
```

**Execution flow in `open_position()`**:
1. Check `max_concurrent_positions` (skip for add-ons to existing poi_id)
2. Apply slippage to get fill price
3. `calculate_position_size(equity, fill, sl, sync_mode, risk_config)` for size
4. Entry commission = fill * size * commission_pct, deduct from cash
5. Record in trade_log, track in _positions
6. Emit ENTRY event

**Execution flow in `close_position()`**:
1. Apply slippage to get fill price
2. Gross P&L = direction * (fill - entry) * size
3. Exit commission = fill * size * commission_pct
4. Cash += direction * (fill - entry) * size - exit_commission
5. Record in trade_log, remove from _positions
6. Emit EXIT event

**Test**: `tests/unit/test_portfolio.py` (~12 tests) -- open/close long/short, slippage, commission, max positions, add-on bypass, equity curve, unrealized P&L, zero-size rejection

---

## Step 3: `engine/metrics.py` -- Performance Analytics

Pure functions, no state. Operates on trade DataFrame and equity curve.

```python
@dataclass
class MetricsResult:
    total_return_pct, cagr_pct, max_drawdown_pct, max_drawdown_duration_bars
    sharpe_ratio, sortino_ratio, calmar_ratio
    total_trades, winning_trades, losing_trades, breakeven_trades
    win_rate_pct, avg_rr, avg_win_rr, avg_loss_rr
    profit_factor, expectancy
    avg_trade_duration_bars
    sync_stats: dict           # {SYNC: {trades, win_rate, avg_rr, pf}, DESYNC: {...}}
    monthly_returns: pd.DataFrame  # month, return_pct, trade_count
    final_equity, peak_equity

def compute_metrics(trade_df, equity_curve, initial_capital, bars_per_year) -> MetricsResult
def compute_drawdown(equity_curve) -> tuple[np.ndarray, float, int]
def compute_sharpe(equity_curve, bars_per_year) -> float
def compute_sortino(equity_curve, bars_per_year) -> float
def compute_calmar(cagr, max_drawdown_pct) -> float
def compute_trade_stats(trade_df) -> dict
def compute_sync_mode_stats(trade_df) -> dict
def compute_monthly_returns(trade_df, equity_curve, timestamps, initial_capital) -> pd.DataFrame
```

**Test**: `tests/unit/test_metrics.py` (~10 tests) -- return, drawdown, Sharpe/Sortino, win rate, profit factor, sync stats, no-trades edge case, all-winners edge case

---

## Step 4: `engine/backtester.py` -- Main Orchestrator

```python
@dataclass
class BacktestResult:
    trade_log: pd.DataFrame
    equity_curve: np.ndarray
    metrics: MetricsResult
    signals: list[Signal]
    events: pd.DataFrame
    config: Config
    timestamps: pd.DatetimeIndex

class Backtester:
    def __init__(self, config: Config)
    def run(self, df_1m: pd.DataFrame) -> BacktestResult

def run_backtest(config: Config, df_1m: pd.DataFrame) -> BacktestResult
```

**Main loop in `_process_bar(candle, bar_index, timestamp)`**:
```
a. For each HTF: if tf_just_closed -> register_new_pois, update_bias_sync
b. Build ConceptData from 1m TimeframeData
c. sm.update(candle, bar_index, timestamp, concept_data)
d. _handle_exits(candle, bar_index, timestamp)     # EXIT BEFORE ENTRY
e. _handle_entries(candle, bar_index, timestamp)
f. _handle_addons(candle, bar_index, timestamp)
g. portfolio.update_mark_to_market(bar_index, high, low, close)
```

**POI registration**: Dynamic -- checks each TF on boundary close, registers any POI with `creation_time <= timestamp` not yet tracked. Fingerprint = `{tf}_{direction}_{top:.6f}_{bottom:.6f}`.

**Bias/sync**: HTF = 1H structure bias, LTF = 5m structure bias. Updated only when 1H or 5m candle just closed.

**Test**: `tests/integration/test_backtest_e2e.py` (~5 tests) -- full run with synthetic data, deterministic results, no look-ahead, max positions respected, equity curve length

---

## Step 5: Integration & Exports

**`engine/__init__.py`**:
```python
from engine.backtester import run_backtest, Backtester, BacktestResult
from engine.portfolio import Portfolio
from engine.trade_log import TradeLog, TradeRecord
from engine.metrics import compute_metrics, MetricsResult
from engine.events import EventLog, EventType
```

---

## File Summary

| Step | Source File | Test File | Depends On |
|------|------------|-----------|------------|
| 1a | `engine/events.py` | `tests/unit/test_events.py` | -- |
| 1b | `engine/trade_log.py` | `tests/unit/test_trade_log.py` | -- |
| 2 | `engine/portfolio.py` | `tests/unit/test_portfolio.py` | trade_log, events, strategy.risk |
| 3 | `engine/metrics.py` | `tests/unit/test_metrics.py` | -- (pure functions) |
| 4 | `engine/backtester.py` | `tests/integration/test_backtest_e2e.py` | all engine + all Phase 3 |
| 5 | `engine/__init__.py` | -- | all engine |

**Total: 5 source files (1 edit + 4 new) + 5 test files = ~10 files, ~910 LOC source + ~400 LOC tests**

## Verification

After each step:
1. `pytest tests/unit/test_{module}.py -v` -- step tests pass
2. `ruff check engine/` -- clean

After all steps:
1. `pytest tests/ -v` -- all tests pass (404 existing + ~42 new)
2. Full e2e integration test passes
3. `run_backtest(config, make_trending_1m(600))` returns valid BacktestResult
