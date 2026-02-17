# IRS Backtester -- Master Project Plan

> Purpose: Complete implementation plan for the IRS strategy backtesting system.
> Covers: file structure, phased implementation, testing strategy, optimization notes,
> visualization tools, and preparation for live trading bot.
> This document serves as a prompt/reference for both the developer and Claude Code.

---

## 1. Project Structure

```
irs-backtest/
│
├── docs/                              # Documentation (these files)
│   ├── 01_STRATEGY.md                 # Strategy specification
│   ├── 02_SMC_CONCEPTS.md             # SMC concepts reference
│   ├── 03_PROJECT_PLAN.md             # This file
│   ├── phases/                        # Detailed phase plans
│   │   ├── PHASE_1_FOUNDATIONS.md
│   │   ├── PHASE_2_CONCEPTS.md
│   │   ├── PHASE_3_STRATEGY.md
│   │   ├── PHASE_4_BACKTEST.md
│   │   ├── PHASE_5_ANALYSIS.md
│   │   └── PHASE_6_LIVE_PREP.md
│   └── CHANGELOG.md                   # Track what's been implemented and tested
│
├── data/
│   ├── raw/                           # Raw 1m candle CSV files (user-provided)
│   │   └── README.md                  # Expected CSV format documentation
│   ├── processed/                     # Resampled and cached data (parquet)
│   └── loader.py                      # Data loading, caching, and resampling
│
├── concepts/                          # SMC concept detection (pure, stateless functions)
│   ├── __init__.py
│   ├── fractals.py                    # Swing highs/lows detection
│   ├── structure.py                   # BOS, cBOS, CISD detection
│   ├── fvg.py                         # FVG detection, mitigation, inversion
│   ├── liquidity.py                   # Equal highs/lows, sweep detection
│   ├── zones.py                       # Premium/Discount, CE/CVB calculation
│   └── registry.py                    # POI registry -- aggregates all concepts into POIs
│
├── context/                           # Market context and multi-TF analysis
│   ├── __init__.py
│   ├── mtf_manager.py                 # Multi-timeframe data manager
│   ├── bias.py                        # HTF bias determination (Daily/4H)
│   ├── sync_checker.py                # HTF/LTF synchronization check
│   └── state_machine.py               # Market state machine (per POI)
│
├── strategy/                          # Strategy logic
│   ├── __init__.py
│   ├── confirmations.py               # Confirmation counting and validation
│   ├── entries.py                     # Entry logic (conservative, aggressive, RTO-based)
│   ├── exits.py                       # Exit logic (targets, BU, flips)
│   ├── fta_handler.py                 # FTA detection, validation, invalidation
│   ├── addons.py                      # Add-on (добор) logic
│   └── risk.py                        # Position sizing, stop-loss placement
│
├── engine/                            # Backtest engine
│   ├── __init__.py
│   ├── backtester.py                  # Main backtest loop (bar-by-bar)
│   ├── portfolio.py                   # Position management (open, close, modify)
│   ├── trade_log.py                   # Trade journal with full metadata
│   └── events.py                      # Event system (signals, fills, state changes)
│
├── analysis/                          # Post-backtest analysis
│   ├── __init__.py
│   ├── metrics.py                     # Performance metrics (winrate, RR, drawdown, PnL)
│   ├── report.py                      # Generate summary reports
│   └── equity_curve.py                # Equity curve and drawdown analysis
│
├── visualization/                     # Chart visualization tools
│   ├── __init__.py
│   ├── chart.py                       # Core charting (candlestick + overlays)
│   ├── overlays.py                    # SMC overlays (FVG boxes, swing markers, structure lines)
│   ├── trade_markers.py               # Entry/exit markers, BU levels, targets
│   └── interactive.py                 # Interactive exploration tools
│
├── notebooks/                         # Jupyter notebooks for exploration and validation
│   ├── 01_data_exploration.ipynb      # Load data, view candles, check quality
│   ├── 02_fractals_viewer.ipynb       # Visualize swing highs/lows detection
│   ├── 03_structure_viewer.ipynb      # Visualize BOS/cBOS/CISD
│   ├── 04_fvg_viewer.ipynb            # Visualize FVG detection and lifecycle
│   ├── 05_poi_viewer.ipynb            # Visualize composite POIs
│   ├── 06_strategy_viewer.ipynb       # Visualize confirmation counting and entries
│   ├── 07_backtest_results.ipynb      # Interactive backtest results analysis
│   └── utils.py                       # Shared notebook utilities
│
├── tests/                             # Test suite
│   ├── unit/                          # Unit tests (synthetic data)
│   │   ├── test_fractals.py
│   │   ├── test_structure.py
│   │   ├── test_fvg.py
│   │   ├── test_liquidity.py
│   │   ├── test_confirmations.py
│   │   └── test_portfolio.py
│   ├── integration/                   # Integration tests (real data scenarios)
│   │   ├── test_mtf_pipeline.py       # Multi-TF data flow
│   │   ├── test_concept_chain.py      # Swings -> Structure -> FVG chain
│   │   └── test_full_trade.py         # Complete trade lifecycle
│   ├── validation/                    # Validation against known setups
│   │   ├── test_known_setups.py       # Manually annotated setups from real data
│   │   ├── setups/                    # JSON files with expected outcomes
│   │   │   ├── setup_001_btc_long.json
│   │   │   └── ...
│   │   └── compare_smc_library.py     # Compare our detection vs smartmoneyconcepts lib
│   └── conftest.py                    # Shared fixtures and synthetic data generators
│
├── config.py                          # Global configuration and parameters
├── requirements.txt                   # Python dependencies
└── README.md                          # Project overview
```

---

## 2. Technology Stack

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.11+ | Ecosystem, pandas, numpy |
| Data format | Parquet (via pyarrow) | Fast I/O, columnar, compressed |
| Data manipulation | pandas + numpy | Vectorized operations |
| Visualization | plotly | Interactive charts, candlestick support |
| Notebooks | Jupyter Lab | Interactive exploration |
| Testing | pytest | Standard, fixtures, parametrize |
| Profiling | line_profiler, memory_profiler | Identify bottlenecks |
| Config | YAML (via pyyaml) | Human-readable parameters |

### Dependencies (requirements.txt)

```
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
plotly>=5.18
jupyter
jupyterlab
ipywidgets
pytest
pytest-cov
pyyaml
tqdm
smartmoneyconcepts>=0.0.21  # For validation comparison only
```

---

## 3. Data Format

### Expected Input (1m CSV)

```
Filename: {SYMBOL}_{EXCHANGE}_1m.csv or any name ending with .csv

Required columns (case-insensitive, will be normalized):
  timestamp (or datetime, date, time) -- UTC, parseable by pandas
  open
  high
  low
  close
  volume (optional but recommended)

Example:
  timestamp,open,high,low,close,volume
  2023-01-01 00:00:00,16540.5,16542.0,16538.1,16540.0,125.3
  2023-01-01 00:01:00,16540.0,16541.5,16539.0,16541.2,98.7
  ...
```

### Resampling Strategy

All higher timeframes are resampled FROM 1m data. This ensures consistency.

```python
TIMEFRAMES = {
    "1m":  None,          # Raw data
    "5m":  "5min",        # For granular analysis
    "15m": "15min",       # Local POIs
    "30m": "30min",       # Local POIs
    "1H":  "1h",          # Reversal POIs
    "4H":  "4h",          # Reversal POIs
    "1D":  "1D",          # Context
}

# Resampling rules:
#   open:   first
#   high:   max
#   low:    min
#   close:  last
#   volume: sum
```

### Caching

Resampled data is cached as parquet files in `data/processed/` to avoid re-computation.
Cache key = hash of (source file path, file modification time, timeframe).

---

## 4. Phased Implementation Plan

### Phase 1: Foundations (Data + Visualization)

**Goal:** Load 1m data, resample to all timeframes, display clean candlestick charts.

**Deliverables:**
- [ ] `data/loader.py` -- load CSV, normalize columns, validate, resample, cache
- [ ] `visualization/chart.py` -- candlestick chart with timeframe selector
- [ ] `notebooks/01_data_exploration.ipynb` -- load data, view stats, plot any day/any TF
- [ ] `tests/unit/test_data_loader.py` -- validate resampling accuracy

**Validation criteria:**
- Can load 1m CSV and display a candlestick chart for any date range
- Can switch between timeframes (1m, 15m, 30m, 1H, 4H, 1D)
- Resampled candles match expected values (manually verify a few)
- Data quality checks: no gaps, correct OHLC relationships (L <= O,C <= H)

**Detailed plan:** `docs/phases/PHASE_1_FOUNDATIONS.md`

---

### Phase 2: SMC Concept Detection (Incremental)

**Goal:** Implement all SMC concepts one by one, with visualization and testing at each step.

**Build order** (each step depends on the previous):

#### Step 2.1: Fractals (Swing Highs/Lows)
- [ ] `concepts/fractals.py`
- [ ] `visualization/overlays.py` -- swing markers (triangles)
- [ ] `notebooks/02_fractals_viewer.ipynb`
- [ ] `tests/unit/test_fractals.py`
- Validation: visually verify swing detection on real charts

#### Step 2.2: Market Structure (BOS/cBOS)
- [ ] `concepts/structure.py` -- BOS, cBOS
- [ ] Update `visualization/overlays.py` -- BOS/cBOS horizontal lines with labels
- [ ] `notebooks/03_structure_viewer.ipynb`
- [ ] `tests/unit/test_structure.py`
- Validation: verify structure matches manual chart analysis

#### Step 2.3: CISD
- [ ] Add CISD to `concepts/structure.py`
- [ ] Update visualization and notebook
- [ ] Tests for CISD detection

#### Step 2.4: Fair Value Gaps
- [ ] `concepts/fvg.py` -- detection, mitigation tracking, inversion
- [ ] Update `visualization/overlays.py` -- FVG boxes (green/red with transparency)
- [ ] `notebooks/04_fvg_viewer.ipynb` -- FVG lifecycle visualization
- [ ] `tests/unit/test_fvg.py`
- Validation: compare with `smartmoneyconcepts` library results

#### Step 2.5: Liquidity
- [ ] `concepts/liquidity.py` -- equal highs/lows, sweeps
- [ ] Update visualization -- liquidity level lines, sweep markers
- [ ] Tests

#### Step 2.6: Zones (Premium/Discount, CE/CVB)
- [ ] `concepts/zones.py`
- [ ] Tests

#### Step 2.7: POI Registry
- [ ] `concepts/registry.py` -- aggregate all concepts into composite POIs
- [ ] `notebooks/05_poi_viewer.ipynb` -- show all POIs on chart with strength scores
- [ ] Integration test: full concept chain

**Detailed plan:** `docs/phases/PHASE_2_CONCEPTS.md`

---

### Phase 3: Strategy Logic

**Goal:** Implement the IRS strategy on top of detected SMC concepts.

**Deliverables:**
- [ ] `context/mtf_manager.py` -- synchronize data across timeframes
- [ ] `context/bias.py` -- determine HTF bias from Daily/4H
- [ ] `context/sync_checker.py` -- check HTF/LTF alignment
- [ ] `context/state_machine.py` -- per-POI state tracking
- [ ] `strategy/confirmations.py` -- count and validate confirmations
- [ ] `strategy/entries.py` -- entry logic (conservative, aggressive, RTO)
- [ ] `strategy/exits.py` -- target selection, BU rules
- [ ] `strategy/fta_handler.py` -- FTA detection and handling
- [ ] `strategy/addons.py` -- add-on position logic
- [ ] `strategy/risk.py` -- position sizing and stop placement
- [ ] `notebooks/06_strategy_viewer.ipynb` -- visualize confirmations and entries on chart

**Detailed plan:** `docs/phases/PHASE_3_STRATEGY.md`

---

### Phase 4: Backtest Engine

**Goal:** Run the strategy over historical data and log all trades.

**Deliverables:**
- [ ] `engine/backtester.py` -- main loop (iterate 1m candles, update all TFs, run strategy)
- [ ] `engine/portfolio.py` -- manage positions (open, close, modify, partial)
- [ ] `engine/trade_log.py` -- detailed trade journal
- [ ] `engine/events.py` -- event system for signals and fills
- [ ] Integration tests with real data scenarios

**Backtest loop architecture:**

```python
for each 1m candle:
    1. Update all timeframe candles (check if new 15m/30m/1H/4H/1D candle closed)
    2. For each timeframe that closed a new candle:
       a. Update fractals
       b. Update structure (BOS/cBOS/CISD)
       c. Update FVG lifecycle (new FVGs, mitigations, inversions)
       d. Update liquidity (new levels, sweeps)
       e. Update POI registry
    3. Run strategy logic:
       a. Check context/bias
       b. Check sync/desync
       c. For each tracked POI: update state machine, count confirmations
       d. Check entry conditions
       e. Check exit conditions (target, BU, FTA)
       f. Check add-on conditions
    4. Execute trades (if any)
    5. Log events
```

**Detailed plan:** `docs/phases/PHASE_4_BACKTEST.md`

---

### Phase 5: Analysis & Optimization

**Goal:** Analyze backtest results, optimize parameters, validate robustness.

**Deliverables:**
- [ ] `analysis/metrics.py` -- winrate, avg RR, max drawdown, Sharpe, Sortino, profit factor
- [ ] `analysis/report.py` -- generate HTML/markdown summary report
- [ ] `analysis/equity_curve.py` -- equity curve, underwater chart, monthly returns
- [ ] `notebooks/07_backtest_results.ipynb` -- interactive results exploration
- [ ] Parameter sensitivity analysis
- [ ] Walk-forward optimization (in-sample / out-of-sample split)

**Data split for robustness:**
```
3 years of data:
  Year 1-2: In-sample (parameter tuning)
  Year 3:   Out-of-sample (validation)

Walk-forward:
  Window 1: Train on Y1, Test on Y2 first half
  Window 2: Train on Y1 + Y2 first half, Test on Y2 second half
  Window 3: Train on Y1 + Y2, Test on Y3
```

**Detailed plan:** `docs/phases/PHASE_5_ANALYSIS.md`

---

### Phase 6: Live Trading Preparation

**Goal:** Prepare the system for potential live trading deployment.

**Deliverables:**
- [ ] Refactor concept detection for streaming (one candle at a time)
- [ ] Add real-time data feed adapter (exchange WebSocket)
- [ ] Add order execution adapter (exchange API)
- [ ] Add risk management guardrails (max daily loss, max positions, etc.)
- [ ] Paper trading mode (simulated execution)
- [ ] Monitoring and alerting

**Architecture for live mode:**

```
[Data Feed] --> [Candle Aggregator] --> [Concept Engine] --> [Strategy Engine]
                                                                    |
                                                            [Risk Manager]
                                                                    |
                                                           [Order Router]
                                                                    |
                                                        [Exchange API / MT5]
```

**Note:** The existing bot at `C:\Trading\ib_trading_bot\dual_v4` can be referenced
for patterns related to MT5 integration, order management, and emulation. The IRS system
should be designed with a clean interface layer so that the execution backend can be
swapped between backtest mode, paper trading, and live trading.

**Detailed plan:** `docs/phases/PHASE_6_LIVE_PREP.md`

---

## 5. Testing Strategy

### 5.1 Unit Tests (Synthetic Data)

Each concept module gets unit tests with **synthetic candle data** specifically designed
to trigger known patterns:

```python
# Example: test_fvg.py
def test_bullish_fvg_detection():
    """Create 3 candles that form a known bullish FVG."""
    ohlc = pd.DataFrame({
        'open':  [100, 101, 104],
        'high':  [101, 105, 106],
        'low':   [99,  100, 103],   # low[2]=103 > high[0]=101 -> bullish FVG
        'close': [101, 104, 105],
    })
    result = detect_fvg(ohlc)
    assert result.iloc[2]['fvg_direction'] == 1
    assert result.iloc[2]['fvg_top'] == 103
    assert result.iloc[2]['fvg_bottom'] == 101

def test_no_fvg_when_candles_overlap():
    """Candles overlap -> no FVG."""
    ohlc = pd.DataFrame({
        'open':  [100, 101, 102],
        'high':  [102, 103, 104],
        'low':   [99,  100, 101],   # low[2]=101 <= high[0]=102 -> no FVG
        'close': [101, 102, 103],
    })
    result = detect_fvg(ohlc)
    assert result['fvg_direction'].isna().all()
```

### 5.2 Integration Tests (Concept Chain)

Test that concepts chain correctly:
```
Fractals -> Structure -> FVG chain
Fractals -> Liquidity -> Sweep detection
FVG -> IFVG (when inverted)
All concepts -> POI registry
```

### 5.3 Validation Tests (Real Data)

Use actual 1m candle data to validate detection against **manually annotated setups**:

```json
// tests/validation/setups/setup_001_btc_long.json
{
    "symbol": "BTCUSDT",
    "date": "2024-03-15",
    "timeframe": "1H",
    "expected_fractals": [
        {"index": 45, "type": "swing_low", "level": 65230.0},
        {"index": 62, "type": "swing_high", "level": 66180.0}
    ],
    "expected_fvg": [
        {"index": 58, "direction": 1, "top": 65890.0, "bottom": 65720.0}
    ],
    "expected_bos": [
        {"index": 72, "direction": 1, "broken_level": 66180.0}
    ],
    "expected_trade": {
        "direction": "long",
        "entry_price": 65750.0,
        "stop_loss": 65400.0,
        "target": 66500.0,
        "result": "win"
    }
}
```

These setups are created by the developer manually analyzing real charts and recording
expected algorithmic outputs. This is the most important test category.

### 5.4 Comparison Tests

Run `smartmoneyconcepts` library alongside our implementation on the same data.
Compare: fractal detection, FVG detection, BOS/cBOS detection.
Document and investigate any differences.

```python
# tests/validation/compare_smc_library.py
def test_fvg_matches_smc_library():
    """Our FVG detection should broadly match the reference library."""
    our_fvg = our_detect_fvg(ohlc)
    smc_fvg = smc.fvg(ohlc)
    # Allow some differences due to parameter choices
    match_rate = compute_match_rate(our_fvg, smc_fvg)
    assert match_rate > 0.90  # 90% agreement threshold
```

---

## 6. Optimization Notes

### 6.1 Vectorization Priority

**Critical:** 1m data over 3 years = ~1.5M candles. Naive Python loops will be too slow.

```
RULE: All concept detection functions must operate on entire DataFrames using
      pandas/numpy vectorized operations. No row-by-row Python loops for detection.

EXCEPTION: The backtest loop itself (engine/backtester.py) will iterate bar-by-bar
           because strategy state depends on sequential events. But concept detection
           within each bar update should be vectorized.

APPROACH:
  1. Pre-compute all static concepts (fractals, FVG, BOS) for all timeframes
     BEFORE running the backtest loop. This is a batch operation.
  2. During the backtest loop, only update LIFECYCLE states
     (FVG mitigation, liquidity sweeps) which requires
     comparing current price to pre-computed zones -- this IS vectorizable.
  3. Strategy logic (confirmations, entries) runs sequentially by necessity
     but operates on pre-computed data, so it's fast.
```

### 6.2 Memory Management

```
1m data for 3 years with 6 columns ≈ 70 MB in memory (float64).
Resampled data adds ~20% overhead.
Concept annotations (FVG, structure, etc.) add ~30% overhead.

TOTAL: ~120 MB -- fits comfortably in memory.

OPTIMIZATION: Use float32 instead of float64 where precision allows
              (saves ~50% memory, negligible precision loss for OHLC data).
```

### 6.3 Profiling Checkpoints

After each phase, profile:
- Time to load and resample data
- Time to detect all concepts (per timeframe)
- Time to run backtest (per year)
- Memory usage at peak

Target: Full 3-year backtest should complete in < 5 minutes.

---

## 7. Visualization Toolkit

### 7.1 Core Chart (`visualization/chart.py`)

Based on `plotly` with `go.Candlestick`:

```python
def create_chart(ohlc, title="", overlays=None):
    """
    Create an interactive candlestick chart with optional overlays.

    Args:
        ohlc: DataFrame with OHLC data
        overlays: List of overlay objects (FVG boxes, swing markers, structure lines, etc.)

    Returns:
        plotly.graph_objects.Figure
    """
```

### 7.2 Overlay Types (`visualization/overlays.py`)

| Overlay | Visual | Color |
|---------|--------|-------|
| Swing High | Triangle down marker | Red |
| Swing Low | Triangle up marker | Green |
| cBOS | Horizontal dashed line + label | Blue (bullish), Orange (bearish) |
| BOS | Horizontal solid line + label | Purple |
| CISD | Horizontal dotted line + label | Cyan |
| Bullish FVG | Semi-transparent green box | rgba(0, 255, 0, 0.15) |
| Bearish FVG | Semi-transparent red box | rgba(255, 0, 0, 0.15) |
| IFVG | Hatched/striped box | Different shade |
| Liquidity Level | Horizontal dotted line | Yellow |
| Sweep | X marker at sweep point | Yellow |
| Entry | Arrow marker | Green (long), Red (short) |
| Stop Loss | Horizontal line | Red |
| Target | Horizontal line | Blue |
| BU Level | Horizontal line | Gray |

### 7.3 Interactive Notebooks

Each notebook follows this pattern:

```python
# Imports and data loading
from data.loader import load_data
from concepts.fractals import detect_fractals
from visualization.chart import create_chart
from visualization.overlays import add_fractals_overlay

# Load data for a specific date range
ohlc_1m = load_data("data/raw/BTCUSDT_1m.csv")
ohlc_15m = resample(ohlc_1m, "15m")

# Detect concepts
fractals = detect_fractals(ohlc_15m, swing_length=5)

# Create chart with overlays
fig = create_chart(ohlc_15m, title="BTCUSDT 15m - Fractals")
fig = add_fractals_overlay(fig, ohlc_15m, fractals)
fig.show()

# Interactive widgets for parameter exploration
import ipywidgets as widgets

@widgets.interact(swing_length=(3, 15, 1), date=date_slider)
def explore(swing_length, date):
    data = filter_by_date(ohlc_15m, date)
    fractals = detect_fractals(data, swing_length=swing_length)
    fig = create_chart(data)
    fig = add_fractals_overlay(fig, data, fractals)
    fig.show()
```

### 7.4 Trade Review Tool (notebook 07)

Interactive tool to review each backtest trade:

```python
# For each trade in the backtest log:
#   1. Show multi-TF chart context (4H for bias, 1H for POI, 15m for structure, 1m for entry)
#   2. Mark all confirmations that were counted
#   3. Show entry point, stop loss, target, actual exit
#   4. Show P&L and RR achieved
#   5. Navigation: "Previous Trade" / "Next Trade" buttons
#   6. Filter: Winners only, Losers only, By date, By RR threshold
```

---

## 8. Phase Dependencies and Timeline

```
Phase 1: Foundations          [Week 1]
  └─> Phase 2: Concepts      [Week 2-4]  (incremental, step by step)
       └─> Phase 3: Strategy  [Week 5-6]
            └─> Phase 4: Backtest Engine  [Week 7-8]
                 └─> Phase 5: Analysis    [Week 9-10]
                      └─> Phase 6: Live Prep  [Week 11+]
```

Each phase has a **gate check** before proceeding:
- Phase 1 gate: Can load data and display charts for any day/TF
- Phase 2 gate: All concepts detected correctly on test data, visual validation passes
- Phase 3 gate: Strategy logic produces expected signals on known setups
- Phase 4 gate: Backtest runs to completion on full dataset, trades logged correctly
- Phase 5 gate: Performance metrics computed, walk-forward shows consistency
- Phase 6 gate: Paper trading matches backtest behavior

---

## 9. Configuration System

```yaml
# config.yaml

data:
  symbol: "BTCUSDT"
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  timeframes: ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]

concepts:
  fractals:
    swing_length:
      1m: 3
      5m: 5
      15m: 5
      30m: 5
      1H: 7
      4H: 10
      1D: 10

  structure:
    break_mode: "close"         # "close" or "wick"
    min_displacement: 0.001     # Minimum % move to count as displacement

  fvg:
    min_gap_pct: 0.0005         # Minimum FVG size as % of price
    join_consecutive: true
    mitigation_mode: "close"    # "wick", "close", "ce", "full"

  liquidity:
    range_percent: 0.001        # How close levels must be for "equal"
    min_touches: 2              # Minimum touches to form liquidity

strategy:
  confirmations:
    min_count: 5
    max_count: 8

  entry:
    mode: "conservative"        # "conservative", "aggressive", "rto_only"
    rto_wait: true              # Wait for RTO on aggressive exits

  breakeven:
    structural_bu: true
    fta_bu: true
    range_bu: true

  risk:
    position_size_sync: 1.0
    position_size_desync: 0.5
    max_risk_per_trade: 0.02    # 2% of account
    max_concurrent_positions: 3

  targets:
    primary_tf: ["4H", "1H"]
    local_tf: ["30m", "15m"]

backtest:
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  initial_capital: 10000
  commission_pct: 0.0006        # 0.06% per trade (taker fee)
  slippage_pct: 0.0002          # 0.02% estimated slippage
```

---

## 10. Key Design Principles

### 10.1 Separation of Concerns

```
Concept Detection (concepts/) -- PURE functions, no state, no strategy knowledge.
  Input: OHLC DataFrame
  Output: DataFrame with annotations (FVG zones, BOS events, etc.)

Context Analysis (context/) -- Combines concept outputs across timeframes.
  Input: Concept annotations from multiple TFs
  Output: Market state, bias, sync/desync

Strategy Logic (strategy/) -- Applies trading rules to context.
  Input: Market state, POI registry, current positions
  Output: Entry/exit/modify signals

Execution Engine (engine/) -- Executes signals and tracks positions.
  Input: Signals
  Output: Trade log, portfolio state
```

### 10.2 Testability

Every function should be independently testable with synthetic or real data.
No hidden dependencies. No global state. Configuration injected, not hardcoded.

### 10.3 Reproducibility

Given the same data and configuration, the backtest must produce identical results.
No randomness (unless explicitly configured for Monte Carlo analysis in Phase 5).

### 10.4 Live-Ready Architecture

From Phase 1, design with streaming in mind:
- Concept detection functions should work both on full DataFrames (batch)
  and on incremental updates (append one candle, recompute affected concepts)
- This dual-mode is NOT required in Phase 1-4 but the API should not prevent it

---

## 11. Risk and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting to historical data | High | Walk-forward validation, out-of-sample testing |
| SMC concept subjectivity | High | Formal specs in docs, multiple parameter configs, visual validation |
| Performance bottleneck on 1m data | Medium | Vectorization, pre-computation, profiling gates |
| Data quality issues (gaps, errors) | Medium | Data validation in loader, gap-filling strategy |
| Complexity explosion in multi-TF | High | Strict phase gates, test each layer before next |
| CISD/IFVG implementation ambiguity | Medium | Compare with TradingView indicators, visual checks |

---

## 12. Getting Started (First Session Checklist)

```
1. Create project directory and virtual environment
2. Install dependencies from requirements.txt
3. Place 1m CSV data files in data/raw/
4. Run Phase 1: data loader + basic chart
5. Open notebook 01, verify data loads and displays correctly
6. Proceed to Phase 2, Step 2.1 (fractals)
```

The project is designed for incremental progress. Each step produces a working,
testable, and visually verifiable artifact before moving to the next.
