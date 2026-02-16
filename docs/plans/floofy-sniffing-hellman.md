# IRS Backtest - Implementation Plan

## Context

Building a backtesting system for the IRS (Intraday High Risk-Reward System) trading strategy based on Smart Money Concepts (SMC/ICT). The strategy requires 5+ confirmations at Points of Interest (POI) before entry, uses 7 timeframes (1D to 1m), and manages positions with structural breakeven, FTA handling, and add-ons.

**Goal**: Implement the full system per `doc/01_STRATEGY.md`, `doc/02_SMC_CONCEPTS.md`, `doc/03_PROJECT_PLAN.md`.

**GitHub**: https://github.com/adubatkov/IRS.git

**Decisions**:
- **Primary instrument**: NAS100 (1.1M rows, 2022-12 to 2026-02) - develop and debug on this, add others later
- **Code language**: English for all code, comments, docstrings, README
- **News data**: Postpone to Phase 5 as optional filter (ForexFactory High Impact events)

## Available Data

| Instrument | Rows | Date Range | Source |
|-----------|------|------------|--------|
| GER40 | 917K | 2023-01 to 2026-02 | Pepperstone |
| NAS100 | 1.1M | 2022-12 to 2026-02 | Forex.com |
| UK100 | 1.07M | 2022-12 to 2026-02 | Forex.com |
| XAUUSD | 1.09M | 2022-12 to 2026-02 | Oanda |

- Format: parquet (`data/optimized/*.parquet`), schema: `time, open, high, low, close, tick_volume`
- Raw CSVs also available in subdirectories (split files from TradingView export)
- News: `data/news/forex_factory_*.json` (2023-2026, impact/forecast/actual)

## Environment

- Python 3.11.9, pandas 2.3.3, numpy 2.3.3, pyarrow 22.0.0, pytest 8.4.2, PyYAML 6.0.3, JupyterLab 4.4.10
- **Need to install**: plotly, smartmoneyconcepts (reference), mypy, ruff

---

## Phase 0: Project Setup (git init, deps, config)

### Step 0.1: Git Init + README
- `git init` in `C:\Trading\IRS`
- Add remote `https://github.com/adubatkov/IRS.git`
- Create `.gitignore` (data/raw/, data/optimized/*.parquet, __pycache__, .venv, *.egg-info, .pytest_cache)
- Create `README.md` with project overview
- Create `PROJECT_INFO.md` in root with GitHub URL and key metadata

**Files**: `.gitignore`, `README.md`, `PROJECT_INFO.md`

### Step 0.2: Dependencies
- Create `requirements.txt` with pinned versions
- Install: `plotly`, `smartmoneyconcepts` (reference), `mypy`, `ruff`
- Verify all imports work

**Files**: `requirements.txt`

### Step 0.3: Configuration
- Create `config.yaml` with all parameters from `doc/03_PROJECT_PLAN.md`
- Create `config.py` to load YAML into Python dataclass
- Default instrument: NAS100, others as secondary (GER40, UK100, XAUUSD)

**Files**: `config.yaml`, `config.py`

### Step 0.4: Project Directory Scaffold
- Create all directories per plan: `concepts/`, `context/`, `strategy/`, `engine/`, `analysis/`, `visualization/`, `notebooks/`, `tests/unit/`, `tests/integration/`, `tests/validation/`
- Add `__init__.py` to each package

**Health Check**: `python -c "from config import Config; c = Config(); print(c)"` works, ruff/mypy pass

---

## Phase 1: Data Foundations

### Step 1.1: Data Loader (`data/loader.py`)
- Load parquet from `data/optimized/` (primary path)
- Fallback: load and merge split CSV files from raw directories
- Validate schema: time (UTC datetime), open, high, low, close, tick_volume
- Handle XAUUSD extra "Shapes" columns (drop on CSV load)
- Handle gaps: detect, log, optionally forward-fill
- Return clean `pd.DataFrame` with DatetimeIndex (UTC)

**Files**: `data/__init__.py`, `data/loader.py`
**Tests**: `tests/unit/test_loader.py` - load parquet, load CSV, validate schema, detect gaps

### Step 1.2: Timeframe Resampler (`data/resampler.py`)
- Resample 1m data to: 5m, 15m, 30m, 1H, 4H, 1D
- OHLC rules: open=first, high=max, low=min, close=last, tick_volume=sum
- Cache resampled data as parquet in `data/processed/`
- Hash-based cache invalidation (file mtime + params)

**Files**: `data/resampler.py`, `data/processed/` (gitignored)
**Tests**: `tests/unit/test_resampler.py` - verify OHLC accuracy, boundary alignment, known values

### Step 1.3: Basic Visualization (`visualization/chart.py`)
- Plotly candlestick chart with timeframe selector
- Date range selector / zoom
- Display any instrument/timeframe/date range
- Utility function for notebook use

**Files**: `visualization/__init__.py`, `visualization/chart.py`
**Tests**: Visual verification via `notebooks/01_data_exploration.ipynb`

### Step 1.4: Notebook 01 - Data Exploration
- Load all 4 instruments
- Show stats: rows, date range, gaps
- Display candlestick charts at different TFs
- Verify resampling visually

**Files**: `notebooks/01_data_exploration.ipynb`

**Health Check Phase 1**:
- `pytest tests/unit/test_loader.py tests/unit/test_resampler.py -v`
- `ruff check data/ visualization/`
- `mypy data/ visualization/ --ignore-missing-imports`
- Visual: notebook charts match TradingView

---

## Phase 2: SMC Concept Detection (incremental, each step testable)

All concept modules are **pure functions** taking DataFrame + params, returning DataFrame with annotations.

### Step 2.1: Fractals (`concepts/fractals.py`)
- Detect swing highs/lows with configurable `swing_length` per timeframe
- Vectorized: rolling window comparison
- Return DataFrame with columns: `swing_high` (bool), `swing_low` (bool), `swing_high_price`, `swing_low_price`
- Lifecycle tracking: ACTIVE, SWEPT, BROKEN

**Tests**: `tests/unit/test_fractals.py` - synthetic zigzag data, known peaks/troughs
**Notebook**: `notebooks/02_fractals_viewer.ipynb` - triangle markers on chart

### Step 2.2: Market Structure (`concepts/structure.py`)
- BOS (Break of Structure): continuation break
- CHoCH (Change of Character): reversal break
- Both modes: `close` break vs `wick` break
- Depends on fractals output
- Return: event type, direction, level, candle index

**Tests**: `tests/unit/test_structure.py` - synthetic uptrend/downtrend with known BOS/CHoCH
**Notebook**: `notebooks/03_structure_viewer.ipynb`

### Step 2.3: CISD (`concepts/structure.py` extension)
- Change in State of Delivery - early momentum shift
- Break of candle opening prices
- Enhanced version with liquidity sweep confirmation

**Tests**: `tests/unit/test_cisd.py`

### Step 2.4: Fair Value Gaps (`concepts/fvg.py`)
- 3-candle pattern detection (vectorized)
- Lifecycle: FRESH -> TESTED -> PARTIALLY_FILLED -> FULLY_FILLED -> MITIGATED -> INVERTED
- Mitigation modes: wick, close, ce, full
- Track: zone high/low, midpoint (CE), fill percentage

**Tests**: `tests/unit/test_fvg.py` - synthetic 3-candle patterns, lifecycle transitions
**Notebook**: `notebooks/04_fvg_viewer.ipynb` - colored rectangles on chart

### Step 2.5: Order Blocks (`concepts/orderblocks.py`)
- Last opposing candle before displacement
- Requires preceding BOS/CHoCH
- Lifecycle: ACTIVE -> TESTED -> MITIGATED -> BROKEN
- Configurable max_age_candles

**Tests**: `tests/unit/test_orderblocks.py`
**Notebook**: `notebooks/05_orderblock_viewer.ipynb`

### Step 2.6: Breaker Blocks (`concepts/breakers.py`)
- Failed Order Block detection (price closed through OB)
- Role inversion logic
- Lifecycle management

**Tests**: `tests/unit/test_breakers.py`

### Step 2.7: Liquidity (`concepts/liquidity.py`)
- Equal highs/lows detection (range_percent tolerance)
- Session highs/lows (prev day, prev week, prev month)
- Sweep detection: wick past level, close doesn't break

**Tests**: `tests/unit/test_liquidity.py`
**Notebook**: `notebooks/06_poi_viewer.ipynb` (shared with 2.9)

### Step 2.8: Zones (`concepts/zones.py`)
- Premium/Discount zones between swing high and swing low
- CE/CVB calculation for any FVG/OB/range

**Tests**: `tests/unit/test_zones.py`

### Step 2.9: POI Registry (`concepts/registry.py`)
- Aggregate FVG + OB + BB + IFVG + Liquidity + Session levels into composite POIs
- POI strength scoring per doc/02 spec
- Confluence bonus calculation
- Track active POIs across all timeframes

**Tests**: `tests/unit/test_registry.py`, `tests/integration/test_concept_chain.py`
**Notebook**: `notebooks/06_poi_viewer.ipynb` - all concepts overlaid

### Step 2.10: Comparison with `smartmoneyconcepts` library
- Compare our fractals, FVG, BOS/CHoCH, OB, liquidity with library output
- Target: 90%+ match on shared concepts

**Tests**: `tests/validation/test_vs_smartmoneyconcepts.py`

**Health Check Phase 2** (after each sub-step):
- `pytest tests/unit/test_<concept>.py -v`
- `ruff check concepts/`
- `mypy concepts/ --ignore-missing-imports`
- Visual verification in corresponding notebook

---

## Phase 3: Multi-TF Context + Strategy Logic

### Step 3.1: MTF Manager (`context/mtf_manager.py`)
- Maintain synchronized state across all 7 timeframes
- When new 1m candle arrives, determine which higher TFs also have a new candle
- Store concept results per timeframe

### Step 3.2: Bias & Sync (`context/bias.py`, `context/sync_checker.py`)
- Determine market direction from HTF (1D, 4H)
- Detect sync vs desync between HTF and LTF
- Position sizing rules: full size (sync), reduced (desync)

### Step 3.3: POI State Machine (`context/state_machine.py`)
- States: IDLE -> POI_TAPPED -> COLLECTING -> READY -> POSITIONED -> MANAGING -> CLOSED
- Transition logic per doc/01 spec

### Step 3.4: Confirmation Counter (`strategy/confirmations.py`)
- Track 5+ confirmations: POI Tap, Liquidity Sweep, FVG Inversion, Inversion Test, Structure Break
- Additional: FVG test, CVB test, OB test, Breaker test, additional BOS
- Validate min_count (5) / max_count (8)

### Step 3.5: Entry Logic (`strategy/entries.py`)
- Conservative: 5+ confirms, structural exit, stop behind liquidity
- Aggressive: with structural breakeven
- RTO-based: wait for return to origin

### Step 3.6: Exit Logic (`strategy/exits.py`)
- Target selection: 4H/1H POI (sync), 15m/30m fractal (desync)
- BU types: Structural, FTA-based, Range boundary
- Position flip logic

### Step 3.7: FTA Handler (`strategy/fta_handler.py`)
- Detect first trouble area on path to target
- Far FTA: enter normally
- Close FTA: wait for invalidation
- FTA validates: move to BU

### Step 3.8: Add-Ons (`strategy/addons.py`)
- Local POI add-ons from 15m/30m
- Short BU add-ons with structural protection

### Step 3.9: Risk Management (`strategy/risk.py`)
- Position sizing: sync vs desync
- Max risk per trade, max concurrent positions
- Stop-loss placement (behind liquidity zone)

**Tests**: `tests/unit/test_confirmations.py`, `tests/unit/test_entries.py`, `tests/unit/test_exits.py`
**Notebook**: `notebooks/07_strategy_viewer.ipynb`

**Health Check Phase 3**: Full test suite, all concepts + strategy tests green

---

## Phase 4: Backtest Engine

### Step 4.1: Event System (`engine/events.py`)
- Event types: NEW_CANDLE, CONCEPT_UPDATE, SIGNAL, FILL, CLOSE, BU_MOVE, ADDON

### Step 4.2: Portfolio (`engine/portfolio.py`)
- Open/close/modify positions
- Track equity, margin, P&L
- Commission and slippage modeling

### Step 4.3: Trade Log (`engine/trade_log.py`)
- Full metadata per trade: entry/exit time, price, confirmations, TF context, BU events, add-ons

### Step 4.4: Main Backtest Loop (`engine/backtester.py`)
```
for each 1m candle:
  1. Update all TF candles (mtf_manager)
  2. Update concepts per TF (fractals, structure, FVG, OB, BB, liquidity, POI)
  3. Run strategy (bias, sync, state machine, confirmations, entry/exit)
  4. Execute trades (portfolio)
  5. Log events (trade_log)
```

**Tests**: `tests/integration/test_backtest_engine.py` - known setup scenarios
**Notebook**: `notebooks/08_backtest_results.ipynb`

**Health Check Phase 4**: Run on 1 month of data, verify trades make sense visually

---

## Phase 5: Analysis & Optimization

### Step 5.1: Performance Metrics (`analysis/metrics.py`)
- Win rate, avg RR, max drawdown, Sharpe, Sortino, profit factor, total P&L

### Step 5.2: Reports (`analysis/report.py`)
- Summary by instrument, timeframe, period
- Best/worst trades analysis

### Step 5.3: Equity Curve (`analysis/equity_curve.py`)
- Equity curve, underwater chart, monthly returns heatmap

### Step 5.4: Walk-Forward Optimization
- In-sample: 2023-2024, Out-of-sample: 2025-2026
- Parameterize: swing_length, min_confirmations, FVG gap threshold, etc.

**Health Check Phase 5**: Out-of-sample results don't degrade >30% vs in-sample

---

## Phase 6: Live Trading Prep (future)

- Refactor for streaming (one candle at a time)
- Real-time data feed adapter
- Paper trading mode
- Risk management guardrails

---

## Execution Order Summary

| Step | Description | Depends On | Deliverables |
|------|-------------|------------|-------------|
| 0.1 | Git + README | - | .gitignore, README.md, PROJECT_INFO.md |
| 0.2 | Dependencies | 0.1 | requirements.txt |
| 0.3 | Config | 0.2 | config.yaml, config.py |
| 0.4 | Scaffold | 0.1 | all directories + __init__.py |
| 1.1 | Data Loader | 0.3 | data/loader.py + tests |
| 1.2 | Resampler | 1.1 | data/resampler.py + tests |
| 1.3 | Chart | 0.2 | visualization/chart.py |
| 1.4 | Notebook 01 | 1.1, 1.2, 1.3 | notebooks/01_data_exploration.ipynb |
| 2.1 | Fractals | 1.2 | concepts/fractals.py + tests + notebook |
| 2.2 | Structure | 2.1 | concepts/structure.py + tests + notebook |
| 2.3 | CISD | 2.2 | concepts/structure.py ext + tests |
| 2.4 | FVG | 1.2 | concepts/fvg.py + tests + notebook |
| 2.5 | Order Blocks | 2.2 | concepts/orderblocks.py + tests + notebook |
| 2.6 | Breakers | 2.5 | concepts/breakers.py + tests |
| 2.7 | Liquidity | 2.1 | concepts/liquidity.py + tests |
| 2.8 | Zones | 2.1 | concepts/zones.py + tests |
| 2.9 | POI Registry | 2.1-2.8 | concepts/registry.py + tests + notebook |
| 2.10 | SMC Comparison | 2.1-2.8 | tests/validation/ |
| 3.1-3.9 | Strategy | Phase 2 | context/ + strategy/ + tests + notebook |
| 4.1-4.4 | Engine | Phase 3 | engine/ + tests + notebook |
| 5.1-5.4 | Analysis | Phase 4 | analysis/ + notebook |

## Health Check Protocol (after every step)

1. **Tests**: `pytest tests/ -v --tb=short` (relevant test files)
2. **Linting**: `ruff check <module>/`
3. **Types**: `mypy <module>/ --ignore-missing-imports`
4. **Visual**: Check corresponding notebook (if applicable)
5. **Commit**: `git add . && git commit -m "Phase X.Y: <description>"`
6. **Push**: `git push`

## Key Design Decisions

1. **Instruments**: GER40, NAS100, UK100, XAUUSD (not BTCUSDT as in doc - adapt config)
2. **Primary data source**: `data/optimized/*.parquet` (already merged + cleaned)
3. **Vectorized operations only** - no row-by-row loops in concept detection
4. **Pure functions** in `concepts/` - stateless, take DataFrame, return annotated DataFrame
5. **State in `context/`** - MTF manager and state machine hold mutable state
6. **Config-driven** - all parameters in `config.yaml`, no magic numbers in code
