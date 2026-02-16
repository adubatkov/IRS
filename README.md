# IRS - Intraday High Risk-Reward System

Backtesting framework for an SMC/ICT-based trading strategy that requires 5+ confirmations at Points of Interest (POI) before entry, operates across 7 timeframes (1D to 1m), and manages positions with structural breakeven, FTA handling, and add-ons.

## Instruments

| Instrument | Source | Period |
|-----------|--------|--------|
| NAS100 | Forex.com | 2022-12 to 2026-02 |
| GER40 | Pepperstone | 2023-01 to 2026-02 |
| UK100 | Forex.com | 2022-12 to 2026-02 |
| XAUUSD | Oanda | 2022-12 to 2026-02 |

## Project Structure

```
data/           - Data loading and resampling
concepts/       - SMC concept detection (stateless pure functions)
context/        - Multi-timeframe analysis and market state
strategy/       - Entry/exit/confirmation logic
engine/         - Backtest execution engine
analysis/       - Post-backtest metrics and reports
visualization/  - Plotly-based charting
notebooks/      - Jupyter exploration notebooks
tests/          - Unit, integration, validation tests
```

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/ -v
jupyter lab notebooks/
```

## Strategy Overview

The IRS strategy is reactive (not predictive). It waits for price to interact with a Point of Interest, collects confirmations (minimum 5), and enters with a tight stop-loss at structurally significant levels.

**Confirmation types**: POI Tap, Liquidity Sweep, FVG Inversion, Inversion Test, Structure Break, FVG Test, CVB Test, OB Test, Breaker Test, Additional BOS.

See `doc/01_STRATEGY.md` for full specification.

## Documentation

- `doc/01_STRATEGY.md` - Strategy specification
- `doc/02_SMC_CONCEPTS.md` - SMC/ICT technical reference
- `doc/03_PROJECT_PLAN.md` - Implementation plan
