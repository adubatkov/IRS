# SMC/ICT Concepts -- Technical Reference for Algorithmic Implementation

> Purpose: Exhaustive reference of all Smart Money Concepts (SMC) / Inner Circle Trader (ICT)
> tools used in the IRS strategy. Each concept includes: definition, detection algorithm,
> lifecycle (creation -> active -> mitigated/invalidated), and implementation notes.
> This document serves as a prompt/reference for both the developer and Claude Code.

---

## Table of Contents

1. [Swing Highs & Lows (Fractals)](#1-swing-highs--lows-fractals)
2. [Market Structure: BOS & cBOS](#2-market-structure-bos--cbos)
3. [Change in State of Delivery (CISD)](#3-change-in-state-of-delivery-cisd)
4. [Fair Value Gap (FVG)](#4-fair-value-gap-fvg)
5. [Inverted Fair Value Gap (IFVG)](#5-inverted-fair-value-gap-ifvg)
6. [Liquidity](#6-liquidity)
7. [Return to Origin (RTO)](#7-return-to-origin-rto)
8. [Point of Interest (POI)](#8-point-of-interest-poi)
9. [First Trouble Area (FTA)](#9-first-trouble-area-fta)
10. [Premium & Discount Zones](#10-premium--discount-zones)
11. [Consequent Encroachment (CE / CVB)](#11-consequent-encroachment-ce--cvb)
12. [Existing Python Libraries](#12-existing-python-libraries)

---

## 1. Swing Highs & Lows (Fractals)

### Definition

A **swing high** is a candle whose high is the highest among `N` candles on each side.
A **swing low** is a candle whose low is the lowest among `N` candles on each side.
These are the building blocks of all structure analysis.

### Detection Algorithm

```
Parameters:
  swing_length: int = 5  (number of candles on each side)

Bullish Fractal (Swing High) at index i:
  high[i] == max(high[i-swing_length : i+swing_length+1])
  AND high[i] > high[i-1] AND high[i] > high[i+1]

Bearish Fractal (Swing Low) at index i:
  low[i] == min(low[i-swing_length : i+swing_length+1])
  AND low[i] < low[i-1] AND low[i] < low[i+1]
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `index` | int | Candle index where fractal was detected |
| `level` | float | Price level (high for swing high, low for swing low) |
| `direction` | int | +1 for swing high, -1 for swing low |
| `timeframe` | str | Timeframe on which it was detected |
| `status` | enum | `ACTIVE`, `SWEPT`, `BROKEN` |

### Lifecycle

```
ACTIVE:  Fractal detected, level is intact
SWEPT:   Price wick went past the level but closed back (liquidity sweep)
BROKEN:  Price closed beyond the level (used for BOS/cBOS detection)
```

### Implementation Notes

- `swing_length` should vary by timeframe: smaller for LTF (3-5), larger for HTF (5-10)
- Fractals on the current bar require `swing_length` future candles to confirm (look-ahead).
  In backtesting, this means fractal confirmation is delayed by `swing_length` bars.
- Use vectorized approach: `pandas.DataFrame.rolling` with custom window comparisons

### Multi-Timeframe Consideration

Fractals detected on 15m are "minor" (local structure).
Fractals detected on 1H/4H are "swing" (major structure).
The strategy references both "minor BOS" (слом минорный) and "swing BOS" (слом свинговый).

---

## 2. Market Structure: BOS & cBOS

### Definition

**BOS (Break of Structure)** -- price breaks a swing high/low in the OPPOSITE direction
to the prevailing trend. This is a potential trend reversal signal.

**cBOS (Continuation BOS)** -- price breaks a swing high/low in the SAME direction as the
prevailing trend. This is a trend continuation signal.

### Detection Algorithm

```
Requires: Ordered sequence of swing highs and lows

Track current trend: BULLISH or BEARISH

BULLISH trend:
  cBOS:  New candle breaks above the most recent swing HIGH
         -> Trend continuation (still bullish)
  BOS:   New candle breaks below the most recent swing LOW
         -> Trend reversal signal (shift to bearish)

BEARISH trend:
  cBOS:  New candle breaks below the most recent swing LOW
         -> Trend continuation (still bearish)
  BOS:   New candle breaks above the most recent swing HIGH
         -> Trend reversal signal (shift to bullish)

Break detection modes:
  close_break=True:  Break requires candle CLOSE beyond the level (stricter, more reliable)
  close_break=False: Break requires only wick/high/low beyond the level (more sensitive)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | enum | `BOS` or `CBOS` |
| `direction` | int | +1 bullish, -1 bearish |
| `broken_level` | float | The swing level that was broken |
| `broken_index` | int | Index of the candle that performed the break |
| `swing_index` | int | Index of the swing that was broken |

### Relationship to Strategy

In the IRS strategy, "слом структуры" (structure break) can refer to either BOS or cBOS
depending on context:
- **Within a POI (reversal context)**: Structure break = BOS (break of structure)
  because price is expected to reverse
- **During movement to target**: Structure break = cBOS (continuation)
  because price is expected to continue

This is a **configurable parameter** in the backtester.

### Implementation Notes

- Requires maintaining an ordered list of swing highs and lows
- The `close_break` parameter significantly affects results -- test both modes
- Reference implementation: `smartmoneyconcepts` library `bos_choch()` method

---

## 3. Change in State of Delivery (CISD)

### Definition

**CISD** is an early reversal signal that detects momentum shifts BEFORE traditional BOS.
It identifies the point where price delivery changes direction by breaking the opening price of
the candle(s) that initiated the previous directional sequence.

CISD is **faster** than BOS but **less reliable**. It provides earlier entry opportunities
at the cost of more false signals.

### Detection Algorithm

```
Bullish CISD:
  1. Identify a sequence of bearish candles (close < open)
  2. Mark the opening price of the FIRST candle in this bearish sequence
  3. Bullish CISD confirmed when a subsequent candle CLOSES ABOVE this opening price

Bearish CISD:
  1. Identify a sequence of bullish candles (close > open)
  2. Mark the opening price of the FIRST candle in this bullish sequence
  3. Bearish CISD confirmed when a subsequent candle CLOSES BELOW this opening price

Enhanced version (with liquidity sweep):
  1. Detect swing high/low (liquidity level)
  2. Wait for price to WICK through the level (sweep)
  3. Then apply standard CISD logic from the sweep point
  -> This produces stronger signals with higher accuracy
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `direction` | int | +1 bullish, -1 bearish |
| `level` | float | The opening price that was broken |
| `trigger_index` | int | Candle that confirmed the CISD |
| `origin_index` | int | First candle of the opposing sequence |
| `is_sweep_confirmed` | bool | Whether a liquidity sweep preceded the CISD |
| `is_continuation` | bool | True if same direction as current trend (dashed line) |
| `is_reversal` | bool | True if opposite direction (solid line) |

### Role in Strategy

In the IRS strategy, CISD serves as:
- An **early confirmation** within a POI (counts toward the 5-confirm threshold)
- A signal for **momentum shift** that may precede the traditional BOS confirmation
- CISD focuses on candle body (open/close), NOT wicks -- more precise for delivery analysis

### Implementation Notes

- Key difference from BOS: CISD uses candle opens/closes, BOS uses swing levels
- CISD can generate many false signals in ranging markets -- use with HTF context filter
- LuxAlgo and AlgoAlpha have TradingView implementations with configurable parameters:
  - `Swing Period`: Controls swing detection granularity
  - `Noise Filter`: Minimum duration/strength for valid CISD
  - `Liquidity Lookback`: How recent a sweep must be for confirmation
- Reference: LuxAlgo CISD indicator, AlgoAlpha CISD indicator (TradingView)

---

## 4. Fair Value Gap (FVG)

### Definition

An **FVG** is a three-candle pattern where price moved so aggressively that it left an
"untouched" zone -- the gap between the wick of candle 1 and the wick of candle 3,
with candle 2 being the displacement candle.

FVGs represent areas where supply/demand was so one-sided that price skipped over a
range. Price tends to return to these zones to "fill" the imbalance.

### Detection Algorithm

```
Bullish FVG at candle index i:
  CONDITION: low[i] > high[i-2]
  GAP_TOP:    low[i]      (low of candle 3)
  GAP_BOTTOM: high[i-2]   (high of candle 1)
  FVG_ZONE:   [high[i-2], low[i]]

Bearish FVG at candle index i:
  CONDITION: high[i] < low[i-2]
  GAP_TOP:    low[i-2]    (low of candle 1)
  GAP_BOTTOM: high[i]     (high of candle 3)
  FVG_ZONE:   [high[i], low[i-2]]

Optional filters:
  - Minimum gap size: abs(top - bottom) > min_gap_pct * close[i]
  - Join consecutive: If multiple FVGs appear in succession, merge into one
    using the highest top and lowest bottom
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `direction` | int | +1 bullish, -1 bearish |
| `top` | float | Upper boundary of the gap |
| `bottom` | float | Lower boundary of the gap |
| `midpoint` | float | (top + bottom) / 2 -- the CE/CVB level |
| `creation_index` | int | Index of candle 3 (confirmation candle) |
| `timeframe` | str | Timeframe on which FVG was detected |
| `status` | enum | See lifecycle below |

### Lifecycle

```
FRESH:      Just created, price hasn't returned to the zone
TESTED:     Price has touched the zone (wick into it) but not filled it
PARTIALLY_FILLED: Price has filled past the midpoint (CE) but not fully
FULLY_FILLED:     Price has closed through the entire zone
MITIGATED:  Fully filled -- zone no longer active
INVERTED:   Price closed through the zone and it now acts as opposite S/R (becomes IFVG)
```

### Mitigation Modes

```
Mode "wick":  FVG is mitigated when any wick touches the zone
Mode "close": FVG is mitigated when a candle CLOSES within/through the zone
Mode "ce":    FVG is mitigated when price reaches the midpoint (CE level)
Mode "full":  FVG is mitigated only when price closes through the ENTIRE zone
```

### Role in Strategy

FVGs are central to the IRS strategy:
- **FVG test** = a confirmation event (counts toward 5-confirm threshold)
- **FVG inversion** = a confirmation event (price closes through FVG, changing its role)
- **Stop-loss placement** = behind an FVG zone
- **RTO target** = price returns to test an FVG after aggressive movement
- **Entry zone** = price touching an FVG with reaction

### Implementation Notes

- FVGs on higher timeframes (4H, 1H) are more significant than on lower (15m, 1m)
- In practice, many FVGs are "noise" -- filter by minimum size and structural significance
  (FVG that caused a BOS is more important than random FVG)
- The `smartmoneyconcepts` library provides `fvg()` with `join_consecutive` parameter
- Track FVGs as active objects with state transitions

---

## 5. Inverted Fair Value Gap (IFVG)

### Definition

An **IFVG** is a regular FVG that has been invalidated -- price closed through it, and it
now acts as support/resistance from the OPPOSITE direction.

### Detection Algorithm

```
Bullish IFVG (former bearish FVG):
  1. A bearish FVG exists (gap zone above price)
  2. Price closes ABOVE the top of this bearish FVG
  3. The bearish FVG is now a bullish IFVG
  4. When price returns to this zone, it may act as SUPPORT

Bearish IFVG (former bullish FVG):
  1. A bullish FVG exists (gap zone below price)
  2. Price closes BELOW the bottom of this bullish FVG
  3. The bullish FVG is now a bearish IFVG
  4. When price returns to this zone, it may act as RESISTANCE
```

### Properties

Same as FVG, plus:

| Property | Type | Description |
|----------|------|-------------|
| `original_direction` | int | Direction of the original FVG before inversion |
| `inversion_index` | int | Candle index where inversion occurred |

### Role in Strategy

- **FVG Inversion** is the 3rd core confirmation in the IRS system
- **Inversion Test** is the 4th core confirmation
- IFVGs serve as entry zones and stop-loss placement areas
- An IFVG test with wick reaction = valid confirmation

### Implementation Notes

- IFVG detection requires tracking all FVG objects and monitoring for inversions
- An FVG can only be inverted once -- after inversion, it becomes an IFVG with its own lifecycle
- IFVG can be mitigated (price returns and trades through it again) or respected (price bounces)

---

## 6. Liquidity

### Definition

**Liquidity** represents clusters of stop-loss orders and pending orders at specific price levels.
In SMC theory, large players ("smart money") drive price toward these levels to collect liquidity
before reversing.

### Types of Liquidity

```
1. EQUAL HIGHS (EQH): Multiple swing highs at approximately the same price level
   -> Stop-loss orders of short sellers sit above these levels
   -> "Sell-side liquidity" from the perspective of order flow

2. EQUAL LOWS (EQL): Multiple swing lows at approximately the same price level
   -> Stop-loss orders of long buyers sit below these levels
   -> "Buy-side liquidity" from the perspective of order flow

3. TRENDLINE LIQUIDITY: Stop-losses placed along a trendline
   -> Less precise, harder to detect algorithmically

4. SESSION HIGHS/LOWS: Previous session, day, week, or month highs/lows
   -> Known liquidity pools
```

### Detection Algorithm

```
Equal Highs/Lows:
  Parameters:
    range_percent: float = 0.01  (how close levels must be to count as "equal")

  For each swing high at level L:
    Count other swing highs where abs(other_level - L) / L < range_percent
    IF count >= 2: This is a LIQUIDITY zone at level L

  Same logic for swing lows.

Liquidity Sweep Detection:
  A sweep occurs when:
    1. Price's wick goes PAST a liquidity level
    2. Price's close comes BACK (doesn't close beyond the level)
    3. This indicates stop-losses were triggered but price reversed
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | enum | `EQH`, `EQL`, `SESSION_HIGH`, `SESSION_LOW`, `TRENDLINE` |
| `level` | float | Price level of the liquidity |
| `direction` | int | +1 (liquidity above = buy-side), -1 (liquidity below = sell-side) |
| `count` | int | Number of touches at this level |
| `status` | enum | `ACTIVE`, `SWEPT`, `BROKEN` |
| `swept_index` | int | Candle that swept the liquidity (if swept) |

### Role in Strategy

- **Liquidity Sweep** is the 2nd core confirmation in the IRS system
- Targets are often liquidity levels (equal highs/lows)
- Stop-loss placement considers where liquidity was already swept
- Liquidity sweep + CISD = strong reversal signal

### Implementation Notes

- `range_percent` parameter is critical -- too tight misses clusters, too loose creates noise
- Session highs/lows are straightforward: track previous day/week/month extremes
- The `smartmoneyconcepts` library provides `liquidity()` with `range_percent` parameter

---

## 7. Return to Origin (RTO)

### Definition

**RTO** is the price behavior of returning to the "origin" of an impulse move.
The origin is typically the FVG or IFVG that initiated the displacement/BOS.

### Detection Algorithm

```
After a BOS or significant displacement:
  1. Identify the ORIGIN:
     - The FVG that formed during the displacement
     - The IFVG if the move inverted a previous FVG
  2. Track whether price RETURNS to this origin zone
  3. RTO is confirmed when price touches the origin zone

RTO completion modes:
  "touch": Any wick into the origin zone
  "close": Candle close within/through the origin zone
  "ce":    Price reaches the midpoint of the origin zone
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `origin_type` | enum | `FVG`, `IFVG` |
| `origin_zone` | tuple | (top, bottom) of the origin zone |
| `displacement_index` | int | Index where the impulse move started |
| `status` | enum | `PENDING`, `COMPLETED`, `EXPIRED` |

### Role in Strategy

- RTO is the preferred entry method after aggressive exits from POIs
- Waiting for RTO provides: tighter stop, higher RR, fewer resweeps
- RTO on local POIs (15m/30m) = add-on opportunity
- The strategy explicitly says: "wait for at least a local RTO for FVG test"

### Implementation Notes

- Not every impulse move gets an RTO -- some moves are so strong they don't retrace
- Set an expiry (e.g., 50 candles) after which the pending RTO is considered expired
- RTO can be partial (price returns only to the edge of the zone) or full (to the origin)

---

## 8. Point of Interest (POI)

### Definition

**POI** is a composite concept -- a price zone where the trader expects a reaction.
It is NOT a standalone technical element but an **aggregation** of multiple SMC elements
at the same price area.

### Composition

A POI can be any of the following (or a combination):

```
- FVG zone (untested or partially tested)
- IFVG zone
- Liquidity level (EQH/EQL)
- Previous session/day/week high or low
```

### POI Strength Scoring

```
Base score per component:
  FVG from HTF (4H/1H):        +3
  FVG from LTF (30m/15m):      +1
  IFVG:                         +2
  Liquidity cluster (3+ touches): +2
  Session High/Low:              +1

Multipliers:
  Untested (fresh):             x1.5
  Tested once:                  x1.0
  Tested twice:                 x0.5
  Mitigated:                    x0 (remove from active POIs)

Confluence bonus:
  2 components overlap:         +2
  3+ components overlap:        +4
```

This scoring is a **starting point** for backtesting -- actual weights should be optimized.

### Implementation Notes

- Maintain a registry of all active POIs across all timeframes
- When new SMC elements form, check for overlap with existing POIs to create confluence
- POIs are invalidated when fully mitigated or when price closes through them cleanly

---

## 9. First Trouble Area (FTA)

### Definition

**FTA** is the first POI on the path between the current price and the target that could
block or reverse price movement.

### Detection Algorithm

```
Given:
  - Current position direction (long/short)
  - Target level
  - All active POIs between current price and target

FTA = The FIRST active POI in the path to target that is OPPOSING:
  For LONG position: First bearish POI above current price (potential resistance)
  For SHORT position: First bullish POI below current price (potential support)
```

### Role in Strategy

FTA handling is critical (see Strategy document section 5):
- Far FTA: Enter normally
- Close FTA: Wait for invalidation
- FTA validates: Move to BU immediately
- FTA invalidated: Continue to target

---

## 10. Premium & Discount Zones

### Definition

The market range between a significant swing high and swing low can be divided into zones:

```
Swing High ─────── 100% (Premium zone)
                    75%
Equilibrium ─────── 50%
                    25%
Swing Low  ─────── 0%  (Discount zone)

RULE: In a bullish trend, only buy in the DISCOUNT zone (below 50%)
RULE: In a bearish trend, only sell in the PREMIUM zone (above 50%)
```

### Detection Algorithm

```
Given swing_high and swing_low:
  equilibrium = (swing_high + swing_low) / 2
  premium_zone = [equilibrium, swing_high]
  discount_zone = [swing_low, equilibrium]

  For any price P:
    zone_pct = (P - swing_low) / (swing_high - swing_low) * 100
    is_premium = zone_pct > 50
    is_discount = zone_pct < 50
```

### Role in Strategy

- Used as a filter: don't buy in premium, don't sell in discount
- Helps validate POI quality: a POI in the "correct" zone is higher probability

---

## 11. Consequent Encroachment (CE / CVB)

### Definition

**Consequent Encroachment** (CE), also known as CVB in Russian-language trading,
is the **50% (midpoint)** of any FVG or price range.

### Detection Algorithm

```
For any zone with [top, bottom]:
  CE = (top + bottom) / 2
```

### Role in Strategy

- CE/CVB of an FVG = key reaction level
- "CVB test" is listed as a valid confirmation event and stop-loss placement zone
- Some traders consider FVG "mitigated" when CE is touched (not full fill)

---

## 12. Existing Python Libraries

### smartmoneyconcepts (PyPI)

**Repository:** https://github.com/joshyattridge/smart-money-concepts
**Install:** `pip install smartmoneyconcepts`
**Status:** BETA, actively maintained

**Available methods:**

```python
import smartmoneyconcepts as smc

# Input: pandas DataFrame with columns ["open", "high", "low", "close"]
# Optional: ["volume"] for volume-based indicators

smc.fvg(ohlc, join_consecutive=True)
# Returns: FVG (+1 bullish, -1 bearish), Top, Bottom, MitigatedIndex

smc.swing_highs_lows(ohlc, swing_length=5)
# Returns: HighLow (+1 swing high, -1 swing low), Level

smc.bos_choch(ohlc, swing_highs_lows, close_break=True)
# Returns: BOS, CHOCH, Level, BrokenIndex

smc.ob(ohlc, swing_highs_lows, close_mitigation=False)
# Returns: OB, Top, Bottom, OBVolume, Percentage

smc.liquidity(ohlc, swing_highs_lows, range_percent=0.01)
# Returns: Liquidity, Level, End, Swept

smc.previous_high_low(ohlc, time_frame="1D")
# Returns: PreviousHigh, PreviousLow, BrokenHigh, BrokenLow

smc.sessions(ohlc, session="London", start_time, end_time, time_zone="UTC")
# Returns: Active, High, Low

smc.retracements(ohlc, swing_highs_lows)
# Returns: Direction, CurrentRetracement%, DeepestRetracement%
```

### Evaluation for Our Project

**Pros:**
- Covers all core SMC elements (FVG, BOS/CHoCH, OB, Liquidity, Swings)
- Well-structured API with pandas DataFrame input/output
- Open source, can study and fork

**Cons:**
- Missing CISD detection
- Missing IFVG detection
- Missing RTO tracking
- Missing multi-timeframe coordination
- Missing real-time state management (designed for batch analysis)
- No confirmation counting or strategy logic

**Recommendation:**
Use as a **reference implementation** for individual concept algorithms.
Build our own system that:
1. Implements missing concepts (CISD, IFVG, RTO)
2. Adds lifecycle management (object state tracking)
3. Adds multi-timeframe coordination
4. Adds the strategy layer (confirmations, entries, exits)

We may use `smartmoneyconcepts` for initial validation -- run both our detection and theirs
on the same data and compare results to verify correctness.

---

## 13. Cross-Concept Relationships

```
Swing Highs/Lows
  ├── BOS/cBOS (requires swings)
  ├── Liquidity (equal highs/lows from swings)
  └── Premium/Discount (range between swings)

FVG (independent detection)
  ├── IFVG (inverted FVG)
  ├── CE/CVB (midpoint of FVG)
  └── RTO (return to FVG origin)

POI = Composite of: FVG + IFVG + Liquidity + Session levels
FTA = First opposing POI on the path to target

Strategy Confirmations = Events within POI:
  1. POI Tap
  2. Liquidity Sweep
  3. FVG Inversion
  4. Inversion Test
  5. Structure Break (BOS/cBOS/CISD)
  + Additional: FVG test, CVB test
```

This dependency graph determines the **build order** in the project:
first swings, then structure, then FVG, then liquidity, then POI, then strategy.
