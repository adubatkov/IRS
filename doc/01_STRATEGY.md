# IRS Strategy -- Intraday High Risk-Reward System

> Source: "IRS - как тянуть HIGH RR внутри дня" by Bikeeper
> Purpose: This document is the authoritative specification of the trading strategy.
> It serves as a prompt/reference for both the developer and Claude Code during implementation.

---

## 1. Strategy Philosophy

The strategy is **reactive, not predictive**. The trader does not build expectations or forecasts.
Instead, the system waits for a sufficient number of **confirmations (confirms)** to accumulate
inside a Point of Interest (POI), then enters a position with a tight stop-loss placed at a
structurally significant level. The goal is to achieve high Risk-to-Reward (RR) ratios by entering
after multiple confluences confirm the trade direction.

Key principle: **"Work with what the market gives in the moment."**

---

## 2. Timeframe Hierarchy

| Timeframe | Role | Purpose |
|-----------|------|---------|
| **1D (Daily)** | Context | Determine global direction, identify what candle is forming, where price is, where it's heading |
| **4H** | POI + Targets | Identify reversal POIs and major targets |
| **1H** | POI + Targets | Identify reversal POIs and targets (same role as 4H but more granular) |
| **30m** | Local/Support POI | Identify local or support POIs; can also serve as local targets |
| **15m** | Local/Support POI | Same as 30m; primary for local structure and support POIs |
| **1m** | Entry | Execute entries (trigger timeframe) |

### Context Determination Algorithm

```
1. Check Daily: What is the current candle doing? Where are we in the range? Global bias.
2. Check 4H/1H: Identify POIs for potential reversal. These are primary targets.
3. Check 30m/15m: Identify local/support POIs. These are secondary targets or add-on zones.
4. Drop to 1m: Execute entry when confirmations are met.
```

---

## 3. Confirmation System

Confirmations (confirms) are discrete events that occur **inside a POI** after price taps it.
The trader counts confirmations sequentially. The **minimum threshold is 5 confirmations**
before considering an entry. The exact sequence varies, but the following are the core events:

### 3.1 Core Confirmation Types

| # | Confirmation | Description |
|---|-------------|-------------|
| 1 | **POI Tap** | Price physically enters/touches the POI zone |
| 2 | **Liquidity Sweep (Снятие)** | Price sweeps liquidity (equal highs/lows, stop-loss clusters) |
| 3 | **FVG Inversion** | A Fair Value Gap that led to the sweep gets inverted (price closes through it) |
| 4 | **Inversion Test** | Price returns to test the inverted FVG zone |
| 5 | **Structure Break (Слом)** | Minor or swing-level break of structure (BOS/CISD) confirming directional change |

### 3.2 Additional Confirmation Types

These also count as valid confirmations:

- **FVG test with wick reaction** -- price touches FVG and reacts with a wick (even without full BOS)
- **CVB (Consequent Encroachment) test** -- test of the 50% level of an FVG
- **Additional cBOS** -- repeated continuation structural confirmations

### 3.3 Counting Rules

```
RULE 1: Minimum 5 confirmations before entry.
RULE 2: Each confirmation type can count once per occurrence.
RULE 3: An FVG wick reaction counts as a confirmation even without full BOS,
        IF 5+ confirmations were already collected before it.
RULE 4: Confirmations are counted PER POI interaction, not globally.
RULE 5: Personal calibration required -- backtest to find your comfortable threshold
        (some traders need 6, 7, or 8 confirms).
```

---

## 4. Entry Logic

### 4.1 Ideal Entry (Conservative)

```
CONDITIONS:
  1. Price taps a POI (4H/1H level)
  2. Inside the POI, 5+ confirmations are collected
  3. Price exits the POI structurally (BOS confirms direction)
  4. Structure is confirmed 1-2 additional times
  5. Stop-loss is placed where price interacted with liquidity
     (behind FVG, CVB, or inversion zone)

RESULT: High RR entry with structurally protected stop-loss.
```

### 4.2 When NOT to Enter (5th Confirm Trap)

```
SCENARIO:
  - Price exits POI aggressively with 4 confirmations
  - 5th confirmation is a structure update (new high/low)
  - BUT: No FVG test, no inversion test, no retracement occurred

ACTION: DO NOT ENTER on the 5th confirm.

REASON: Price will likely give an RTO (Return to Origin) to test the FVG/inversion zone.

WAIT FOR: 6th confirmation after the RTO pullback. This is a safer entry.
```

### 4.3 Aggressive Entry (When RTO is Expected)

```
SCENARIO:
  - Price exits POI very aggressively (large displacement candles)
  - 5+ confirms exist but price is moving fast

ACTION:
  Option A: Enter with immediate structural breakeven (short BU)
  Option B: Wait for local RTO to test FVG/inversion, then enter

PREFERENCE: Option B is safer. It provides:
  - Tighter stop-loss
  - Higher RR
  - Avoids catching retracements (resweeps)
```

---

## 5. FTA (First Trouble Area) Handling

FTA is the first opposing POI on the path to the target. It can block or reverse price.

### 5.1 FTA Decision Matrix

```
IF FTA is FAR from current price:
  -> Enter position normally
  -> Monitor FTA as price approaches

IF FTA is CLOSE (right in front of price):
  -> DO NOT enter until FTA is invalidated
  -> Invalidation = price closes through FTA zone
  -> After invalidation -> enter position

IF FTA VALIDATES (price reverses from FTA):
  -> Immediately move position to breakeven (BU)
  -> Original POI loses validity
  -> Price will likely move to next HTF POI

IF FTA is invalidated successfully:
  -> Continue holding position toward target
```

### 5.2 FTA Invalidation Criteria

FTA invalidation occurs when:
- Price closes through the FTA zone (body close, not just wick)
- The FTA zone is fully mitigated (price trades through it completely)

See separate video/documentation for detailed FTA invalidation rules.

---

## 6. Breakeven (BU) Rules

Three types of breakeven are used depending on context:

### 6.1 Structural BU (Short BU)

```
WHEN: Used after aggressive exits or when open POIs exist behind the position.
HOW:  Move stop-loss to breakeven at the FIRST structural confirmation after entry.
      (i.e., first new higher-low in longs, or first new lower-high in shorts)

LOGIC: If price moves structurally, it will continue structurally.
       If price gives a reverse BOS, it will RTO to a local POI -> add-on opportunity.
```

### 6.2 FTA-Based BU

```
WHEN: Price approaches an FTA.
HOW:  Move to breakeven BEFORE price reaches the FTA zone.

LOGIC: If FTA validates, the trade thesis may be invalid.
       BU protects capital while maintaining potential upside.
```

### 6.3 Range Boundary BU

```
WHEN: Trading inside a range.
HOW:  Place BU at the range boundary from which price exited.

LOGIC: Exit from range could be a stop-hunt (spring/upthrust).
       BU at boundary ensures minimal loss if price reverses back.
```

---

## 7. Add-On Strategy (Доборы)

Add-ons (additional position entries) are taken as price moves toward the target.

### 7.1 Local POI Add-Ons

```
SOURCE: 15m and 30m POIs that form ALONG THE PATH to the target.
METHOD:
  1. As price moves toward target, it forms local POIs (pullbacks on 15m/30m)
  2. When price returns to test a local POI, add position
  3. Use the local POI boundaries for the add-on stop-loss

TIMEFRAMES: Primarily 15m and 30m. Sometimes 1H if targeting a distant level.
```

### 7.2 Short BU Add-Ons

```
METHOD:
  1. Enter add-on position at a favorable level
  2. Immediately apply structural BU (short BU) after first structure confirmation
  3. If price reverses (BOS against), position closes at BU -> no loss
  4. If price continues, position adds to profit
```

---

## 8. Session Logic

The strategy does not formally trade sessions but observes session behavior:

### 8.1 Asian Session Flow

```
IF Asian session:
  - Moves cleanly and tests a logical POI
  - Sets up an "IRS" (entry setup) within the POI
  - BUT: This is NOT yet a valid entry trigger (too early)

THEN during Frankfurt/London session:
  - DO NOT wait for retests of Asian session POIs
  - Instead, work through ADD-ONS:
    * From local POIs on 15m/30m
    * Via structural short BU entries
  - Continue until a new local opposing POI forms
```

### 8.2 General Session Principle

The trader looks at **price logic**, not session labels. If price behaves cleanly during any
period and tests logical POIs, subsequent sessions can leverage that structure for add-on
entries without waiting for retests.

---

## 9. Range Trading

When price is ranging (consolidating), two approaches:

### 9.1 Breakout Approach

```
WAIT for price to exit the range in either direction.
THEN work on continuation from the breakout side.
```

### 9.2 Confirmation Count Approach

```
COUNT confirmations for BOTH directions (long and short) inside the range.
IF more confirms for long -> take long position
IF more confirms for short -> take short position

CRITICAL: Apply BU at range boundaries immediately.
REASON: Breakout may be a stop-hunt. BU protects if price reverses.
```

---

## 10. Position Flipping ("Переобувание")

Flipping means closing a position in one direction and opening in the opposite direction.

### 10.1 When to Flip

```
SCENARIO:
  1. Short position is open (from a reversal POI)
  2. Price approaches a SUPPORT POI (for long continuation)
  3. Support POI shows strong reaction:
     - Multiple structural confirmations
     - Clean interaction with liquidity
  4. Logical long targets exist above

ACTION:
  1. Gather structural confirmations at the support POI
  2. Observe liquidity interaction
  3. Close all short positions
  4. Open long positions

CONDITION: Only flip after sufficient confirmations at the support POI.
           Never flip on speculation alone.
```

### 10.2 When POI is Invalidated

```
SCENARIO:
  - Price arrives at POI "dirtily" (no clean reaction)
  - POI is invalidated (price closes through it)

ACTION:
  Option A: Move to higher timeframe to find additional POI above/below
  Option B: Work fractally -- behind every POI there are fractals or
            additional POIs on the same timeframe

TREATMENT:
  - Invalidated POI = compression zone
  - Look for the NEXT POI above/below for reversal
  - If invalidation creates a new local support POI -> can trade further
    in the original direction, as all intermediate POIs may be
    invalidated or rebalanced into support zones
```

---

## 11. RTO (Return to Origin) Expectations

### 11.1 During Entry

```
IF price has 5 confirmations BUT exits POI very aggressively:
  -> Do NOT chase
  -> Wait for RTO to test:
    * FVG zone
    * Inversion zone
    * Any significant structure level
  -> After RTO: collect additional confirmations
  -> THEN enter with tighter stop and better RR
```

### 11.2 During Movement

```
IF price forms a local POI (15m/30m) during movement to target:
  -> Expect potential RTO to this local POI
  -> This RTO is an add-on opportunity
  -> Place structural BU on existing position
```

---

## 12. Resweep (Ресип) Handling

```
IF price resweeps (re-enters the POI after initial exit):
  POSITIVE interpretation:
    1. It's an ADDITIONAL confirmation
    2. It allows placing a MORE PROTECTED stop-loss
    3. It creates opportunity to add position at better levels
    4. Can help offset BU or stop-loss from prior entry

ACTION: Treat resweep as opportunity, not threat.
```

---

## 13. Timeframe Desynchronization

When Higher Timeframe (HTF) and Lower Timeframe (LTF) point in opposite directions:

### 13.1 Desync Trading Rules

```
PRINCIPLE: Trade what price gives in the moment.

IF 15m draws a long FVG and prior fractal was swept:
  -> Work locally toward the nearest 15m/30m fractal
  -> Use reduced position size
  -> Use shorter targets (local fractals only)

REASON: Unknown which fractal or FTA will become
        the reversal point for HTF synchronization.

PREFERENCE: Trading in sync with HTF is always better.
           Desync trades are acceptable but with reduced risk.
```

### 13.2 Sync/Desync Decision Matrix

```
HTF = Bullish, LTF = Bullish:
  -> Full position size, distant targets, high conviction

HTF = Bullish, LTF = Bearish (Desync):
  -> Reduced position, local targets only
  -> OR wait for LTF to re-synchronize with HTF

HTF = Bearish, LTF = Bearish:
  -> Full position size, distant targets, high conviction

HTF = Bearish, LTF = Bullish (Desync):
  -> Reduced position, local targets only
  -> OR wait for LTF to re-synchronize with HTF
```

---

## 14. Target Selection

### 14.1 Target Hierarchy

```
PRIORITY 1 (Primary Target):
  POI on 4H/1H from which reversal is expected.
  This can be: FVG, Liquidity zone.

PRIORITY 2 (Secondary Target):
  Liquidity sweep level (equal highs/lows, swing on the path).

PRIORITY 3 (Local Target -- used in desync):
  Nearest 15m/30m fractal in the movement direction.
```

### 14.2 Target Extension

```
IF primary target is reached via sweep (wick through):
  AND additional local POIs exist beyond the target:
  AND there are logical targets further away:
    -> Can continue working in the same direction

IF no logical targets exist beyond:
  -> Can work toward the next swing low/high (перелой/перехай)
  -> But with reduced conviction
```

### 14.3 Post-Target Behavior

```
AFTER target is reached:
  1. Look for reversal setup at the target POI
  2. If reversal POI gives confirmations -> flip position
  3. If no clear setup -> close and wait
```

---

## 15. Risk Management Summary

| Aspect | Rule |
|--------|------|
| **Minimum confirmations** | 5 before entry |
| **Stop-loss placement** | Behind liquidity interaction point (FVG, CVB, inversion zone) |
| **Breakeven** | Structural (first BOS), FTA-based, or range boundary |
| **Add-ons** | From local 15m/30m POIs with structural BU |
| **Position sizing (sync)** | Full size when HTF and LTF agree |
| **Position sizing (desync)** | Reduced size when HTF and LTF disagree |
| **Flipping** | Only after sufficient confirmations at opposing POI |
| **RTO** | Always preferred over chasing aggressive moves |
| **Resweeps** | Treated as additional confirmations and entry opportunities |

---

## 16. Implementation Notes for Backtesting

### 16.1 State Machine

The strategy operates as a **finite state machine** per POI interaction:

```
States:
  IDLE           -> Waiting for price to approach a POI
  POI_TAPPED     -> Price has entered a POI zone
  COLLECTING     -> Counting confirmations inside/around the POI
  READY          -> Minimum confirmations met, waiting for entry trigger
  POSITIONED     -> Position is open
  MANAGING       -> Managing position (BU, add-ons, FTA monitoring)
  CLOSED         -> Position closed (target hit, BU hit, or flipped)

Transitions:
  IDLE -> POI_TAPPED:        Price enters a tracked POI zone
  POI_TAPPED -> COLLECTING:  First confirmation event detected
  COLLECTING -> READY:       Confirmation count >= threshold
  READY -> POSITIONED:       Entry trigger conditions met
  POSITIONED -> MANAGING:    First BU or add-on action
  MANAGING -> CLOSED:        Target hit / BU hit / manual close
  MANAGING -> POSITIONED:    Flip (close and reopen in opposite direction)
  ANY -> IDLE:               POI invalidated without entry
```

### 16.2 Multi-Timeframe Requirement

The backtester MUST maintain state across all timeframes simultaneously:
- 1D state for context/bias
- 4H/1H states for POI tracking and targets
- 30m/15m states for local POIs and add-on zones
- 1m state for entry execution

All states must be synchronized by timestamp.

### 16.3 Parameterization

The following parameters should be configurable for optimization:

```yaml
confirmations:
  min_count: 5           # Minimum confirmations for entry
  max_count: 8           # Maximum (stop collecting and enter)

fractals:
  swing_length: 5        # Candles on each side for swing detection
  swing_length_htf: 10   # For higher timeframes

fvg:
  min_gap_pct: 0.001     # Minimum gap size as % of price
  mitigation_mode: "close"  # "close" or "wick" for mitigation detection

structure:
  break_mode: "close"    # "close" (CISD) or "wick" (BOS) for structure breaks

risk:
  position_size_sync: 1.0    # Full size when in sync
  position_size_desync: 0.5  # Reduced size when desynced

targets:
  primary_tf: ["4H", "1H"]
  local_tf: ["30m", "15m"]
```
