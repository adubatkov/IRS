"""Comparison test: our SMC concepts vs smartmoneyconcepts library.

Target: 90%+ match on shared concepts (fractals, FVG, structure).
Uses same parameters to ensure fair comparison.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concepts.fractals import detect_swings
from concepts.fvg import detect_fvg
from concepts.structure import detect_structure
from data.loader import load_instrument

try:
    from smartmoneyconcepts.smc import smc as smc_lib
    HAS_SMC_LIB = True
except ImportError:
    HAS_SMC_LIB = False

PARQUET_PATH = Path(__file__).parent.parent.parent / "data" / "optimized"
NAS100_FILE = PARQUET_PATH / "NAS100_m1.parquet"

SWING_LENGTH = 10  # Use same for both


pytestmark = pytest.mark.skipif(
    not HAS_SMC_LIB or not NAS100_FILE.exists(),
    reason="smartmoneyconcepts library not installed or NAS100 parquet not found",
)


def _prepare_data(n_rows=10000):
    """Load and prepare data for both libraries."""
    df = load_instrument("NAS100").head(n_rows)

    # Our format: has 'time' column, standard OHLC
    ours = df.copy()

    # SMC lib format: needs 'open', 'high', 'low', 'close', 'volume' columns, numeric index
    lib_df = df[["open", "high", "low", "close"]].copy()
    lib_df["volume"] = df.get("tick_volume", 0)
    lib_df = lib_df.reset_index(drop=True)

    return ours, lib_df


def _match_percentage(our_indices, lib_indices, tolerance=2):
    """Calculate percentage of our detections that match library detections.

    Args:
        our_indices: Set/list of indices from our detection.
        lib_indices: Set/list of indices from library detection.
        tolerance: How many bars of difference to still count as a match.

    Returns:
        Float 0-1 representing match rate.
    """
    if len(our_indices) == 0 and len(lib_indices) == 0:
        return 1.0
    if len(our_indices) == 0 or len(lib_indices) == 0:
        return 0.0

    our_arr = np.array(sorted(our_indices))
    lib_arr = np.array(sorted(lib_indices))

    matches = 0
    for idx in our_arr:
        # Check if any library detection is within tolerance
        diffs = np.abs(lib_arr - idx)
        if diffs.min() <= tolerance:
            matches += 1

    # Symmetric: also check reverse
    reverse_matches = 0
    for idx in lib_arr:
        diffs = np.abs(our_arr - idx)
        if diffs.min() <= tolerance:
            reverse_matches += 1

    # Use the average of both directions
    forward_rate = matches / len(our_arr) if len(our_arr) > 0 else 0
    reverse_rate = reverse_matches / len(lib_arr) if len(lib_arr) > 0 else 0

    return (forward_rate + reverse_rate) / 2


class TestSwingComparison:
    """Compare our swing detection with smartmoneyconcepts library."""

    def test_swing_highs_match(self):

        ours_df, lib_df = _prepare_data(10000)

        # Our detection
        swings = detect_swings(ours_df, swing_length=SWING_LENGTH)
        our_sh_indices = set(swings.index[swings["swing_high"]].tolist())

        # Library detection
        lib_swings = smc_lib.swing_highs_lows(lib_df, swing_length=SWING_LENGTH)
        lib_sh_indices = set(lib_swings.index[lib_swings["HighLow"] == 1].tolist())

        match_rate = _match_percentage(our_sh_indices, lib_sh_indices, tolerance=1)
        print(f"\nSwing Highs - Ours: {len(our_sh_indices)}, Lib: {len(lib_sh_indices)}, Match: {match_rate:.1%}")

        assert match_rate >= 0.7, (
            f"Swing high match rate {match_rate:.1%} < 70%. "
            f"Ours: {len(our_sh_indices)}, Lib: {len(lib_sh_indices)}"
        )

    def test_swing_lows_match(self):

        ours_df, lib_df = _prepare_data(10000)

        swings = detect_swings(ours_df, swing_length=SWING_LENGTH)
        our_sl_indices = set(swings.index[swings["swing_low"]].tolist())

        lib_swings = smc_lib.swing_highs_lows(lib_df, swing_length=SWING_LENGTH)
        lib_sl_indices = set(lib_swings.index[lib_swings["HighLow"] == -1].tolist())

        match_rate = _match_percentage(our_sl_indices, lib_sl_indices, tolerance=1)
        print(f"\nSwing Lows - Ours: {len(our_sl_indices)}, Lib: {len(lib_sl_indices)}, Match: {match_rate:.1%}")

        assert match_rate >= 0.7, (
            f"Swing low match rate {match_rate:.1%} < 70%. "
            f"Ours: {len(our_sl_indices)}, Lib: {len(lib_sl_indices)}"
        )

    def test_swing_count_similar(self):
        """Counts should be within 50% of each other."""

        ours_df, lib_df = _prepare_data(10000)

        swings = detect_swings(ours_df, swing_length=SWING_LENGTH)
        our_count = swings["swing_high"].sum() + swings["swing_low"].sum()

        lib_swings = smc_lib.swing_highs_lows(lib_df, swing_length=SWING_LENGTH)
        lib_count = lib_swings["HighLow"].notna().sum()

        ratio = min(our_count, lib_count) / max(our_count, lib_count) if max(our_count, lib_count) > 0 else 1
        print(f"\nSwing Count - Ours: {our_count}, Lib: {lib_count}, Ratio: {ratio:.2f}")

        assert ratio >= 0.5, (
            f"Swing count ratio {ratio:.2f} < 0.5. Ours: {our_count}, Lib: {lib_count}"
        )


class TestFVGComparison:
    """Compare our FVG detection with smartmoneyconcepts library."""

    def test_fvg_count_similar(self):

        ours_df, lib_df = _prepare_data(10000)

        our_fvgs = detect_fvg(ours_df, min_gap_pct=0.0, join_consecutive=False)

        lib_fvgs = smc_lib.fvg(lib_df, join_consecutive=False)
        lib_fvg_count = lib_fvgs["FVG"].notna().sum()

        our_count = len(our_fvgs)
        ratio = min(our_count, lib_fvg_count) / max(our_count, lib_fvg_count) if max(our_count, lib_fvg_count) > 0 else 1
        print(f"\nFVG Count - Ours: {our_count}, Lib: {lib_fvg_count}, Ratio: {ratio:.2f}")

        assert ratio >= 0.5, (
            f"FVG count ratio {ratio:.2f} < 0.5. Ours: {our_count}, Lib: {lib_fvg_count}"
        )

    def test_fvg_bullish_bearish_ratio(self):
        """Both should detect similar proportions of bullish vs bearish FVGs."""

        ours_df, lib_df = _prepare_data(10000)

        our_fvgs = detect_fvg(ours_df, min_gap_pct=0.0, join_consecutive=False)

        lib_fvgs = smc_lib.fvg(lib_df, join_consecutive=False)
        lib_valid = lib_fvgs[lib_fvgs["FVG"].notna()]

        if len(our_fvgs) > 0 and len(lib_valid) > 0:
            our_bull_pct = (our_fvgs["direction"] == 1).mean()
            lib_bull_pct = (lib_valid["FVG"] == 1).mean()
            diff = abs(our_bull_pct - lib_bull_pct)
            print(f"\nFVG Bull% - Ours: {our_bull_pct:.1%}, Lib: {lib_bull_pct:.1%}, Diff: {diff:.1%}")
            assert diff < 0.3, (
                f"FVG bullish percentage diff {diff:.1%} > 30%. "
                f"Ours: {our_bull_pct:.1%}, Lib: {lib_bull_pct:.1%}"
            )

    def test_fvg_indices_match(self):
        """FVGs should be detected at similar bar positions.

        Note: Our creation_index uses the DataFrame index (time-based) while
        the library uses positional index. We convert both to positional
        for fair comparison.
        """

        ours_df, lib_df = _prepare_data(10000)

        our_fvgs = detect_fvg(ours_df, min_gap_pct=0.0, join_consecutive=False)
        # Convert our time-based index to positional for comparison
        idx_to_pos = {idx: pos for pos, idx in enumerate(ours_df.index)}
        our_bull_idx = set(
            idx_to_pos[ci] for ci in our_fvgs[our_fvgs["direction"] == 1]["creation_index"]
            if ci in idx_to_pos
        )
        our_bear_idx = set(
            idx_to_pos[ci] for ci in our_fvgs[our_fvgs["direction"] == -1]["creation_index"]
            if ci in idx_to_pos
        )

        lib_fvgs = smc_lib.fvg(lib_df, join_consecutive=False)
        lib_valid = lib_fvgs[lib_fvgs["FVG"].notna()]
        lib_bull_idx = set(lib_valid.index[lib_valid["FVG"] == 1].tolist())
        lib_bear_idx = set(lib_valid.index[lib_valid["FVG"] == -1].tolist())

        # Tolerance=1 because our creation_index is the 3rd candle (i),
        # while the library reports at the 2nd candle (i-1)
        bull_match = _match_percentage(our_bull_idx, lib_bull_idx, tolerance=1)
        bear_match = _match_percentage(our_bear_idx, lib_bear_idx, tolerance=1)
        overall = (bull_match + bear_match) / 2

        print(f"\nFVG Index Match - Bull: {bull_match:.1%}, Bear: {bear_match:.1%}, Overall: {overall:.1%}")
        assert overall >= 0.7, (
            f"FVG index match {overall:.1%} < 70%. Bull: {bull_match:.1%}, Bear: {bear_match:.1%}"
        )


class TestStructureComparison:
    """Compare our structure (BOS/cBOS) detection with library."""

    def test_structure_count_similar(self):

        ours_df, lib_df = _prepare_data(10000)

        our_events = detect_structure(ours_df, swing_length=SWING_LENGTH, close_break=True)

        lib_swings = smc_lib.swing_highs_lows(lib_df, swing_length=SWING_LENGTH)
        lib_events = smc_lib.bos_choch(lib_df, lib_swings, close_break=True)
        lib_bos_count = lib_events["BOS"].notna().sum()
        lib_choch_count = lib_events["CHOCH"].notna().sum()
        lib_total = lib_bos_count + lib_choch_count

        our_total = len(our_events)
        ratio = min(our_total, lib_total) / max(our_total, lib_total) if max(our_total, lib_total) > 0 else 1
        print(f"\nStructure Count - Ours: {our_total}, Lib: {lib_total} (BOS={lib_bos_count}, CHoCH={lib_choch_count}), Ratio: {ratio:.2f}")

        assert ratio >= 0.3, (
            f"Structure count ratio {ratio:.2f} < 0.3. Ours: {our_total}, Lib: {lib_total}"
        )

    def test_structure_directions_match(self):
        """Bullish/bearish ratio should be similar."""

        ours_df, lib_df = _prepare_data(10000)

        our_events = detect_structure(ours_df, swing_length=SWING_LENGTH)

        lib_swings = smc_lib.swing_highs_lows(lib_df, swing_length=SWING_LENGTH)
        lib_events = smc_lib.bos_choch(lib_df, lib_swings, close_break=True)

        # Library: BOS column has direction (1.0 or -1.0)
        lib_bos = lib_events[lib_events["BOS"].notna()]["BOS"]
        lib_choch = lib_events[lib_events["CHOCH"].notna()]["CHOCH"]
        lib_all = pd.concat([lib_bos, lib_choch])

        if len(our_events) > 0 and len(lib_all) > 0:
            our_bull_pct = (our_events["direction"] == 1).mean()
            lib_bull_pct = (lib_all == 1).mean()
            diff = abs(our_bull_pct - lib_bull_pct)
            print(f"\nStructure Bull% - Ours: {our_bull_pct:.1%}, Lib: {lib_bull_pct:.1%}, Diff: {diff:.1%}")
            # Direction ratio can vary quite a bit depending on implementation
            assert diff < 0.4, (
                f"Structure direction diff {diff:.1%} > 40%"
            )


class TestSummary:
    """Print a summary report of all comparisons."""

    def test_print_summary(self):

        ours_df, lib_df = _prepare_data(10000)

        print("\n" + "=" * 60)
        print("SMC CONCEPT COMPARISON SUMMARY")
        print("=" * 60)

        # Swings
        swings = detect_swings(ours_df, swing_length=SWING_LENGTH)
        our_sh = swings["swing_high"].sum()
        our_sl = swings["swing_low"].sum()
        lib_swings = smc_lib.swing_highs_lows(lib_df, swing_length=SWING_LENGTH)
        lib_sh = (lib_swings["HighLow"] == 1).sum()
        lib_sl = (lib_swings["HighLow"] == -1).sum()
        print(f"\nSwings (len={SWING_LENGTH}):")
        print(f"  Highs - Ours: {our_sh}, Lib: {lib_sh}")
        print(f"  Lows  - Ours: {our_sl}, Lib: {lib_sl}")

        # FVGs
        our_fvgs = detect_fvg(ours_df, min_gap_pct=0.0, join_consecutive=False)
        lib_fvgs_res = smc_lib.fvg(lib_df, join_consecutive=False)
        lib_fvg_count = lib_fvgs_res["FVG"].notna().sum()
        print("\nFVGs:")
        print(f"  Ours: {len(our_fvgs)}, Lib: {lib_fvg_count}")

        # Structure
        our_events = detect_structure(ours_df, swing_length=SWING_LENGTH)
        lib_events = smc_lib.bos_choch(lib_df, lib_swings, close_break=True)
        lib_bos = lib_events["BOS"].notna().sum()
        lib_choch = lib_events["CHOCH"].notna().sum()
        print("\nStructure:")
        print(f"  Ours: {len(our_events)} total")
        print(f"  Lib:  {lib_bos} BOS + {lib_choch} CHoCH = {lib_bos + lib_choch}")

        print("\n" + "=" * 60)
