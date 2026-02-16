"""Data loading utilities for the IRS backtesting system.

Loads 1m OHLC candle data from parquet (primary) or split CSV files (fallback).
Returns clean DataFrames with UTC timestamps.
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"time", "open", "high", "low", "close"}
OPTIONAL_COLUMNS = {"tick_volume", "volume"}
DROP_COLUMNS = {"Shapes"}  # Extra columns from TradingView export


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a single parquet file and return a clean OHLC DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    table = pq.read_table(path)
    df = table.to_pandas()
    return _clean_dataframe(df, source=str(path))


def load_csv_directory(directory: str | Path) -> pd.DataFrame:
    """Load and merge all CSV files from a directory (split TradingView exports).

    Files are sorted by name to maintain chronological order after merge,
    then de-duplicated by timestamp.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Drop TradingView extra columns
        cols_to_drop = [c for c in df.columns if c in DROP_COLUMNS]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    return _clean_dataframe(merged, source=str(directory))


def load_instrument(
    symbol: str,
    optimized_path: str | Path = "data/optimized",
    parquet_filename: str | None = None,
) -> pd.DataFrame:
    """Load 1m data for an instrument.

    Tries optimized parquet first, falls back to raw CSV directory.
    """
    optimized_path = Path(optimized_path)

    # Try parquet first
    if parquet_filename:
        parquet_file = optimized_path / parquet_filename
    else:
        parquet_file = optimized_path / f"{symbol}_m1.parquet"

    if parquet_file.exists():
        logger.info("Loading %s from parquet: %s", symbol, parquet_file)
        return load_parquet(parquet_file)

    raise FileNotFoundError(
        f"No data source found for {symbol}. "
        f"Expected parquet at {parquet_file}"
    )


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Validate an OHLC DataFrame and return list of issues found."""
    issues = []

    # Check required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")

    if "time" not in df.columns:
        return issues

    # Check time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        issues.append("'time' column is not datetime type")

    # Check for duplicates
    n_dupes = df["time"].duplicated().sum()
    if n_dupes > 0:
        issues.append(f"Found {n_dupes} duplicate timestamps")

    # Check OHLC consistency
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        bad_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
        bad_low = (df["low"] > df["open"]) | (df["low"] > df["close"])
        n_bad = bad_high.sum() + bad_low.sum()
        if n_bad > 0:
            issues.append(f"Found {n_bad} candles with OHLC inconsistency")

    # Check for NaN
    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if ohlc_cols:
        n_nan = df[ohlc_cols].isna().sum().sum()
        if n_nan > 0:
            issues.append(f"Found {n_nan} NaN values in OHLC columns")

    return issues


def detect_gaps(df: pd.DataFrame, expected_freq: str = "1min") -> pd.DataFrame:
    """Detect gaps in 1m data. Returns DataFrame of gap start/end times and duration."""
    if "time" not in df.columns or len(df) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_minutes"])

    time_diff = df["time"].diff()
    expected = pd.Timedelta(expected_freq)
    # Allow up to 2x expected frequency before calling it a gap
    gap_mask = time_diff > (expected * 2)
    gap_indices = df.index[gap_mask]

    gaps = []
    for idx in gap_indices:
        pos = df.index.get_loc(idx)
        if isinstance(pos, int) and pos > 0:
            prev_idx = df.index[pos - 1]
            gap_start = df.loc[prev_idx, "time"]
            gap_end = df.loc[idx, "time"]
            gap_minutes = (gap_end - gap_start).total_seconds() / 60
            gaps.append({
                "gap_start": gap_start,
                "gap_end": gap_end,
                "gap_minutes": gap_minutes,
            })

    return pd.DataFrame(gaps)


def get_data_stats(df: pd.DataFrame) -> dict:
    """Get summary statistics for an OHLC DataFrame."""
    stats = {
        "rows": len(df),
        "columns": list(df.columns),
    }
    if "time" in df.columns and len(df) > 0:
        stats["start"] = str(df["time"].min())
        stats["end"] = str(df["time"].max())
        stats["duration_days"] = (df["time"].max() - df["time"].min()).days
    if "close" in df.columns and len(df) > 0:
        stats["price_min"] = float(df["close"].min())
        stats["price_max"] = float(df["close"].max())
        stats["price_mean"] = float(df["close"].mean())
    return stats


def file_hash(path: str | Path) -> str:
    """Compute a hash of file path + modification time for cache invalidation."""
    path = Path(path)
    stat = path.stat()
    key = f"{path.resolve()}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(key.encode()).hexdigest()


def _clean_dataframe(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Standardize an OHLC DataFrame."""
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop known junk columns
    cols_to_drop = [c for c in df.columns if c in {s.lower() for s in DROP_COLUMNS}]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Parse time column
    if "time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], utc=True)
        elif df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")

    # Ensure tick_volume exists
    if "tick_volume" not in df.columns:
        if "volume" in df.columns:
            df = df.rename(columns={"volume": "tick_volume"})
        else:
            df["tick_volume"] = 0

    # Keep only expected columns
    keep_cols = ["time", "open", "high", "low", "close", "tick_volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Sort by time, drop duplicates
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    logger.info("Loaded %d rows from %s", len(df), source)
    return df
