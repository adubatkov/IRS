"""Timeframe resampling for OHLC data.

Resamples 1m candle data to higher timeframes (5m, 15m, 30m, 1H, 4H, 1D).
Supports caching resampled data as parquet files.
"""

import logging
from pathlib import Path

import pandas as pd

from data.loader import file_hash

logger = logging.getLogger(__name__)

# Mapping from our TF names to pandas offset aliases
TF_TO_PANDAS_FREQ: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1D",
}

OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "tick_volume": "sum",
}


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1m OHLC data to a higher timeframe.

    Args:
        df: DataFrame with 'time' column and OHLC columns.
        timeframe: Target timeframe (e.g., '5m', '15m', '1H', '4H', '1D').

    Returns:
        Resampled DataFrame with the same column structure.
    """
    if timeframe == "1m":
        return df.copy()

    freq = TF_TO_PANDAS_FREQ.get(timeframe)
    if freq is None:
        raise ValueError(
            f"Unknown timeframe: {timeframe}. "
            f"Supported: {list(TF_TO_PANDAS_FREQ.keys())}"
        )

    # Set time as index for resampling
    df_indexed = df.set_index("time")

    # Determine which columns to aggregate
    agg_dict = {k: v for k, v in OHLC_AGG.items() if k in df_indexed.columns}

    resampled = df_indexed.resample(freq).agg(agg_dict).dropna(subset=["open"])  # type: ignore[arg-type]
    resampled = resampled.reset_index()

    logger.info(
        "Resampled %d rows to %s: %d candles",
        len(df), timeframe, len(resampled),
    )
    return resampled


def resample_all(
    df: pd.DataFrame,
    timeframes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Resample 1m data to all specified timeframes.

    Args:
        df: 1m OHLC DataFrame.
        timeframes: List of target timeframes. Defaults to all supported.

    Returns:
        Dict mapping timeframe string to resampled DataFrame.
    """
    if timeframes is None:
        timeframes = list(TF_TO_PANDAS_FREQ.keys())

    result = {}
    for tf in timeframes:
        result[tf] = resample(df, tf)

    return result


def save_resampled(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_dir: str | Path = "data/processed",
) -> Path:
    """Save a resampled DataFrame as parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %s %s to %s (%d rows)", symbol, timeframe, path, len(df))
    return path


def load_or_resample(
    source_path: str | Path,
    symbol: str,
    timeframe: str,
    df_1m: pd.DataFrame,
    cache_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Load cached resampled data or resample from 1m and cache.

    Uses file hash of source parquet to invalidate cache.
    """
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{symbol}_{timeframe}.parquet"
    hash_file = cache_dir / f"{symbol}_{timeframe}.hash"

    source_path = Path(source_path)
    current_hash = file_hash(source_path) if source_path.exists() else ""

    # Check cache validity
    if cache_file.exists() and hash_file.exists():
        stored_hash = hash_file.read_text().strip()
        if stored_hash == current_hash:
            logger.info("Loading cached %s %s from %s", symbol, timeframe, cache_file)
            return pd.read_parquet(cache_file)

    # Resample and cache
    resampled = resample(df_1m, timeframe)
    save_resampled(resampled, symbol, timeframe, cache_dir)

    # Save hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(current_hash)

    return resampled
