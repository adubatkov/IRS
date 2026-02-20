"""Global configuration loader for the IRS backtesting system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class FractalsConfig:
    swing_length: dict[str, int] = field(default_factory=lambda: {
        "1m": 3, "5m": 5, "15m": 5, "30m": 5, "1H": 7, "4H": 10, "1D": 10
    })


@dataclass
class StructureConfig:
    break_mode: str = "close"
    min_displacement: float = 0.001


@dataclass
class FVGConfig:
    min_gap_pct: float = 0.0005
    join_consecutive: bool = True
    mitigation_mode: str = "close"


@dataclass
class LiquidityConfig:
    range_percent: float = 0.001
    min_touches: int = 2


@dataclass
class ConceptsConfig:
    fractals: FractalsConfig = field(default_factory=FractalsConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    fvg: FVGConfig = field(default_factory=FVGConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)


@dataclass
class ConfirmationsConfig:
    min_count: int = 5
    max_count: int = 8


@dataclass
class EntryConfig:
    mode: str = "conservative"
    rto_wait: bool = True


@dataclass
class BreakevenConfig:
    structural_bu: bool = True
    fta_bu: bool = True
    range_bu: bool = True


@dataclass
class RiskConfig:
    position_size_sync: float = 1.0
    position_size_desync: float = 0.5
    max_risk_per_trade: float = 0.02
    max_concurrent_positions: int = 3


@dataclass
class TargetsConfig:
    primary_tf: list[str] = field(default_factory=lambda: ["4H", "1H"])
    local_tf: list[str] = field(default_factory=lambda: ["30m", "15m"])


@dataclass
class FTAConfig:
    close_threshold_pct: float = 0.3
    invalidation_mode: str = "close"


@dataclass
class StrategyConfig:
    confirmations: ConfirmationsConfig = field(default_factory=ConfirmationsConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    breakeven: BreakevenConfig = field(default_factory=BreakevenConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    fta: FTAConfig = field(default_factory=FTAConfig)


@dataclass
class InstrumentConfig:
    file: str = ""
    source: str = ""


@dataclass
class DataConfig:
    symbol: str = "NAS100"
    optimized_path: str = "data/optimized/"
    processed_path: str = "data/processed/"
    timeframes: list[str] = field(default_factory=lambda: [
        "1m", "5m", "15m", "30m", "1H", "4H", "1D"
    ])
    instruments: dict[str, InstrumentConfig] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 10000
    commission_pct: float = 0.0006
    slippage_pct: float = 0.0002


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    concepts: ConceptsConfig = field(default_factory=ConceptsConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _build_nested(cls: type, raw: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict."""
    if not isinstance(raw, dict):
        return raw
    dc_fields = getattr(cls, "__dataclass_fields__", {})
    # Use typing.get_type_hints to safely resolve string annotations
    try:
        resolved_hints = get_type_hints(cls)
    except Exception:
        resolved_hints = {}
    kwargs: dict[str, Any] = {}
    for key, val in raw.items():
        if key in dc_fields:
            field_type = resolved_hints.get(key, dc_fields[key].type)
            # Skip unresolved string annotations
            if isinstance(field_type, str):
                kwargs[key] = val
                continue
            origin = getattr(field_type, "__origin__", None)
            if hasattr(field_type, "__dataclass_fields__") and isinstance(val, dict):
                kwargs[key] = _build_nested(field_type, val)
            elif origin is dict and isinstance(val, dict):
                args = getattr(field_type, "__args__", None)
                if args and len(args) == 2 and hasattr(args[1], "__dataclass_fields__"):
                    kwargs[key] = {
                        k: _build_nested(args[1], v) for k, v in val.items()
                    }
                else:
                    kwargs[key] = val
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        return Config()
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not raw:
        return Config()
    return _build_nested(Config, raw)
