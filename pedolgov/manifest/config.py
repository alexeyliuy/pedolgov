from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import msgspec
import yaml


class RetrievalConfig(msgspec.Struct, frozen=True):
    chunk_size: int = 512
    overlap: int = 64
    top_k: int = 10
    encoder: str = "hash"
    dim: int = 256


class SignalConfig(msgspec.Struct, frozen=True):
    prr_min: float = 2.0
    chi2_min: float = 4.0
    ror_ci_min: float = 1.0
    consensus_min: int = 3


class InferenceConfig(msgspec.Struct, frozen=True):
    backbone: str = "offline"
    temperature_extract: float = 0.1
    temperature_summary: float = 0.3
    max_input_tokens: int = 4096
    max_output_tokens: int = 2048


class RiskConfig(msgspec.Struct, frozen=True):
    n_factors: int = 8
    weight_cap: float = 0.25
    thresholds: tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)


class FitConfig(msgspec.Struct, frozen=True):
    learning_rate: float = 0.05
    steps: int = 300
    weight_decay: float = 0.0
    kappa_grid: int = 41


class ConformalConfig(msgspec.Struct, frozen=True):
    alpha: float = 0.10
    set_size_target: int = 3
    calibration_sizes: tuple[int, ...] = (50, 100, 200, 500)
    ood_percentile: float = 95.0


class GovernanceConfig(msgspec.Struct, frozen=True):
    large_set_threshold: int = 4
    hitl_on_critical: bool = True
    hitl_on_large_set: bool = True
    hitl_on_hard_guardrail: bool = True
    hitl_on_neonate: bool = True


class BenchConfig(msgspec.Struct, frozen=True):
    n_items: int = 1847
    n_drugs: int = 187
    n_indications: int = 94
    n_classes: int = 15
    train_frac: float = 0.4
    calibration_frac: float = 0.2
    test_frac: float = 0.4


class Ablation(msgspec.Struct, frozen=True):
    use_rag: bool = True
    use_multi_source: bool = True
    use_conformal: bool = True
    use_governance: bool = True
    use_grading: bool = True
    use_kg: bool = True
    shuffle_inputs: bool = False


class Config(msgspec.Struct, frozen=True):
    name: str = "main"
    seeds: tuple[int, ...] = (42, 123, 456, 789, 1024)
    retrieval: RetrievalConfig = msgspec.field(default_factory=RetrievalConfig)
    signal: SignalConfig = msgspec.field(default_factory=SignalConfig)
    inference: InferenceConfig = msgspec.field(default_factory=InferenceConfig)
    risk: RiskConfig = msgspec.field(default_factory=RiskConfig)
    fit: FitConfig = msgspec.field(default_factory=FitConfig)
    conformal: ConformalConfig = msgspec.field(default_factory=ConformalConfig)
    governance: GovernanceConfig = msgspec.field(default_factory=GovernanceConfig)
    bench: BenchConfig = msgspec.field(default_factory=BenchConfig)
    ablation: Ablation = msgspec.field(default_factory=Ablation)


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_path(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def load_config(path: str | os.PathLike[str], overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_path(raw, key.strip(), _coerce(value.strip()))
    return msgspec.convert(raw, Config)


def to_builtins(config: Config) -> dict[str, Any]:
    result = msgspec.to_builtins(config)
    if not isinstance(result, dict):
        raise TypeError("configuration did not serialise to a mapping")
    return result
