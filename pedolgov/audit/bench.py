from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pedolgov.lanes.contracts import (
    Abstract,
    Contingency,
    KgNeighborhood,
    LabelSections,
    RawSources,
)
from pedolgov.manifest.config import BenchConfig
from pedolgov.manifest.declaration import AgeGroup, Query, RiskTier


@dataclass(frozen=True)
class BenchItem:
    query: Query
    therapeutic_class: str
    sources: RawSources
    ground_truth: RiskTier


@dataclass
class PedOLBench:
    items: list[BenchItem]
    splits: dict[str, list[int]]

    @classmethod
    def build(cls, cfg: BenchConfig, seed: int) -> PedOLBench:
        if not cfg.path:
            raise ValueError(
                "Set bench.path to a validated benchmark JSON file with recorded splits"
            )
        data = json.loads(Path(cfg.path).read_text(encoding="utf-8"))
        items = []
        for row in data["items"]:
            q = row["query"]
            source = row["sources"]
            label = dict(source["label"])
            label["indications"] = tuple(label["indications"])
            label["contraindications"] = tuple(label["contraindications"])
            label["passages"] = tuple(label.get("passages", []))
            kg = dict(source["kg"])
            kg["triples"] = tuple(kg.get("triples", []))
            kg["interactions"] = tuple(kg.get("interactions", []))
            items.append(
                BenchItem(
                    query=Query(
                        drug=q["drug"],
                        indication=q["indication"],
                        age_group=AgeGroup.parse(q["age_group"]),
                        context=tuple(q.get("context", [])),
                    ),
                    therapeutic_class=row["therapeutic_class"],
                    sources=RawSources(
                        label=LabelSections(**label),
                        contingency=Contingency(**source["contingency"]),
                        abstracts=tuple(Abstract(**a) for a in source["abstracts"]),
                        kg=KgNeighborhood(**kg),
                    ),
                    ground_truth=RiskTier.parse(row["ground_truth"]),
                )
            )
        splits_by_seed = data["splits"]
        splits = splits_by_seed.get(str(seed))
        if not isinstance(splits, dict) or set(splits) != {"train", "calibration", "test"}:
            raise ValueError(f"benchmark has no complete recorded split for seed {seed}")
        indices = [i for name in ("train", "calibration", "test") for i in splits[name]]
        if any(type(i) is not int or i < 0 or i >= len(items) for i in indices):
            raise ValueError("split indices must identify benchmark rows")
        if len(indices) != len(set(indices)) or set(indices) != set(range(len(items))):
            raise ValueError("splits must be disjoint and cover all benchmark rows")
        if any(not splits[name] for name in ("train", "calibration", "test")):
            raise ValueError("each partition must be non-empty")
        return cls(items=items, splits=splits)

    def subset(self, name: str) -> list[BenchItem]:
        return [self.items[i] for i in self.splits[name]]
