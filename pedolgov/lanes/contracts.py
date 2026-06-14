from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from pedolgov.manifest.declaration import AgeGroup

OFF_LABEL_CATEGORIES = ("indication", "age", "dose", "route")


class EvidenceGrade(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class LabelSections:
    drug: str
    indications: tuple[str, ...]
    contraindications: tuple[str, ...]
    pediatric_use: str
    dose_min: float
    dose_max: float
    off_label_category: str


@dataclass(frozen=True)
class Contingency:
    a: int
    b: int
    c: int
    d: int

    @property
    def total(self) -> int:
        return self.a + self.b + self.c + self.d


@dataclass(frozen=True)
class Abstract:
    design: str
    pediatric: bool
    pmid: str


@dataclass(frozen=True)
class KgNeighborhood:
    interaction_count: int
    max_interaction_count: int
    narrow_therapeutic_index: bool
    on_kids_list: bool
    contraindicated_with_context: bool


@dataclass(frozen=True)
class RawSources:
    label: LabelSections
    contingency: Contingency
    abstracts: tuple[Abstract, ...]
    kg: KgNeighborhood


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    grade: EvidenceGrade
    statement: str
    citation: str


@dataclass(frozen=True)
class EvidenceSummary:
    query_id: str
    drug: str
    age_group: AgeGroup
    disproportionality_consensus: float
    off_label_category: str
    level_a_count: int
    level_b_count: int
    interaction_count: int
    max_interaction_count: int
    narrow_ti: bool
    on_kids_list: bool
    label_gap: bool
    has_contraindication: bool
    dose_min: float
    dose_max: float
    source_risk: dict[str, float]
    items: tuple[EvidenceItem, ...] = field(default_factory=tuple)
    grounded: bool = True
    hallucinated_claims: int = 0

    @property
    def citations(self) -> tuple[str, ...]:
        return tuple(item.citation for item in self.items)
