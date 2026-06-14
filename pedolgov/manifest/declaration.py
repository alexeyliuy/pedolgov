from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum


class AgeGroup(IntEnum):
    NEONATE = 0
    INFANT = 1
    CHILD = 2
    ADOLESCENT = 3

    @classmethod
    def parse(cls, token: str) -> AgeGroup:
        return _AGE_LOOKUP[token.strip().lower()]

    @property
    def label(self) -> str:
        return self.name.lower()


_AGE_LOOKUP = {
    "neonate": AgeGroup.NEONATE,
    "neonatal": AgeGroup.NEONATE,
    "infant": AgeGroup.INFANT,
    "child": AgeGroup.CHILD,
    "children": AgeGroup.CHILD,
    "adolescent": AgeGroup.ADOLESCENT,
}


class RiskTier(IntEnum):
    MINIMAL = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, token: str) -> RiskTier:
        return cls[token.strip().upper()]

    @property
    def label(self) -> str:
        return self.name.capitalize()


@dataclass(frozen=True)
class Query:
    drug: str
    indication: str
    age_group: AgeGroup
    context: tuple[str, ...] = field(default_factory=tuple)

    @property
    def query_id(self) -> str:
        raw = f"{self.drug}|{self.indication}|{self.age_group.label}|{','.join(self.context)}"
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()

    def describe(self) -> str:
        ctx = f" with {', '.join(self.context)}" if self.context else ""
        return f"{self.drug} for {self.indication} in {self.age_group.label}{ctx}"
