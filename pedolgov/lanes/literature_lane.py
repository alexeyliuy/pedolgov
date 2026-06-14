from __future__ import annotations

from collections.abc import Sequence

from pedolgov.classification.hierarchy import count_levels, grade_abstract
from pedolgov.lanes.contracts import Abstract, EvidenceGrade


def grade_literature(abstracts: Sequence[Abstract]) -> list[EvidenceGrade]:
    return [grade_abstract(a) for a in abstracts]


def literature_levels(abstracts: Sequence[Abstract], use_grading: bool) -> tuple[int, int]:
    if not use_grading:
        downgraded = [EvidenceGrade.C for _ in abstracts]
        return count_levels(downgraded)
    return count_levels(grade_literature(abstracts))
