from __future__ import annotations

from pedolgov.classification.hierarchy import (
    composite_grade,
    count_levels,
    grade_abstract,
)
from pedolgov.lanes.contracts import Abstract, EvidenceGrade


def test_rct_pediatric_is_level_a() -> None:
    assert grade_abstract(Abstract("rct", True, "1")) == EvidenceGrade.A


def test_observational_is_level_b() -> None:
    assert grade_abstract(Abstract("observational", False, "2")) == EvidenceGrade.B


def test_case_and_expert_levels() -> None:
    assert grade_abstract(Abstract("case", False, "3")) == EvidenceGrade.C
    assert grade_abstract(Abstract("expert", False, "4")) == EvidenceGrade.D


def test_composite_takes_best() -> None:
    grades = [EvidenceGrade.C, EvidenceGrade.A, EvidenceGrade.D]
    assert composite_grade(grades) == EvidenceGrade.A


def test_count_levels() -> None:
    grades = [EvidenceGrade.A, EvidenceGrade.A, EvidenceGrade.B, EvidenceGrade.C]
    assert count_levels(grades) == (2, 1)
