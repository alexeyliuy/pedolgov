from __future__ import annotations

from collections.abc import Iterable

from pedolgov.lanes.contracts import Abstract, EvidenceGrade


def grade_abstract(abstract: Abstract) -> EvidenceGrade:
    if abstract.design == "rct" and abstract.pediatric:
        return EvidenceGrade.A
    if abstract.design == "rct":
        return EvidenceGrade.B
    if abstract.design == "observational":
        return EvidenceGrade.B
    if abstract.design == "case":
        return EvidenceGrade.C
    if abstract.design == "expert":
        return EvidenceGrade.D
    return EvidenceGrade.E


def grade_label_statement(has_pediatric_statement: bool) -> EvidenceGrade:
    return EvidenceGrade.C if has_pediatric_statement else EvidenceGrade.E


def grade_pharmacovigilance(consensus_positive: bool) -> EvidenceGrade:
    return EvidenceGrade.B if consensus_positive else EvidenceGrade.E


def composite_grade(grades: Iterable[EvidenceGrade]) -> EvidenceGrade:
    best = EvidenceGrade.E
    for grade in grades:
        if grade < best:
            best = grade
    return best


def count_levels(grades: Iterable[EvidenceGrade]) -> tuple[int, int]:
    level_a = 0
    level_b = 0
    for grade in grades:
        if grade == EvidenceGrade.A:
            level_a += 1
        elif grade == EvidenceGrade.B:
            level_b += 1
    return level_a, level_b
