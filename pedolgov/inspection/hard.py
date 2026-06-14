from __future__ import annotations

from pedolgov.lanes.contracts import EvidenceSummary
from pedolgov.manifest.declaration import Query


def check_drug_name(summary: EvidenceSummary, query: Query) -> bool:
    return summary.drug.strip().lower() != query.drug.strip().lower()


def check_contraindication(summary: EvidenceSummary) -> bool:
    return summary.has_contraindication


def check_citation_traceability(summary: EvidenceSummary) -> bool:
    if summary.hallucinated_claims > 0:
        return True
    return any(not item.citation for item in summary.items)


def check_dose_range(summary: EvidenceSummary) -> bool:
    return not (summary.dose_min > 0.0 and summary.dose_max > summary.dose_min)


def check_interaction(summary: EvidenceSummary) -> bool:
    return summary.narrow_ti and summary.interaction_count > 0


class HardGuardrails:
    def evaluate(self, summary: EvidenceSummary, query: Query) -> list[str]:
        triggered: list[str] = []
        if check_drug_name(summary, query):
            triggered.append("drug_name")
        if check_contraindication(summary):
            triggered.append("contraindication")
        if check_citation_traceability(summary):
            triggered.append("citation_traceability")
        if check_dose_range(summary):
            triggered.append("dose_range")
        if check_interaction(summary):
            triggered.append("interaction")
        return triggered
