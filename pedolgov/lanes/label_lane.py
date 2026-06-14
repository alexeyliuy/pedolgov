from __future__ import annotations

from dataclasses import dataclass

from pedolgov.lanes.contracts import LabelSections

_GAP_MARKERS = ("insufficient", "not established", "avoid", "not recommended", "no data")


@dataclass(frozen=True)
class LabelReadout:
    drug: str
    has_contraindication: bool
    dose_min: float
    dose_max: float
    label_gap: bool
    off_label_category: str


def parse_label(label: LabelSections) -> LabelReadout:
    text = label.pediatric_use.lower()
    gap = any(marker in text for marker in _GAP_MARKERS)
    return LabelReadout(
        drug=label.drug,
        has_contraindication=len(label.contraindications) > 0,
        dose_min=label.dose_min,
        dose_max=label.dose_max,
        label_gap=gap,
        off_label_category=label.off_label_category,
    )
