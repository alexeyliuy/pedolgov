from __future__ import annotations

import math
from collections.abc import Iterable

from pedolgov.lanes.contracts import Contingency

_UNITS = {
    "800": 10.0,
    "801": 1.0,
    "802": 1 / 12,
    "803": 7 / 365.25,
    "804": 1 / 365.25,
    "805": 1 / (365.25 * 24),
}


def pediatric_reports(
    records: Iterable[dict], synonyms: dict[str, str]
) -> list[tuple[set[str], set[str]]]:
    normalized = {
        key.strip().casefold(): value.strip().casefold() for key, value in synonyms.items()
    }
    newest: dict[str, tuple[int, dict]] = {}
    for record in records:
        identifier = str(record.get("safetyreportid", "")).strip()
        if not identifier:
            continue
        try:
            version = int(record.get("safetyreportversion", 0))
        except (ValueError, TypeError):
            continue
        if identifier not in newest or version > newest[identifier][0]:
            newest[identifier] = (version, record)
    result = []
    for _, record in newest.values():
        patient = record.get("patient") or {}
        unit = _UNITS.get(str(patient.get("patientonsetageunit", "")))
        try:
            age = float(patient["patientonsetage"]) * unit if unit is not None else math.nan
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(age) or not 0 <= age < 18:
            continue
        drugs = set()
        for drug in patient.get("drug", []):
            active = (drug.get("activesubstance") or {}).get("activesubstancename", "")
            name = str(active or drug.get("medicinalproduct", "")).strip().casefold()
            if name in normalized:
                drugs.add(normalized[name])
        if not drugs:
            continue
        events = {
            str(x.get("reactionmeddrapt", "")).strip().casefold()
            for x in patient.get("reaction", [])
            if x.get("reactionmeddrapt")
        }
        result.append((drugs, events))
    return result


def contingency_table(
    records: list[tuple[set[str], set[str]]], drug: str, event: str
) -> Contingency:
    drug, event = drug.strip().casefold(), event.strip().casefold()
    if not drug or not event:
        raise ValueError("drug and event must be provided")
    cells = [0, 0, 0, 0]
    for drugs, events in records:
        present_drug, present_event = drug in drugs, event in events
        index = (
            0
            if present_drug and present_event
            else 1
            if present_drug
            else 2
            if present_event
            else 3
        )
        cells[index] += 1
    if not records:
        raise ValueError("no eligible records")
    return Contingency(a=cells[0], b=cells[1], c=cells[2], d=cells[3])
