from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pedolgov.records.io import atomic_write_json


@dataclass(frozen=True)
class LedgerEntry:
    query_id: str
    stage: str
    payload: dict[str, Any]


@dataclass
class Ledger:
    entries: list[LedgerEntry] = field(default_factory=list)

    def record(self, query_id: str, stage: str, payload: dict[str, Any]) -> None:
        self.entries.append(LedgerEntry(query_id=query_id, stage=stage, payload=dict(payload)))

    def trail(self, query_id: str) -> list[LedgerEntry]:
        return [e for e in self.entries if e.query_id == query_id]

    def citations(self, query_id: str) -> list[str]:
        cites: list[str] = []
        for entry in self.trail(query_id):
            for cite in entry.payload.get("citations", []):
                if cite not in cites:
                    cites.append(str(cite))
        return cites

    def dump(self, path: str | os.PathLike[str]) -> None:
        rows = [
            {"query_id": e.query_id, "stage": e.stage, "payload": e.payload} for e in self.entries
        ]
        atomic_write_json(path, rows)
