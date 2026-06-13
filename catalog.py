"""Feature catalogue — loaded from catalog.json, fully mutable at runtime."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


@dataclass
class Spec:
    name: str
    category: str
    dtype: Literal["num", "cat", "bin"]
    params: dict = field(default_factory=dict)
    meaningful: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "dtype": self.dtype,
            "params": self.params,
            "meaningful": self.meaningful,
        }


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def _load_from_file() -> list[Spec]:
    """Read catalog.json and return a list of Spec objects."""
    with CATALOG_PATH.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return [Spec(**item) for item in raw]


def save(catalog: list[Spec] | None = None) -> None:
    """Persist *catalog* (defaults to the live CATALOG) to catalog.json."""
    target = catalog if catalog is not None else CATALOG
    with CATALOG_PATH.open("w", encoding="utf-8") as fh:
        json.dump([s.to_dict() for s in target], fh, indent=2, ensure_ascii=False)


def reload() -> None:
    """Re-read catalog.json into the module-level CATALOG list in-place."""
    fresh = _load_from_file()
    CATALOG.clear()
    CATALOG.extend(fresh)


# Module-level list — mutated in-place so all importers see changes immediately.
CATALOG: list[Spec] = _load_from_file()


# ---------------------------------------------------------------------------
# Helpers (unchanged public API)
# ---------------------------------------------------------------------------

def by_category() -> dict[str, list[Spec]]:
    out: dict[str, list[Spec]] = {}
    for s in CATALOG:
        out.setdefault(s.category, []).append(s)
    return out


def get(name: str) -> Spec:
    for s in CATALOG:
        if s.name == name:
            return s
    raise KeyError(name)


def names() -> list[str]:
    return [s.name for s in CATALOG]


def categories() -> list[str]:
    return sorted({s.category for s in CATALOG})