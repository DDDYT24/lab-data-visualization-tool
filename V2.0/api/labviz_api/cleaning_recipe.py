"""The recipe a cleaned dataset version records, and the hash that names it.

A cleaned version is its parent minus the rows of every ``remove`` decision. Its recipe lists
each decision that drops rows, with a digest of the rows it froze, so two versions with one
recipe hash hold the same table, and a chart drawn from one can be cached under that hash.
``exclude`` decisions are in the recipe too, although they only leave the chart and keep the
rows in the cleaned table.

Kept free of the scientific stack, so the repository and migrations can build recipes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labviz_api.parquet import ParquetArtifact

RECIPE_VERSION = "labviz.cleaning-recipe/1"
# Decision actions that drop a finding's rows from charts; "remove" also drops them from
# the cleaned table.
EXCLUDING_ACTIONS = frozenset({"exclude", "remove"})


@dataclass(frozen=True)
class CleanedVersion:
    """A cleaned version to point the project at.

    ``dataset`` is None when the recipe removes no rows, so the version shares its parent's
    bytes.
    """

    recipe: dict[str, Any]
    sha256: str
    dataset: ParquetArtifact | None


def rows_sha256(rows: Iterable[int]) -> str:
    return hashlib.sha256(",".join(str(row) for row in sorted(rows)).encode("ascii")).hexdigest()


def build_recipe(
    parent_sha256: str, decisions: Iterable[Mapping[str, str]], frozen: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    steps = []
    for item in sorted(decisions, key=lambda item: item["findingId"]):
        if item["action"] not in EXCLUDING_ACTIONS:
            continue
        rows = frozen.get(item["findingId"])
        if rows is None:
            raise ValueError(f"Decision on {item['findingId']} has no frozen rows.")
        steps.append(
            {
                "findingId": item["findingId"],
                "action": item["action"],
                "rowCount": len(rows),
                "rowsSha256": rows_sha256(rows),
            }
        )
    return {"recipeVersion": RECIPE_VERSION, "parentSha256": parent_sha256, "steps": steps}


def recipe_sha256(recipe: Mapping[str, Any]) -> str:
    canonical = json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decided_rows(
    decisions: Iterable[Mapping[str, str]],
    frozen: Mapping[str, Sequence[int]],
    actions: Iterable[str],
) -> list[int]:
    """Return the sorted union of the rows frozen by decisions with one of ``actions``."""
    wanted = set(actions)
    rows: set[int] = set()
    for item in decisions:
        if item["action"] in wanted:
            rows.update(frozen[item["findingId"]])
    return sorted(rows)
