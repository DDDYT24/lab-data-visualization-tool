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
import sys
import zlib
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labviz_api.parquet import ParquetArtifact

RECIPE_VERSION = "labviz.cleaning-recipe/1"
# Decision actions that drop a finding's rows from charts; "remove" also drops them from
# the cleaned table.
EXCLUDING_ACTIONS = frozenset({"exclude", "remove"})
PACKED_ROWS_FORMAT = b"LVR1"


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


def _little_endian(values: array[int]) -> array[int]:
    if sys.byteorder == "big":
        values.byteswap()
    return values


@dataclass(frozen=True)
class FrozenRows:
    """The rows one decision froze, packed as zlib-compressed little-endian uint32 gaps.

    ``sha256`` is ``rows_sha256`` of the positions, kept so a recipe can be rebuilt without
    unpacking them.
    """

    row_count: int
    sha256: str
    packed: bytes

    @classmethod
    def of(cls, rows: Iterable[int]) -> FrozenRows:
        ordered = sorted(set(rows))
        gaps = _little_endian(
            array("I", (row - previous for previous, row in pairwise([0, *ordered])))
        )
        return cls(
            len(ordered), rows_sha256(ordered), PACKED_ROWS_FORMAT + zlib.compress(gaps.tobytes())
        )

    def gaps(self) -> bytes:
        """Return the little-endian uint32 gaps whose running sum is the positions."""
        if not self.packed.startswith(PACKED_ROWS_FORMAT):
            raise ValueError("Frozen rows use an unknown packing.")
        gaps = zlib.decompress(self.packed[len(PACKED_ROWS_FORMAT) :])
        if len(gaps) != 4 * self.row_count:
            raise ValueError("Frozen rows do not match their recorded count.")
        return gaps

    def positions(self) -> array[int]:
        gaps = array("I")
        gaps.frombytes(self.gaps())
        return array("I", accumulate(_little_endian(gaps)))


def build_recipe(
    parent_sha256: str, decisions: Iterable[Mapping[str, str]], frozen: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    return recipe_from_sets(
        parent_sha256, decisions, {finding: FrozenRows.of(rows) for finding, rows in frozen.items()}
    )


def recipe_from_sets(
    parent_sha256: str, decisions: Iterable[Mapping[str, str]], frozen: Mapping[str, FrozenRows]
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
                "rowCount": rows.row_count,
                "rowsSha256": rows.sha256,
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
