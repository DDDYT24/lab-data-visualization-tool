from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from labviz_api.research_figure_board import SceneIndex, interaction, scene_row_references
from labviz_api.research_preparation import PreparationError


def scene_fixture(pooled: bool) -> dict[str, Any]:
    pools = {
        str(pool): [
            {
                "id": f"row-{index:04d}",
                "sampleId": f"sample-{index // 3 + pool}",
                "value": index / 10 + pool,
                "label": "Needle" if index % 17 == 0 else "ordinary",
                "unit": "mV",
            }
            for index in reversed(range(230))
        ]
        for pool in range(2)
    }
    scene: dict[str, Any] = {
        "rowSets": pools,
        "panelRowSets": {label: str(index % 2) for index, label in enumerate("ABCD")},
        "marks": [
            {"id": "aggregate", "rowIds": ["row-0001", "row-0120"], "kind": "aggregate"},
            {"id": "point", "rowIds": ["row-0001"], "kind": "point"},
        ],
        "panels": [{"label": label} for label in "ABCD"],
        "sampled": False,
    }
    if not pooled:
        scene["rows"] = [{**row, "panel": label} for row, label in scene_row_references(scene)]
        del scene["rowSets"], scene["panelRowSets"]
    return scene


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("duplicates", [False, True])
def test_indexed_paging_selection_and_search_match_full_source_scan(
    pooled: bool, duplicates: bool
) -> None:
    scene = scene_fixture(pooled)
    if duplicates:
        rows = scene["rowSets"]["0"] if pooled else scene["rows"]
        rows.extend({**rows[0], "value": value} for value in (-100, -200))
    original = deepcopy(scene)
    index = SceneIndex.build(scene)
    ordered = list(index.references)
    for mark, row in ((None, None), ("point", None), ("aggregate", None), (None, "row-0120")):
        for query in ("", "nEeDlE", "missing-term"):
            for descending in (False, True):
                for offset in (0, 100, 850, 920, 1000):
                    expected = interaction(scene, mark, row, offset, query, descending)
                    assert (
                        interaction(scene, mark, row, offset, query, descending, index) == expected
                    )
    assert index.references == ordered
    assert scene == original
    # An observation can have distinct frozen values/sample membership in two row pools.
    selected = interaction(scene, None, "row-0120", 0, "", False, index)
    assert selected["selectedInputs"] == 1
    assert {row["panel"] for row in selected["rows"]} == set("ABCD")
    assert len({row["value"] for row in selected["rows"]}) > 1


@pytest.mark.parametrize("pooled", [False, True])
def test_indexed_invalid_selection_keeps_contract_error(pooled: bool) -> None:
    scene = scene_fixture(pooled)
    index = SceneIndex.build(scene)
    for mark, row, code in (("missing", None, "figure-mark"), (None, "missing", "figure-row")):
        with pytest.raises(PreparationError) as caught:
            interaction(scene, mark, row, 0, "", False, index)
        assert caught.value.code == code


@pytest.mark.parametrize("pooled", [False, True])
def test_compact_interaction_keeps_all_linked_rows_and_only_highlighted_geometry(
    pooled: bool,
) -> None:
    scene = scene_fixture(pooled)
    index = SceneIndex.build(scene)
    for mark, row in ((None, None), ("point", None), ("aggregate", None), (None, "row-0120")):
        full = interaction(scene, mark, row, 0, "", False, index)
        expected = {**full, "marks": [item for item in full["marks"] if item["highlighted"]]}
        for cached_index in (None, index):
            assert (
                interaction(scene, mark, row, 0, "", False, cached_index, highlights_only=True)
                == expected
            )
