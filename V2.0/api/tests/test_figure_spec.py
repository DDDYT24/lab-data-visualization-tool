"""FigureSpec v2 and its v1 projection, against the matrix the browser tests also read."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from labviz_api.figure_spec import FigureBoardSpec, FigureSpec, project_board, upgrade_board

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
MATRIX = json.loads((CONTRACTS / "fixtures/v24/figure-spec.json").read_text(encoding="utf-8"))


def test_schema_matches_the_checked_in_contract() -> None:
    schema = json.loads((CONTRACTS / "figure-spec-v2.schema.json").read_text(encoding="utf-8"))
    assert schema == FigureSpec.model_json_schema(by_alias=True)


@pytest.mark.parametrize("pair", MATRIX["roundTrip"])
def test_v1_boards_round_trip(pair: dict[str, Any]) -> None:
    board = FigureBoardSpec.model_validate_json(json.dumps(pair["v1"]))
    figure = FigureSpec.model_validate_json(json.dumps(pair["v2"]))
    assert upgrade_board(board) == figure
    assert project_board(figure) == board
    assert json.loads(figure.model_dump_json()) == pair["v2"]


@pytest.mark.parametrize("body", MATRIX["onlyV2"])
def test_figures_beyond_v1_have_no_projection(body: dict[str, Any]) -> None:
    figure = FigureSpec.model_validate_json(json.dumps(body))
    assert json.loads(figure.model_dump_json()) == body
    assert project_board(figure) is None


@pytest.mark.parametrize("body", MATRIX["invalid"])
def test_invalid_figures_are_rejected(body: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FigureSpec.model_validate_json(json.dumps(body))


def test_blank_v1_titles_project_to_no_titles() -> None:
    board = FigureBoardSpec(panel_chart_ids=["a", "b"], panel_titles=["", ""])
    assert project_board(upgrade_board(board)) == FigureBoardSpec(panel_chart_ids=["a", "b"])
