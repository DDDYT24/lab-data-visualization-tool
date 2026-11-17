"""Write contracts/figure-spec-v2.schema.json and the v24 figure-spec fixture matrix."""

import json
from pathlib import Path

from pydantic import ValidationError

from labviz_api.figure_spec import FigureBoardSpec, FigureSpec, project_board, upgrade_board

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def panel(index: int, label: str, **extra: object) -> dict[str, object]:
    return {
        "id": f"p{index}",
        "label": label,
        "layers": [{"kind": "chart", "chartId": "a"}],
        **extra,
    }


def one(**extra: object) -> dict[str, object]:
    return {"panels": [panel(1, "A")], **extra}


V1 = [
    {"panelChartIds": ["a"]},
    {
        "panelChartIds": ["a", "b"],
        "title": "Two",
        "axisMode": "shared_y",
        "palette": "grayscale",
        "fontSize": 12,
        "gridVisible": False,
        "pointRendering": "raster",
        "panelTitles": ["Left", "Right"],
    },
    {
        "panelChartIds": ["a", "b", "a", "c"],
        "axisMode": "shared_xy",
        "panelTitles": ["", "B", "", ""],
    },
]
ONLY_V2 = [
    {"layout": {"rows": 1, "cols": 3}, "panels": [panel(1, "A"), panel(2, "B"), panel(3, "C")]},
    {"layout": {"rows": 2, "cols": 3}, "panels": [panel(i + 1, "ABCDEF"[i]) for i in range(6)]},
    {
        "layout": {"kind": "mosaic", "rows": 2, "cols": 2, "mosaic": ["AA", "BC"]},
        "panels": [panel(1, "A"), panel(2, "B"), panel(3, "C")],
    },
    {
        "panels": [
            panel(
                1,
                "A",
                axes={"y": {"scale": "log10", "domain": [1, 1000], "ticks": [1, 10, 100, 1000]}},
            )
        ]
    },
    {
        "panels": [
            panel(
                1,
                "A",
                annotations=[
                    {"kind": "hline", "value": 0.5, "label": "limit"},
                    {"kind": "text", "x": 1, "y": 2, "text": "peak"},
                    {"kind": "rect", "x": [0, 1], "y": [0, 2]},
                    {"kind": "bracket", "x": [0, 1], "y": 3, "label": "p < 0.05"},
                    {"kind": "vline", "value": 2},
                ],
            )
        ]
    },
    {
        "panels": [
            {
                "id": "p1",
                "label": "A",
                "datasetRef": {"datasetId": "d1", "versionId": "v1"},
                "layers": [
                    {
                        "kind": "scatter",
                        "encoding": {"x": "time", "y": "signal", "group": "batch"},
                        "style": {"marker": "o", "opacity": 0.6},
                    },
                    {
                        "kind": "fit",
                        "encoding": {"x": "time", "y": "signal"},
                        "method": {"id": "linear", "params": {"robust": False}, "seed": "s1"},
                        "style": {"color": "#1f77b4", "lineStyle": "dashed", "zIndex": 1},
                    },
                ],
            }
        ]
    },
    one(
        canvas={
            "sizePreset": "custom",
            "width": 120,
            "height": 80,
            "dpi": 600,
            "transparent": True,
        },
        palette={"categorical": "okabe_ito"},
    ),
]
INVALID = [
    {"panels": []},
    {"layout": {"rows": 2, "cols": 4}, "panels": [panel(i + 1, "ABCDEFG"[i]) for i in range(7)]},
    {"layout": {"rows": 1, "cols": 2}, "panels": [panel(1, "A"), panel(1, "B")]},
    {"layout": {"rows": 1, "cols": 2}, "panels": [panel(1, "A"), panel(2, "A")]},
    {"panels": [panel(1, "A"), panel(2, "B")]},
    {"layout": {"rows": 2, "cols": 1}, "panels": [panel(1, "A")]},
    {
        "layout": {"kind": "mosaic", "rows": 2, "cols": 2, "mosaic": ["AB", "BA"]},
        "panels": [panel(1, "A"), panel(2, "B")],
    },
    {
        "layout": {"kind": "mosaic", "rows": 1, "cols": 2, "mosaic": ["AX"]},
        "panels": [panel(1, "A")],
    },
    {"panels": [panel(1, "A", axes={"y": {"scale": "log10", "domain": [0, 10]}})]},
    {"panels": [panel(1, "A", axes={"x": {"domain": [2, 1]}})]},
    {"panels": [panel(1, "A", axes={"x": {"ticks": [1, 1]}})]},
    {"panels": [panel(1, "A", annotations=[{"kind": "arrow", "x": 1, "y": 1}])]},
    {"panels": [{"id": "p1", "label": "A", "layers": []}]},
    {"panels": [{"id": "p1", "label": "A", "layers": [{"kind": "chart"}]}]},
    {"panels": [{"id": "p1", "label": "A", "layers": [{"kind": "line", "chartId": "a"}]}]},
    {
        "panels": [
            {
                "id": "p1",
                "label": "A",
                "layers": [{"kind": "line", "style": {"color": "#000000", "colorBy": "g"}}],
            }
        ]
    },
    one(canvas={"sizePreset": "custom"}),
    one(canvas={"width": 10, "height": 10}),
    one(schemaVersion=1),
    one(script="x"),
]


def main() -> None:
    schema = json.dumps(FigureSpec.model_json_schema(by_alias=True), indent=2) + "\n"
    (CONTRACTS / "figure-spec-v2.schema.json").write_text(schema, encoding="utf-8", newline="\n")
    pairs = []
    for body in V1:
        board = FigureBoardSpec.model_validate_json(json.dumps(body))
        figure = upgrade_board(board)
        assert project_board(figure) == board
        pairs.append(
            {"v1": json.loads(board.model_dump_json()), "v2": json.loads(figure.model_dump_json())}
        )
    only_v2 = []
    for body in ONLY_V2:
        figure = FigureSpec.model_validate_json(json.dumps(body))
        assert project_board(figure) is None
        only_v2.append(json.loads(figure.model_dump_json()))
    for body in INVALID:
        try:
            FigureSpec.model_validate_json(json.dumps(body))
        except ValidationError:
            continue
        raise SystemExit("accepted: " + json.dumps(body))
    target = CONTRACTS / "fixtures/v24/figure-spec.json"
    target.parent.mkdir(exist_ok=True)
    matrix = {"roundTrip": pairs, "onlyV2": only_v2, "invalid": INVALID}
    target.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(len(pairs), len(only_v2), len(INVALID))


if __name__ == "__main__":
    main()
