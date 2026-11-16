"""FigureSpec v2: a figure made of panels, each panel made of layers.

`FigureBoardSpec` v1 stays readable: `upgrade_board` turns it into a v2 figure and
`project_board` gives the v1 view back when the figure uses nothing v1 cannot say.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .research_contract import Identifier, ResearchModel
from .research_figure_board import FigureBoardSpec

MAX_PANELS = 6
MAX_LAYERS = 8
Text = Annotated[str, Field(max_length=160)]
FieldName = Annotated[str, Field(min_length=1, max_length=255)]
Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
Scalar = bool | int | float | str | None


class Canvas(ResearchModel):
    size_preset: Literal["single_column", "double_column", "custom"] = "double_column"
    width: Annotated[float, Field(gt=0, le=2000)] | None = None
    height: Annotated[float, Field(gt=0, le=2000)] | None = None
    unit: Literal["mm", "in"] = "mm"
    dpi: int = Field(default=300, ge=72, le=1200)
    background: Color = "#ffffff"
    transparent: bool = False
    grayscale: bool = False

    @model_validator(mode="after")
    def custom_size_is_complete(self) -> Canvas:
        sized = self.width is not None and self.height is not None
        if (self.size_preset == "custom") != sized or (self.width is None) != (self.height is None):
            raise ValueError("Give width and height for a custom canvas, and only then.")
        return self


class Layout(ResearchModel):
    kind: Literal["grid", "mosaic"] = "grid"
    rows: int = Field(default=1, ge=1, le=MAX_PANELS)
    cols: int = Field(default=1, ge=1, le=MAX_PANELS)
    # One string per row, one character per cell; each character is a panel label.
    mosaic: list[Annotated[str, Field(min_length=1, max_length=MAX_PANELS)]] | None = Field(
        default=None, max_length=MAX_PANELS
    )
    gutter: float = Field(default=0.08, ge=0, le=0.5)
    label_style: Literal["upper", "lower", "none"] = "upper"


class Palette(ResearchModel):
    categorical: Identifier = "blue_teal"
    sequential: Identifier = "viridis"
    diverging: Identifier = "blue_red"


class AxisControl(ResearchModel):
    scale: Literal["linear", "log10", "symlog"] = "linear"
    domain: tuple[float, float] | None = None
    ticks: list[float] | None = Field(default=None, min_length=2, max_length=50)

    @model_validator(mode="after")
    def ordered(self) -> AxisControl:
        if self.domain and self.domain[0] >= self.domain[1]:
            raise ValueError("An axis domain runs from low to high.")
        if self.ticks and any(a >= b for a, b in zip(self.ticks, self.ticks[1:], strict=False)):
            raise ValueError("Axis ticks must increase.")
        if self.scale == "log10" and any(
            v <= 0 for v in (*(self.domain or ()), *(self.ticks or ()))
        ):
            raise ValueError("A log axis needs positive limits and ticks.")
        return self


class PanelAxes(ResearchModel):
    x: AxisControl = Field(default_factory=AxisControl)
    y: AxisControl = Field(default_factory=AxisControl)


class TextNote(ResearchModel):
    kind: Literal["text"] = "text"
    x: float
    y: float
    text: Annotated[str, Field(min_length=1, max_length=160)]


class LineNote(ResearchModel):
    kind: Literal["hline", "vline"]
    value: float
    label: Text = ""


class RectNote(ResearchModel):
    kind: Literal["rect"] = "rect"
    x: tuple[float, float]
    y: tuple[float, float]
    label: Text = ""


class BracketNote(ResearchModel):
    kind: Literal["bracket"] = "bracket"
    x: tuple[float, float]
    y: float
    label: Text = ""


Annotation = Annotated[TextNote | LineNote | RectNote | BracketNote, Field(discriminator="kind")]


class Encoding(ResearchModel):
    x: FieldName | None = None
    y: FieldName | None = None
    y2: FieldName | None = None
    z: FieldName | None = None
    group: FieldName | None = None
    error: FieldName | None = None
    weight: FieldName | None = None
    sample_id: FieldName | None = None


class Method(ResearchModel):
    id: Identifier
    params: dict[Annotated[str, Field(min_length=1, max_length=64)], Scalar] = Field(
        default_factory=dict, max_length=32
    )
    seed: Annotated[str, Field(max_length=64)] | None = None


class Style(ResearchModel):
    color: Color | None = None
    color_by: FieldName | None = None
    marker: Literal["o", "s", "^", "D", "v"] | None = None
    size: Annotated[float, Field(gt=0, le=40)] | None = None
    line_style: Literal["solid", "dashed", "dotted"] | None = None
    width: Annotated[float, Field(gt=0, le=10)] | None = None
    opacity: float = Field(default=1.0, ge=0, le=1)
    z_index: int = Field(default=0, ge=-100, le=100)

    @model_validator(mode="after")
    def one_colour_source(self) -> Style:
        if self.color and self.color_by:
            raise ValueError("Use a fixed colour or a colour field, not both.")
        return self


class Layer(ResearchModel):
    # "chart" draws one frozen saved chart whole; it is what every v1 panel is.
    kind: Literal[
        "chart", "line", "scatter", "bar", "histogram", "box", "violin", "strip", "paired",
        "heatmap", "contour", "surface3d", "fit", "errorbar", "band", "reference",
    ]  # fmt: skip
    chart_id: Identifier | None = None
    encoding: Encoding = Field(default_factory=Encoding)
    method: Method | None = None
    style: Style = Field(default_factory=Style)

    @model_validator(mode="after")
    def chart_reference(self) -> Layer:
        if (self.kind == "chart") != (self.chart_id is not None):
            raise ValueError("A chart layer names a saved chart; other layers do not.")
        return self


class DatasetRef(ResearchModel):
    dataset_id: Identifier
    version_id: Identifier


class FigurePanel(ResearchModel):
    id: Identifier
    label: Annotated[str, Field(pattern=r"^[A-Za-z]$")]
    title: Text = ""
    dataset_ref: DatasetRef | None = None
    axes: PanelAxes = Field(default_factory=PanelAxes)
    annotations: list[Annotation] = Field(default_factory=list, max_length=20)
    layers: list[Layer] = Field(min_length=1, max_length=MAX_LAYERS)


class FigureSpec(ResearchModel):
    schema_version: Literal[2] = 2
    title: Text = ""
    font_size: int = Field(default=10, ge=8, le=16)
    grid_visible: bool = True
    axis_mode: Literal["independent", "shared_x", "shared_y", "shared_xy"] = "independent"
    point_rendering: Literal["auto", "vector", "raster"] = "auto"
    canvas: Canvas = Field(default_factory=Canvas)
    layout: Layout = Field(default_factory=Layout)
    palette: Palette = Field(default_factory=Palette)
    panels: list[FigurePanel] = Field(min_length=1, max_length=MAX_PANELS)

    @model_validator(mode="after")
    def panels_fit_the_layout(self) -> FigureSpec:
        labels = [panel.label for panel in self.panels]
        if len({panel.id for panel in self.panels}) != len(labels) or len(set(labels)) != len(
            labels
        ):
            raise ValueError("Panel ids and labels must be unique.")
        layout = self.layout
        if layout.kind == "grid":
            if layout.mosaic is not None:
                raise ValueError("A grid layout has no mosaic.")
            if not len(labels) <= layout.rows * layout.cols < len(labels) + layout.cols:
                raise ValueError("The grid must hold every panel without an empty row.")
            return self
        rows = layout.mosaic or []
        if len(rows) != layout.rows or any(len(row) != layout.cols for row in rows):
            raise ValueError("The mosaic must have rows x cols cells.")
        if {cell for row in rows for cell in row} != set(labels):
            raise ValueError("The mosaic must place every panel and nothing else.")
        for label in labels:
            cells = [
                (r, c) for r, row in enumerate(rows) for c, cell in enumerate(row) if cell == label
            ]
            height = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
            width = max(c for _, c in cells) - min(c for _, c in cells) + 1
            if height * width != len(cells):
                raise ValueError("Each mosaic panel must cover one rectangle.")
        return self


GRID = {1: (1, 1), 2: (1, 2), 4: (2, 2)}


def upgrade_board(board: FigureBoardSpec) -> FigureSpec:
    """The v2 figure that draws exactly what a v1 board draws."""
    rows, cols = GRID[len(board.panel_chart_ids)]
    titles = board.panel_titles or [""] * len(board.panel_chart_ids)
    return FigureSpec(
        title=board.title,
        font_size=board.font_size,
        grid_visible=board.grid_visible,
        axis_mode=board.axis_mode,
        point_rendering=board.point_rendering,
        layout=Layout(rows=rows, cols=cols),
        palette=Palette(categorical=board.palette),
        panels=[
            FigurePanel(
                id=f"p{index + 1}",
                label="ABCD"[index],
                title=title,
                layers=[Layer(kind="chart", chart_id=chart)],
            )
            for index, (chart, title) in enumerate(zip(board.panel_chart_ids, titles, strict=True))
        ],
    )


def project_board(figure: FigureSpec) -> FigureBoardSpec | None:
    """The v1 view of a figure, or None when the figure says more than v1 can."""
    charts = [
        panel.layers[0].chart_id
        for panel in figure.panels
        if len(panel.layers) == 1 and panel.layers[0].chart_id
    ]
    if (
        len(charts) != len(figure.panels)
        or len(charts) not in GRID
        or figure.palette.categorical not in ("blue_teal", "grayscale")
    ):
        return None
    titles = [panel.title for panel in figure.panels]
    board = FigureBoardSpec(
        title=figure.title,
        font_size=figure.font_size,
        grid_visible=figure.grid_visible,
        axis_mode=figure.axis_mode,
        point_rendering=figure.point_rendering,
        palette=figure.palette.categorical,  # type: ignore[arg-type]
        panel_titles=titles if any(titles) else [],
        panel_chart_ids=charts,
    )
    # Anything v1 cannot carry (axes, notes, canvas, mosaic ...) shows up as a difference here.
    return board if upgrade_board(board) == figure else None
