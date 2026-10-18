"""Data-free workflow recipes and transactional replacement-data application."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import Field, model_validator

from .models import ChartSpec
from .processing import ProcessingError
from .research_chart_render import COLORS, render_chart
from .research_charts import ScientificChartSpec, compute_chart
from .research_contract import ResearchModel
from .research_figure_board import FigureBoardSpec
from .research_figure_render import render_board
from .research_figure_sources import digest, legacy_snapshot, scientific_source, source
from .research_preparation import PreparationError, PreparationPlan, prepare
from .research_store import ResearchStore


class TemplatePanel(ResearchModel):
    kind: Literal["scientific", "legacy"]
    source_columns: list[str] = Field(max_length=128)
    plan: PreparationPlan | None = None
    chart: ScientificChartSpec | None = None
    legacy_chart: ChartSpec | None = None
    sample_field: str | None = None
    expected_unit: str
    legacy_colors: list[dict[str, str | int | None]] = Field(default_factory=list, max_length=240)

    @model_validator(mode="after")
    def recipe_only(self) -> TemplatePanel:
        if self.kind == "scientific":
            if self.plan is None or self.chart is None or self.legacy_chart is not None:
                raise ValueError("A scientific panel needs preparation and chart recipes.")
            if self.plan.exclusions:
                raise ValueError("Templates cannot carry row-specific exclusion decisions.")
        elif self.legacy_chart is None or self.plan is not None or self.chart is not None:
            raise ValueError("A legacy panel needs only its chart recipe.")
        for item in self.legacy_colors:
            if set(item) != {"series", "group", "color"} or not isinstance(item["series"], int):
                raise ValueError("Invalid legacy style mapping.")
            color = str(item["color"])
            if (
                len(color) != 7
                or color[0] != "#"
                or any(c not in "0123456789abcdefABCDEF" for c in color[1:])
            ):
                raise ValueError("Use a hex color.")
        return self


class WorkflowTemplate(ResearchModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=100)
    figure: FigureBoardSpec
    panels: list[TemplatePanel] = Field(min_length=1, max_length=4)
    exclusion_policy: Literal["review-new-data"] = "review-new-data"
    export_formats: list[Literal["svg", "png", "pdf"]] = Field(
        default=["svg", "png", "pdf"], min_length=1, max_length=3
    )
    png_dpi: Literal[300] = 300

    @model_validator(mode="after")
    def check_bindings(self) -> WorkflowTemplate:
        if self.export_formats != ["svg", "png", "pdf"]:
            raise ValueError("This version supports the SVG/PNG/PDF export set only.")
        if len(self.panels) not in {1, 2, 4} or self.figure.panel_chart_ids != [
            f"panel-{i}" for i in range(len(self.panels))
        ]:
            raise ValueError("Template panels use positional recipe keys, never experiment IDs.")
        return self


def now() -> str:
    return datetime.now(UTC).isoformat()


def install_reuse(store: ResearchStore) -> None:
    with store.connect() as c:
        c.executescript("""
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS research_templates (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
          spec_json TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS research_workflow_drafts (
          project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
          token TEXT NOT NULL UNIQUE, expected_revision TEXT, payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS research_output_drafts (
          project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
          token TEXT NOT NULL UNIQUE,
          board_id TEXT NOT NULL REFERENCES research_figure_boards(id) ON DELETE CASCADE,
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS research_outputs (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
          board_id TEXT NOT NULL REFERENCES research_figure_boards(id) ON DELETE CASCADE,
          manifest_json TEXT NOT NULL, notes_json TEXT NOT NULL, zip_blob BLOB NOT NULL,
          sha256 TEXT NOT NULL, created_at TEXT NOT NULL);
        COMMIT;
        """)


def save_template(store: ResearchStore, project_id: str, spec: WorkflowTemplate) -> dict[str, Any]:
    identifier = uuid4().hex
    with store.connect() as c:
        c.execute(
            "INSERT INTO research_templates VALUES (?,?,?,?)",
            (identifier, project_id, spec.model_dump_json(), now()),
        )
    return {"id": identifier, "spec": spec.model_dump()}


def template_from_board(
    store: ResearchStore, project_id: str, board_id: str, name: str
) -> WorkflowTemplate:
    with store.connect() as c:
        board = c.execute(
            "SELECT spec_json FROM research_figure_boards WHERE project_id=? AND id=?",
            (project_id, board_id),
        ).fetchone()
        if board is None:
            raise PreparationError("figure-not-found", "Choose a saved figure.")
        figure = FigureBoardSpec.model_validate_json(board["spec_json"])
        panels = [source(c, project_id, key) for key in figure.panel_chart_ids]
        labels = sorted(
            {g["label"] for p in panels if "scientific" in p for g in p["scientific"]["groups"]},
            key=lambda x: (x is not None, x or ""),
        )
        palette = (
            COLORS
            if figure.palette == "blue_teal"
            else ["#20252d", "#5b616b", "#858b94", "#b0b4ba"]
        )
        recipes = []
        for panel in panels:
            if "scientific" in panel:
                revision = c.execute(
                    (
                        "SELECT raw_json,prepared_json FROM research_revisions WHERE "
                        "project_id=? AND id=?"
                    ),
                    (project_id, panel["revisionId"]),
                ).fetchone()
                prepared = json.loads(revision["prepared_json"])
                raw = json.loads(revision["raw_json"])
                plan = PreparationPlan.model_validate(prepared["plan"])
                plan.exclusions = []
                chart = ScientificChartSpec.model_validate(panel["scientific"]["spec"])
                existing_styles = {item.group: item.model_dump() for item in chart.category_styles}
                styles = [
                    {"group": label, "color": palette[i % len(palette)], "marker": "o"}
                    | existing_styles.get(label, {})
                    for i, label in enumerate(labels)
                ]
                chart = ScientificChartSpec.model_validate(
                    chart.model_dump() | {"categoryStyles": styles}
                )
                recipes.append(
                    TemplatePanel(
                        kind="scientific",
                        source_columns=[
                            col for col in raw["columns"] if not col.startswith("__labviz_")
                        ],
                        plan=plan,
                        chart=chart,
                        expected_unit=panel["unit"],
                    )
                )
            else:
                chart_spec = ChartSpec.model_validate(panel["legacySpec"])
                from .render_contract import chart_series_color

                saved_colors = []
                for i, analysis in enumerate(panel["analysis"]["series"]):
                    series_index = next(
                        j for j, s in enumerate(chart_spec.series) if s.field == analysis["field"]
                    )
                    saved_colors.append(
                        {
                            "series": series_index,
                            "group": None if analysis["group"] is None else str(analysis["group"]),
                            "color": panel.get("seriesColors", {}).get(
                                analysis["label"],
                                chart_series_color(
                                    configured_color=palette[series_index % len(palette)],
                                    grayscale=figure.palette == "grayscale",
                                    grouped=bool(chart_spec.group_field),
                                    index=i,
                                ),
                            ),
                        }
                    )
                recipes.append(
                    TemplatePanel(
                        kind="legacy",
                        source_columns=panel["frame"]["columns"],
                        legacy_chart=chart_spec,
                        sample_field=panel.get("sampleField"),
                        expected_unit=panel["unit"],
                        legacy_colors=saved_colors,
                    )
                )
        figure.panel_chart_ids = [f"panel-{i}" for i in range(len(recipes))]
        return WorkflowTemplate(name=name, figure=figure, panels=recipes)


def read_template(store: ResearchStore, project_id: str, template_id: str) -> WorkflowTemplate:
    with store.connect() as c:
        row = c.execute(
            "SELECT spec_json FROM research_templates WHERE project_id=? AND id=?",
            (project_id, template_id),
        ).fetchone()
    if row is None:
        raise PreparationError(
            "template-not-found", "This template is not available in the project."
        )
    return WorkflowTemplate.model_validate_json(row["spec_json"])


def map_plan(plan: PreparationPlan, mapping: dict[str, str]) -> PreparationPlan:
    value = copy.deepcopy(plan.model_dump())
    for role, field in value["mapping"].items():
        if role != "defaultUnit" and field:
            value["mapping"][role] = mapping.get(field, field)
    for step in value["steps"]:
        for key in ("field", "valueField", "unitField", "limitField", "nameField"):
            if step.get(key):
                step[key] = mapping.get(step[key], step[key])
        for key in ("fields", "indexFields"):
            if key in step:
                step[key] = [mapping.get(field, field) for field in step[key]]
    return PreparationPlan.model_validate(value)


def legacy_replacement(
    recipe: TemplatePanel, raw: dict[str, Any], mapping: dict[str, str]
) -> dict[str, Any]:
    assert recipe.legacy_chart is not None
    chart = copy.deepcopy(recipe.legacy_chart.model_dump(mode="json", by_alias=True))
    for axis in ("xAxis", "yAxis"):
        chart[axis]["field"] = mapping.get(chart[axis]["field"], chart[axis]["field"])
    for series in chart["series"]:
        series["field"] = mapping.get(series["field"], series["field"])
    if chart["groupField"]:
        chart["groupField"] = mapping.get(chart["groupField"], chart["groupField"])
    for option in ("uncertainty", "secondaryYAxis"):
        key = "errorField" if option == "uncertainty" else "field"
        if chart[option].get(key):
            chart[option][key] = mapping.get(chart[option][key], chart[option][key])
    frame = pd.DataFrame([r["cells"] for r in raw["rows"]])
    numeric = [s["field"] for s in chart["series"]]
    if chart["uncertainty"].get("errorField"):
        numeric.append(chart["uncertainty"]["errorField"])
    for field in numeric:
        if field not in frame:
            raise PreparationError("template-field", f"Map the missing field: {field}")
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    x = chart["xAxis"]["field"]
    if x in frame and pd.to_numeric(frame[x], errors="coerce").notna().all():
        frame[x] = pd.to_numeric(frame[x])

    class Replacement:
        def get_workspace(self, _: str) -> dict[str, Any]:
            return {"chart": chart, "decisions": [], "quality": {}, "preview": {}}

        def load_chart_dataframe(self, _: str) -> Any:
            return frame

        def load_quality_dataframe(self, _: str) -> Any:
            return frame

    sample = mapping.get(recipe.sample_field, recipe.sample_field) if recipe.sample_field else None
    result = legacy_snapshot(Replacement(), "replacement", 1, sample, raw["namespace"])
    result["seriesColors"] = {}
    for a in result["analysis"]["series"]:
        i = next(i for i, s in enumerate(chart["series"]) if s["field"] == a["field"])
        group = None if a["group"] is None else str(a["group"])
        saved = next(
            (s for s in recipe.legacy_colors if s["series"] == i and s["group"] == group), None
        )
        result["seriesColors"][a["label"]] = (
            saved["color"]
            if saved
            else COLORS[int(hashlib.sha256(str(group).encode()).hexdigest()[:8], 16) % len(COLORS)]
        )
    result["resultSha256"] = digest({k: v for k, v in result.items() if k != "resultSha256"})
    return result


def preview_template(
    store: ResearchStore,
    project_id: str,
    spec: WorkflowTemplate,
    raw: dict[str, Any],
    mapping: dict[str, str],
    expected: str | None,
) -> dict[str, Any]:
    unknown = set(mapping) - {field for panel in spec.panels for field in panel.source_columns}
    if unknown or any(not value.strip() for value in mapping.values()):
        raise PreparationError(
            "template-mapping", "Map only listed source columns to non-empty replacement names."
        )
    reports: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    for i, recipe in enumerate(spec.panels):
        label = "ABCD"[i]
        try:
            if recipe.kind == "scientific":
                assert recipe.plan is not None and recipe.chart is not None
                prepared = prepare(raw, map_plan(recipe.plan, mapping))
                chart_data = recipe.chart.model_dump()
                known = {s["group"] for s in chart_data["categoryStyles"]}
                groups = {o["group"] for o in prepared["observations"]}
                for group in sorted(groups - known, key=lambda x: (x is not None, x or "")):
                    chart_data["categoryStyles"].append(
                        {
                            "group": group,
                            "color": COLORS[
                                int(hashlib.sha256(str(group).encode()).hexdigest()[:8], 16)
                                % len(COLORS)
                            ],
                            "marker": "o",
                        }
                    )
                result = compute_chart(prepared, ScientificChartSpec.model_validate(chart_data))
                if result["unit"] != recipe.expected_unit:
                    raise PreparationError(
                        "template-unit",
                        f"Expected {recipe.expected_unit}; got {result['unit']}. "
                        "Add an explicit conversion to the recipe.",
                    )
                ready.append(
                    {"index": i, "kind": "scientific", "prepared": prepared, "result": result}
                )
                count = len(result["points"])
                styles = result["spec"]["categoryStyles"]
            else:
                result = legacy_replacement(recipe, raw, mapping)
                ready.append({"index": i, "kind": "legacy", "result": result})
                count = len(result["rows"])
                styles = result.get("seriesColors", {})
            reports.append(
                {
                    "panel": label,
                    "status": "ready",
                    "observations": count,
                    "unit": recipe.expected_unit,
                    "styles": styles,
                    "message": (
                        "Ready. Review source identity, QC and declared units before applying."
                    ),
                }
            )
        except (PreparationError, ProcessingError, ValueError, KeyError) as error:
            reports.append(
                {
                    "panel": label,
                    "status": "blocked",
                    "observations": 0,
                    "unit": recipe.expected_unit,
                    "styles": [],
                    "message": str(error),
                }
            )
    figure_error = None
    if len(ready) == len(spec.panels):
        try:
            panels = [
                scientific_source(p["result"], "preview")
                if p["kind"] == "scientific"
                else p["result"]
                for p in ready
            ]
            render_board(panels, spec.figure)
        except (PreparationError, ValueError) as error:
            figure_error = str(error)
    token = uuid4().hex
    payload = {
        "raw": raw,
        "ready": ready,
        "template": spec.model_dump(),
        "reports": reports,
        "figureError": figure_error,
    }
    with store.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        store.check_head(c, project_id, expected)
        c.execute(
            (
                "INSERT INTO research_workflow_drafts VALUES (?,?,?,?,?) ON "
                "CONFLICT(project_id) DO UPDATE SET token=excluded.token,expected_revision=ex"
                "cluded.expected_revision,payload_json=excluded.payload_json,created_at=exclu"
                "ded.created_at"
            ),
            (project_id, token, expected, json.dumps(payload, allow_nan=False), now()),
        )
    return {
        "token": token,
        "expectedRevision": expected,
        "panels": reports,
        "readyCount": len(ready),
        "figureReady": len(ready) == len(spec.panels) and figure_error is None,
        "figureError": figure_error,
        "source": raw["sources"],
        "notice": (
            "No old row exclusions are copied. Legacy units use declared axes. "
            "Unknown categories receive explicit styles shown above."
        ),
    }


def apply_template(
    store: ResearchStore, project_id: str, token: str, expected: str | None
) -> dict[str, Any]:
    with store.connect() as c:
        draft = c.execute(
            "SELECT * FROM research_workflow_drafts WHERE project_id=? AND token=?",
            (project_id, token),
        ).fetchone()
    if (
        draft is None
        or draft["expected_revision"] != expected
        or datetime.fromisoformat(draft["created_at"]) < datetime.now(UTC) - timedelta(minutes=30)
    ):
        raise PreparationError("preview-expired", "Preview the replacement workflow again.")
    payload = json.loads(draft["payload_json"])
    if not payload["ready"]:
        raise PreparationError("template-blocked", "Resolve at least one panel before applying.")
    spec = WorkflowTemplate.model_validate(payload["template"])
    timestamp = now()
    revisions, charts, legacy_panels, sources, ids = [], [], [], [], []
    artifacts: dict[str, bytes] | None = None
    parent = expected
    for entry in payload["ready"]:
        key = uuid4().hex
        ids.append(key)
        result = entry["result"]
        if entry["kind"] == "scientific":
            revision = uuid4().hex
            revisions.append(
                (
                    revision,
                    project_id,
                    parent,
                    json.dumps(payload["raw"], allow_nan=False),
                    json.dumps(entry["prepared"], allow_nan=False),
                    timestamp,
                )
            )
            parent = revision
            artifacts = render_chart(result)
            charts.append(
                (
                    key,
                    project_id,
                    revision,
                    json.dumps(result, allow_nan=False),
                    artifacts["svg"],
                    artifacts["png"],
                    artifacts["pdf"],
                    timestamp,
                )
            )
            sources.append(scientific_source(result, revision))
        else:
            legacy_panels.append((key, project_id, json.dumps(result, allow_nan=False), timestamp))
            sources.append(result)
    board_id = None
    artifacts = None
    scene = None
    if len(ids) == len(spec.panels) and payload["figureError"] is None:
        spec.figure.panel_chart_ids = ids
        artifacts, scene = render_board(sources, spec.figure)
        board_id = uuid4().hex
    with store.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        store.check_head(c, project_id, expected)
        if (
            c.execute(
                "SELECT token FROM research_workflow_drafts WHERE project_id=? AND token=?",
                (project_id, token),
            ).fetchone()
            is None
        ):
            raise PreparationError(
                "preview-expired", "Preview has already been consumed or replaced."
            )
        c.executemany("INSERT INTO research_revisions VALUES (?,?,?,?,?,?)", revisions)
        c.executemany("INSERT INTO research_charts VALUES (?,?,?,?,?,?,?,?)", charts)
        c.executemany("INSERT INTO research_figure_sources VALUES (?,?,?,?)", legacy_panels)
        if revisions:
            c.execute(
                (
                    "INSERT INTO research_heads VALUES (?,?,'[]') ON CONFLICT(project_id) DO "
                    "UPDATE SET revision_id=excluded.revision_id,future_json='[]'"
                ),
                (project_id, parent),
            )
            c.execute("DELETE FROM research_drafts WHERE project_id=?", (project_id,))
        if board_id and artifacts is not None:
            c.execute(
                "INSERT INTO research_figure_boards VALUES (?,?,?,?)",
                (board_id, project_id, spec.figure.model_dump_json(), timestamp),
            )
            c.execute(
                "INSERT INTO research_figure_artifacts VALUES (?,?,?,?,?)",
                (
                    board_id,
                    artifacts["svg"],
                    artifacts["png"],
                    artifacts["pdf"],
                    json.dumps(scene, allow_nan=False),
                ),
            )
        c.execute("DELETE FROM research_workflow_drafts WHERE project_id=?", (project_id,))
    return {
        "chartIds": ids,
        "boardId": board_id,
        "revisionId": parent,
        "panels": payload["reports"],
    }
