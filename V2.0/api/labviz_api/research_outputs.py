"""Inspectable, immutable research output ZIPs; not project backups."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import Field

from .research_contract import ResearchModel
from .research_figure_board import artifact_row, read_board
from .research_figure_sources import source
from .research_preparation import PreparationError
from .research_reuse import now
from .research_store import ResearchStore


class OutputNotes(ResearchModel):
    legend_en: str = Field(max_length=20000)
    legend_zh: str = Field(max_length=20000)
    methods_en: str = Field(max_length=20000)
    methods_zh: str = Field(max_length=20000)


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def csv_bytes(rows: list[dict[str, Any]], headers: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers)
    writer.writeheader()
    for row in rows:
        clean = {}
        for key in headers:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if isinstance(value, str) and value.lstrip().startswith(
                ("=", "+", "-", "@", "\t", "\r")
            ):
                value = "'" + value
            clean[key] = value
        writer.writerow(clean)
    return stream.getvalue().encode("utf-8-sig")


def materialize(
    store: ResearchStore, project_id: str, board_id: str
) -> tuple[dict[str, Any], OutputNotes, dict[str, bytes]]:
    board = read_board(store, project_id, board_id)
    artifacts = artifact_row(store, project_id, board_id)
    files = {"figure." + kind: bytes(artifacts[kind]) for kind in ("svg", "png", "pdf")}
    panels, legends_en, legends_zh, methods_en, methods_zh = [], [], [], [], []
    rendering = json.loads(artifacts["scene_json"]).get("exportRendering")
    if rendering and rendering["rasterPanels"]:
        labels = ", ".join(rendering["rasterPanels"])
        methods_en.append(
            f"Panels {labels}: full dense data layer embedded at 300 dpi in SVG/PDF; "
            "axes/text and density contours remain vector. "
            "Individual embedded points are not editable. Full source CSV retained."
        )
        methods_zh.append(
            f"面板 {labels}：SVG/PDF 中完整密集数据层以 300 dpi 嵌入；"
            "坐标轴、文字和密度轮廓保留矢量。嵌入点无法逐个编辑，保留完整源数据 CSV。"
        )
    with store.connect() as c:
        for i, chart_id in enumerate(board["spec"]["panelChartIds"]):
            label = "ABCD"[i]
            panel = source(c, project_id, chart_id)
            panel["title"] = board["panels"][i]["title"]
            science = panel.get("scientific")
            observations = panel["rows"] + (science["excluded"] if science else [])
            if science:
                csv_rows = [
                    {
                        "panel": label,
                        **row,
                        "derivedValue": row["value"],
                        "sourceReferences": row["lineage"],
                    }
                    for row in observations
                ]
                headers = [
                    "panel",
                    "id",
                    "sampleId",
                    "sampleLabel",
                    "experimentalUnit",
                    "group",
                    "time",
                    "batch",
                    "technicalRepeat",
                    "pairKey",
                    "reportedText",
                    "reportedUnit",
                    "derivedValue",
                    "unit",
                    "state",
                    "limit",
                    "exclusionReason",
                    "sourceReferences",
                ]
                revision = c.execute(
                    "SELECT prepared_json FROM research_revisions WHERE project_id=? AND id=?",
                    (project_id, panel["revisionId"]),
                ).fetchone()
                prepared = json.loads(revision["prepared_json"])
                plan, history = prepared["plan"], prepared["history"]
                method = {
                    "chart": science["spec"],
                    "algorithmVersion": science["algorithmVersion"],
                    "preparation": plan,
                    "processingHistory": history,
                    "densityMethods": [
                        {
                            k: v
                            for k, v in d.items()
                            if k not in {"grid", "density", "halfWidths", "observationIds"}
                        }
                        for d in science["densities"]
                    ],
                }
                analysis = science
                reported_role = (
                    "Reported measurement text/unit plus derived value; "
                    "exclusions and source references retained. Original "
                    "upload bytes are not included."
                )
                count = len(science["points"])
                excluded = len(science["excluded"])
                units = {r.get("experimentalUnit") for r in panel["rows"]}
                independent = None if None in units else len(units)
            else:
                csv_rows = [
                    {
                        "panel": label,
                        "rowId": row["id"],
                        "sampleId": row.get("sampleId"),
                        "sampleLabel": row.get("sampleLabel"),
                        "capturedValues": row["values"],
                        "sourceReferences": row["lineage"],
                    }
                    for row in observations
                ]
                headers = [
                    "panel",
                    "rowId",
                    "sampleId",
                    "sampleLabel",
                    "capturedValues",
                    "sourceReferences",
                ]
                method = {
                    "chart": panel["legacySpec"],
                    "sampleField": panel.get("sampleField"),
                    "sourcePolicy": (
                        "Frozen legacy chart input; original upload bytes and"
                        " prior cleaning records are not reconstructed."
                    ),
                }
                analysis = {
                    "analysis": panel["analysis"],
                    "seriesColors": panel.get("seriesColors"),
                    "resultSha256": panel["resultSha256"],
                }
                reported_role = (
                    "Captured legacy input values, after the original "
                    "workspace's exclusions; not a claim of original upload "
                    "contents."
                )
                count, excluded, independent = len(observations), None, None
            files[f"panels/{label}-source.csv"] = csv_bytes(csv_rows, headers)
            files[f"panels/{label}-analysis.json"] = encoded(analysis)
            panels.append(
                {
                    "label": label,
                    "chartId": chart_id,
                    "revisionId": panel["revisionId"],
                    "resultSha256": panel["resultSha256"],
                    "kind": panel["kind"],
                    "observations": count,
                    "excluded": excluded,
                    "independentUnits": independent,
                    "unit": panel["unit"],
                    "dataRole": reported_role,
                    "method": method,
                    "csvColumns": headers,
                    "sourceCsvRows": len(csv_rows),
                }
            )
            legends_en.append(
                f"{label}: {panel['title'] or panel['kind']}; observations/input rows={count}; "
                f"independent units={independent if independent is not None else 'unknown'}; "
                f"unit={panel['unit']}. Panel counts may overlap."
            )
            legends_zh.append(
                f"{label}：{panel['title'] or panel['kind']}；观测／输入行={count}；"
                f"独立实验单位={independent if independent is not None else '未知'}；"
                f"单位={panel['unit']}。面板样本可能重复。"
            )
    # Notes describe executed operations; exact parameters remain inspectable in the manifest.
    for p in panels:
        method = p["method"]
        if "preparation" in method:
            plan = method["preparation"]
            steps = (
                "; ".join(
                    step["kind"] + (f" to {step['toUnit']}" if "toUnit" in step else "")
                    for step in plan["steps"]
                )
                or "none"
            )
            chart = method["chart"]
            detail = (
                f"summary={chart['summary']}; summaryUnit={chart['summaryUnit']}; "
                f"pairTimes={chart['pairTimes']}; bandwidth={chart['bandwidth']}; "
                f"densityScale={chart['densityScale']}"
            )
            methods_en.append(
                f"{p['label']}: {p['kind']}; preparation steps={steps}; "
                f"excluded observations={p['excluded']}; {detail}. "
                "Full parameters and source references: manifest.json."
            )
            methods_zh.append(
                f"{p['label']}：{p['kind']}；准备步骤={steps}；"
                f"排除观测={p['excluded']}；{detail}。"
                "完整参数及来源见 manifest.json。"
            )
        else:
            methods_en.append(
                f"{p['label']}: {p['kind']}; frozen legacy chart input and mappings; "
                "prior cleaning records are not reconstructed."
            )
            methods_zh.append(
                f"{p['label']}：{p['kind']}；使用冻结的旧图表输入与映射，不重建此前清洗记录。"
            )
        if p["kind"] == "violin":
            methods_en.append(
                f"{p['label']}: Gaussian KDE; density is not additional observations."
            )
            methods_zh.append(f"{p['label']}：高斯核密度不是额外观测。")
    notes = OutputNotes(
        legend_en="\n".join(legends_en),
        legend_zh="\n".join(legends_zh),
        methods_en="\n".join(methods_en),
        methods_zh="\n".join(methods_zh),
    )
    files["README.txt"] = (
        "LabViz research output / 研究输出\nThis ZIP is not a round-trip project backup. "
        "Original upload files are not included.\nScientific CSV: reported text/unit and "
        "derived values, including exclusions. Legacy CSV: captured chart input values "
        "only.\nIdentifiers, groups, source references, exclusion reasons and user notes "
        "may be included. CSV formula-like text is prefixed with an apostrophe; JSON "
        "retains recorded values.\nSelection overlays are not exported. No generated "
        "significance or causal conclusion.\n"
        "本包不是可恢复项目的备份。包含标识符、来源、排除理由及用户说明；"
        "不含原始上传文件。\n"
    ).encode()
    manifest = {
        "schemaVersion": 1,
        "role": "research-output-not-project-backup",
        "projectId": project_id,
        "boardId": board_id,
        "figureSpec": board["spec"],
        "exportRendering": rendering,
        "figureCreatedAt": board["createdAt"],
        "panels": panels,
        "originalUploadsIncluded": False,
        "identifiersIncluded": [
            "project/chart/revision IDs",
            "observation/sample IDs and labels",
            "group/time/batch",
            "source references",
            "exclusion reasons",
            "editable user notes",
        ],
        "csvFormulaTextEscaped": True,
    }
    return manifest, notes, files


def preview_output(store: ResearchStore, project_id: str, board_id: str) -> dict[str, Any]:
    manifest, notes, files = materialize(store, project_id, board_id)
    token = uuid4().hex
    with store.connect() as c:
        c.execute(
            (
                "INSERT INTO research_output_drafts VALUES (?,?,?,?) ON CONFLICT(project_id) "
                "DO UPDATE SET token=excluded.token,board_id=excluded.board_id,created_at=exc"
                "luded.created_at"
            ),
            (project_id, token, board_id, now()),
        )
    return {
        "token": token,
        "manifest": manifest,
        "notes": notes.model_dump(),
        "files": [
            {"name": name, "bytes": len(value), "sha256": sha(value)}
            for name, value in files.items()
        ]
        + [
            {"name": name, "bytes": None, "sha256": None}
            for name in ["manifest.json", "notes.en.txt", "notes.zh.txt"]
        ],
    }


def freeze_output(
    store: ResearchStore, project_id: str, token: str, notes: OutputNotes
) -> dict[str, Any]:
    with store.connect() as c:
        draft = c.execute(
            "SELECT * FROM research_output_drafts WHERE project_id=? AND token=?",
            (project_id, token),
        ).fetchone()
    if draft is None or datetime.fromisoformat(draft["created_at"]) < datetime.now(UTC) - timedelta(
        minutes=30
    ):
        raise PreparationError("preview-expired", "Preview the included output data again.")
    manifest, defaults, files = materialize(store, project_id, draft["board_id"])
    identifier, timestamp = uuid4().hex, now()
    files["notes.en.txt"] = (notes.legend_en + "\n\n" + notes.methods_en).encode("utf-8")
    files["notes.zh.txt"] = (notes.legend_zh + "\n\n" + notes.methods_zh).encode("utf-8")
    manifest.update(
        outputId=identifier,
        frozenAt=timestamp,
        userEditedNotes=notes != defaults,
        files=[
            {"name": name, "bytes": len(value), "sha256": sha(value)}
            for name, value in sorted(files.items())
        ],
    )
    files["manifest.json"] = encoded(manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    payload = stream.getvalue()
    with store.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        if (
            c.execute(
                "SELECT 1 FROM research_output_drafts WHERE project_id=? AND token=?",
                (project_id, token),
            ).fetchone()
            is None
        ):
            raise PreparationError(
                "preview-expired", "This output preview was already consumed or replaced."
            )
        c.execute(
            "INSERT INTO research_outputs VALUES (?,?,?,?,?,?,?,?)",
            (
                identifier,
                project_id,
                draft["board_id"],
                json.dumps(manifest, ensure_ascii=False),
                notes.model_dump_json(),
                payload,
                sha(payload),
                timestamp,
            ),
        )
        c.execute("DELETE FROM research_output_drafts WHERE project_id=?", (project_id,))
    return {
        "id": identifier,
        "boardId": draft["board_id"],
        "sha256": sha(payload),
        "createdAt": timestamp,
        "manifest": manifest,
        "notes": notes.model_dump(),
    }
