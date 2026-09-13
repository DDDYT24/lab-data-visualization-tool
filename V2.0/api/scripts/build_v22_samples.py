"""Build the tracked, synthetic V2.2 example and edge-fixture corpus.

The generated files are intentionally small and contain no experiment data.  Keep this builder
deterministic so the manifest and answer-keyed tests can be regenerated during release work.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1] / "samples" / "v22"


def write_csv(name: str, headers: list[str], rows: list[list[Any]], *, bom: bool = False) -> None:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if bom else "utf-8"
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def write_json(name: str, rows: list[dict[str, Any]]) -> None:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_xlsx(name: str, sheets: dict[str, tuple[list[str], list[list[Any]]]]) -> None:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    first = True
    for sheet_name, (headers, rows) in sheets.items():
        sheet = workbook.active if first else workbook.create_sheet()
        first = False
        sheet.title = sheet_name
        sheet.append(headers)
        for row in rows:
            sheet.append(row)
    workbook.save(path)


def text(en: str, zh: str) -> dict[str, str]:
    return {"en": en, "zh": zh}


def field(name: str, kind: str, role: str, en: str, zh: str) -> dict[str, str]:
    return {"name": name, "kind": kind, "role": role, "titleEn": en, "titleZh": zh}


def public_entry(
    *,
    slug: str,
    filename: str,
    media_type: str,
    file_format: str,
    title_en: str,
    title_zh: str,
    purpose_en: str,
    purpose_zh: str,
    goal_en: str,
    goal_zh: str,
    chart_type: str,
    difficulty: str,
    fields: list[dict[str, str]],
    row_count: int,
    sheet_name: str | None = None,
    available_sheets: list[str] | None = None,
    quality_issues: list[dict[str, str]] | None = None,
    expected_quality_kinds: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "visibility": "public",
        "filename": filename,
        "mediaType": media_type,
        "format": file_format,
        "sheetName": sheet_name,
        "availableSheets": available_sheets or [],
        "headerRow": 1,
        "title": text(title_en, title_zh),
        "purpose": text(purpose_en, purpose_zh),
        "learningGoal": text(goal_en, goal_zh),
        "recommendedChart": chart_type,
        "difficulty": difficulty,
        "fields": fields,
        "rowCount": row_count,
        "qualityIssues": quality_issues or [],
        "expectedQualityKinds": expected_quality_kinds or [],
        "answerKey": {
            "previewRows": row_count,
            "recommendedChart": chart_type,
            "requiredQualityKinds": expected_quality_kinds or [],
            "exportFormats": ["png", "svg", "pdf"],
        },
        "synthetic": True,
    }


def edge_entry(
    *,
    slug: str,
    filename: str,
    media_type: str,
    file_format: str,
    expected: list[str],
) -> dict[str, Any]:
    return {
        "slug": slug,
        "visibility": "edge",
        "filename": filename,
        "mediaType": media_type,
        "format": file_format,
        "sheetName": None,
        "availableSheets": [],
        "headerRow": 1,
        "expectedErrors": expected,
        "answerKey": {"expectedErrors": expected},
        "synthetic": True,
    }


def build() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    time_rows = []
    for index in range(24):
        time = index * 0.5
        response = 18 + 0.62 * time + 3.4 * math.sin(time / 4.8)
        if index == 10:
            response += 15
        time_rows.append(
            [
                f"{time:.1f}",
                "" if index == 15 else f"{response:.2f}",
                f"{23.2 + 0.25 * math.sin(time / 7):.2f}",
            ]
        )
    write_csv(
        "01_time_series.csv",
        ["time_min", "response_mV", "temperature_C"],
        time_rows,
    )

    repeated_rows = []
    for condition, offset in (("Control", 0.0), ("Treatment", 3.5)):
        for replicate in range(1, 4):
            for time in (0, 5, 10, 15):
                mean = 10 + 0.28 * time + offset + (replicate - 2) * 0.4
                repeated_rows.append(
                    [
                        time,
                        round(mean, 3),
                        condition,
                        f"R{replicate}",
                        round(0.35 + replicate * 0.03, 3),
                    ]
                )
    write_xlsx(
        "02_repeated_runs.xlsx",
        {
            "Measurements": (
                ["time_min", "response_mV", "condition", "replicate", "error_sd_mV"],
                repeated_rows,
            ),
            "Metadata": (
                ["key", "value"],
                [["dataset", "Synthetic repeated experiment"], ["unit", "mV"]],
            ),
        },
    )

    scatter_rows = []
    for index, dose in enumerate(
        (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 28, 35, 45, 60, 80)
    ):
        response = 2.5 + 0.82 * dose - 0.002 * dose * dose + ((index % 3) - 1) * 0.35
        if dose == 35:
            response += 7
        scatter_rows.append([dose, round(response, 3), f"B{1 + index % 3}"])
    write_csv("03_dose_response.tsv", ["dose_uM", "response_mV", "batch"], scatter_rows)
    (ROOT / "03_dose_response.tsv").write_text(
        (ROOT / "03_dose_response.tsv").read_text(encoding="utf-8").replace(",", "\t"),
        encoding="utf-8",
    )

    category_rows = []
    for group, base in (("Control", 12), ("Low dose", 16), ("High dose", 22)):
        for replicate in range(1, 7):
            value = base + (replicate - 3.5) * 0.55
            if group == "High dose" and replicate == 6:
                value += 12
            category_rows.append([group, round(value, 2), f"B{1 + replicate % 2}"])
    write_csv("04_group_comparison.txt", ["group", "measurement", "batch"], category_rows)
    (ROOT / "04_group_comparison.txt").write_text(
        (ROOT / "04_group_comparison.txt").read_text(encoding="utf-8").replace(",", "\t"),
        encoding="utf-8",
    )

    distribution_rows = []
    for index in range(48):
        group = "Control" if index < 24 else "Treatment"
        center = 5.0 if group == "Control" else 5.8
        distribution_rows.append(
            {
                "sample_id": f"S{index + 1:02d}",
                "group": group,
                "measurement": round(center + ((index * 7) % 17 - 8) * 0.12, 3),
            }
        )
    write_json("05_distribution.json", distribution_rows)

    correlation_rows = []
    for index in range(30):
        temperature = 18 + index * 0.4
        signal = 3.2 + temperature * 0.18 + math.sin(index / 2) * 0.15
        mass = 40 + index * 1.8 + math.cos(index / 3) * 1.2
        recovery = 72 + signal * 2.4 - mass * 0.08 + math.sin(index) * 0.5
        correlation_rows.append(
            {
                "temperature_C": round(temperature, 3),
                "signal_mV": round(signal, 3),
                "mass_mg": round(mass, 3),
                "recovery_pct": round(recovery, 3),
            }
        )
    write_json("06_correlation.json", correlation_rows)

    surface_rows = []
    for y_index in range(21):
        for x_index in range(21):
            x = -5 + x_index * 0.5
            y = -5 + y_index * 0.5
            surface_rows.append([f"{x:.1f}", f"{y:.1f}", f"{x * x + y * y:.3f}"])
    write_csv("07_surface3d.csv", ["x_mm", "y_mm", "response_mV"], surface_rows)

    write_csv("edge_missing.csv", ["time_min", "response_mV"], [[0, 1.2], [1, ""], [2, 1.8]])
    write_csv(
        "edge_duplicate.csv", ["time_min", "response_mV"], [[0, 1.2], [1, 1.5], [1, 1.5], [2, 1.8]]
    )
    write_csv(
        "edge_mixed_types.csv",
        ["time_min", "response_mV"],
        [[0, 1.2], [1, 1.5], [2, "not-a-number"], [3, 1.9], [4, 2.1]],
    )
    write_csv(
        "edge_outlier.csv",
        ["time_min", "response_mV"],
        [[index, 1 + index * 0.1 if index < 19 else 90] for index in range(20)],
    )
    write_csv("edge_bom.txt", ["time_min", "response_mV"], [[0, 1.2], [1, 1.5]], bom=True)
    write_csv("edge_中文文件名.csv", ["时间_min", "信号_mV"], [[0, 1.2], [1, 1.5]])
    write_csv(
        "edge_surface_invalid.csv",
        ["x", "y", "z"],
        [[0, 0, 1], [1, 0, 2], [0, 1, 2], [1, 1, 3], [0, 0, 1]],
    )
    write_csv(
        "edge_surface_missing_grid.csv",
        ["x", "y", "z"],
        [[0, 0, 1], [1, 0, 2], [0, 1, 2]],
    )
    write_csv("edge_duplicate_columns.csv", ["time", "time"], [[0, 1], [1, 2]])

    public = [
        public_entry(
            slug="time-series",
            filename="01_time_series.csv",
            media_type="text/csv",
            file_format="csv",
            title_en="Time-series response",
            title_zh="时间响应曲线",
            purpose_en="Follow a measurement over time and inspect missing or unusual points.",
            purpose_zh="观察测量值随时间变化，并检查缺失或异常点。",
            goal_en="Learn when a line chart is appropriate and why quality findings matter.",
            goal_zh="理解什么时候适合用折线图，以及为什么要先检查数据质量。",
            chart_type="line",
            difficulty="beginner",
            fields=[
                field("time_min", "number", "x", "Time", "时间"),
                field("response_mV", "number", "y", "Response", "响应"),
                field("temperature_C", "number", "series", "Temperature", "温度"),
            ],
            row_count=len(time_rows),
            quality_issues=[
                text(
                    "One missing response and one unusual response are intentional.",
                    "故意包含一个缺失响应值和一个异常响应值。",
                )
            ],
            expected_quality_kinds=["missing", "extreme-value"],
        ),
        public_entry(
            slug="repeated-runs",
            filename="02_repeated_runs.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            file_format="xlsx",
            title_en="Repeated experiment runs",
            title_zh="重复实验运行",
            purpose_en=(
                "Compare control and treatment runs with replicate identifiers and error values."
            ),
            purpose_zh="比较对照和处理组，并保留重复编号与误差值。",
            goal_en=(
                "Learn how long-format repeated measurements support grouped lines and error bars."
            ),
            goal_zh="理解长表格式的重复测量如何支持分组折线和误差棒。",
            chart_type="line",
            difficulty="intermediate",
            fields=[
                field("time_min", "number", "x", "Time", "时间"),
                field("response_mV", "number", "y", "Response", "响应"),
                field("condition", "text", "group", "Condition", "条件"),
                field("replicate", "text", "label", "Replicate", "重复"),
                field("error_sd_mV", "number", "error", "SD", "标准差"),
            ],
            row_count=len(repeated_rows),
            sheet_name="Measurements",
            available_sheets=["Measurements", "Metadata"],
        ),
        public_entry(
            slug="scatter-fit",
            filename="03_dose_response.tsv",
            media_type="text/tab-separated-values",
            file_format="tsv",
            title_en="Dose-response scatter",
            title_zh="剂量-响应散点",
            purpose_en="Relate a numeric dose to a response and compare candidate fits.",
            purpose_zh="观察数值剂量与响应的关系，并比较候选拟合。",
            goal_en="Learn why an association and a fitted curve do not prove causality.",
            goal_zh="理解相关关系和拟合曲线并不等于因果关系。",
            chart_type="scatter",
            difficulty="intermediate",
            fields=[
                field("dose_uM", "number", "x", "Dose", "剂量"),
                field("response_mV", "number", "y", "Response", "响应"),
                field("batch", "text", "group", "Batch", "批次"),
            ],
            row_count=len(scatter_rows),
            quality_issues=[
                text(
                    "One high response is intentional so residual checks are visible.",
                    "故意包含一个偏高响应值，便于观察残差检查。",
                )
            ],
            expected_quality_kinds=["extreme-value"],
        ),
        public_entry(
            slug="categorical-comparison",
            filename="04_group_comparison.txt",
            media_type="text/plain",
            file_format="txt",
            title_en="Categorical group comparison",
            title_zh="分类分组比较",
            purpose_en="Compare distributions across named experimental groups.",
            purpose_zh="比较不同实验分组中的测量分布。",
            goal_en="Learn when grouped bars or box plots communicate categories clearly.",
            goal_zh="理解什么时候分组柱状图或箱线图更适合表达分类数据。",
            chart_type="box",
            difficulty="beginner",
            fields=[
                field("group", "text", "group", "Group", "分组"),
                field("measurement", "number", "y", "Measurement", "测量值"),
                field("batch", "text", "label", "Batch", "批次"),
            ],
            row_count=len(category_rows),
            quality_issues=[
                text(
                    "The high-dose group contains one intentionally unusual replicate.",
                    "高剂量组故意包含一个较异常的重复值。",
                )
            ],
            expected_quality_kinds=["extreme-value"],
        ),
        public_entry(
            slug="distribution",
            filename="05_distribution.json",
            media_type="application/json",
            file_format="json",
            title_en="Measurement distributions",
            title_zh="测量值分布",
            purpose_en="Inspect the spread and overlap of two synthetic measurement groups.",
            purpose_zh="观察两组模拟测量值的离散程度和重叠情况。",
            goal_en="Learn how histograms show distributions rather than individual time trends.",
            goal_zh="理解直方图展示的是分布，而不是单个时间趋势。",
            chart_type="histogram",
            difficulty="beginner",
            fields=[
                field("sample_id", "text", "label", "Sample ID", "样本编号"),
                field("group", "text", "group", "Group", "分组"),
                field("measurement", "number", "y", "Measurement", "测量值"),
            ],
            row_count=len(distribution_rows),
        ),
        public_entry(
            slug="correlation-heatmap",
            filename="06_correlation.json",
            media_type="application/json",
            file_format="json",
            title_en="Multi-variable correlation",
            title_zh="多变量相关性",
            purpose_en="Compare pairwise relationships among several numeric measurements.",
            purpose_zh="比较多个数值测量之间的两两关系。",
            goal_en="Learn that a heatmap summarizes associations and is not a causal model.",
            goal_zh="理解热力图概括的是关联关系，并不是因果模型。",
            chart_type="heatmap",
            difficulty="intermediate",
            fields=[
                field("temperature_C", "number", "series", "Temperature", "温度"),
                field("signal_mV", "number", "series", "Signal", "信号"),
                field("mass_mg", "number", "series", "Mass", "质量"),
                field("recovery_pct", "number", "series", "Recovery", "回收率"),
            ],
            row_count=len(correlation_rows),
        ),
        public_entry(
            slug="surface-3d",
            filename="07_surface3d.csv",
            media_type="text/csv",
            file_format="csv",
            title_en="Regular X/Y/Z surface",
            title_zh="规则 X/Y/Z 3D 曲面",
            purpose_en="Visualize a complete rectangular grid with X, Y, and measured Z values.",
            purpose_zh="可视化包含 X、Y 和测量 Z 值的完整矩形网格。",
            goal_en=(
                "Learn the difference between a valid structured surface and "
                "an irregular point cloud."
            ),
            goal_zh="理解有效结构化曲面与不规则散点云的区别。",
            chart_type="surface3d",
            difficulty="intermediate",
            fields=[
                field("x_mm", "number", "x", "X", "X"),
                field("y_mm", "number", "y", "Y", "Y"),
                field("response_mV", "number", "z", "Response", "响应"),
            ],
            row_count=len(surface_rows),
        ),
    ]
    edges = [
        edge_entry(
            slug="missing-values",
            filename="edge_missing.csv",
            media_type="text/csv",
            file_format="csv",
            expected=["missing"],
        ),
        edge_entry(
            slug="duplicate-rows",
            filename="edge_duplicate.csv",
            media_type="text/csv",
            file_format="csv",
            expected=["duplicate"],
        ),
        edge_entry(
            slug="mixed-types",
            filename="edge_mixed_types.csv",
            media_type="text/csv",
            file_format="csv",
            expected=["type-conflict"],
        ),
        edge_entry(
            slug="outlier",
            filename="edge_outlier.csv",
            media_type="text/csv",
            file_format="csv",
            expected=["extreme-value"],
        ),
        edge_entry(
            slug="bom-header",
            filename="edge_bom.txt",
            media_type="text/plain",
            file_format="txt",
            expected=["bom-normalized"],
        ),
        edge_entry(
            slug="unicode-filename",
            filename="edge_中文文件名.csv",
            media_type="text/csv",
            file_format="csv",
            expected=["unicode-download"],
        ),
        edge_entry(
            slug="invalid-surface-duplicate",
            filename="edge_surface_invalid.csv",
            media_type="text/csv",
            file_format="csv",
            expected=["surface-duplicate-coordinates"],
        ),
        edge_entry(
            slug="invalid-surface-missing-grid",
            filename="edge_surface_missing_grid.csv",
            media_type="text/csv",
            file_format="csv",
            expected=["surface-missing-grid"],
        ),
        edge_entry(
            slug="duplicate-columns",
            filename="edge_duplicate_columns.csv",
            media_type="text/csv",
            file_format="csv",
            expected=["duplicate-columns"],
        ),
    ]
    for entry in [*public, *edges]:
        entry["byteSize"] = (ROOT / entry["filename"]).stat().st_size
    statistics_answer_key = ROOT / "statistics_answer_keys.json"
    if not statistics_answer_key.is_file():
        raise RuntimeError(
            "statistics_answer_keys.json must be present before rebuilding the catalog"
        )
    statistics_payload = json.loads(statistics_answer_key.read_text(encoding="utf-8"))
    if not statistics_payload.get("syntheticOnly"):
        raise RuntimeError("The statistics answer key must be syntheticOnly")
    manifest = {
        "catalogVersion": "v1",
        "release": "V2.2",
        "syntheticOnly": True,
        "statisticsAnswerKeyFile": statistics_answer_key.name,
        "examples": [*public, *edges],
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
