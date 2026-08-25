"""Tabular parsing, quality inspection, sampling, and figure rendering."""

from __future__ import annotations

import io
import json
import math
import re
import zlib
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast
from zipfile import BadZipFile, ZipFile

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import t as student_t

from .models import ChartSpec, JsonScalar
from .render_contract import (
    CONFIDENCE_BAND_OPACITY,
    GRID_COLOR,
    chart_series_color,
    matplotlib_line_style,
)

matplotlib.use("Agg")

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".xlsx"}
MAX_PREVIEW_ROWS = 200
MAX_PREVIEW_CELLS = 15_000
MAX_FINDING_ROW_IDS = 100
MAX_ANALYSIS_POINTS = 250
MAX_XLSX_ENTRIES = 5_000
MAX_XLSX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
UNIT_PATTERN = re.compile(r"^\s*(.*?)\s*(?:\(([^()]+)\)|\[([^\[\]]+)\])\s*$")


class ProcessingError(ValueError):
    def __init__(self, message: str, code: str = "processing-failed") -> None:
        super().__init__(message)
        self.code = code


def validate_upload(payload: bytes, filename: str, max_upload_bytes: int) -> str:
    if not payload:
        raise ProcessingError("The uploaded file is empty.", "empty-file")
    if len(payload) > max_upload_bytes:
        limit_mb = max_upload_bytes // (1024 * 1024)
        raise ProcessingError(f"The website accepts files up to {limit_mb} MB.", "file-too-large")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ProcessingError("Choose an XLSX, CSV, TSV, TXT, or JSON file.", "unsupported-format")
    return suffix


def _validate_xlsx_archive(payload: bytes) -> None:
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > MAX_XLSX_ENTRIES:
                raise ProcessingError(
                    "The workbook contains too many internal files.", "unsafe-workbook"
                )
            total_size = sum(member.file_size for member in members)
            if total_size > MAX_XLSX_UNCOMPRESSED_BYTES:
                raise ProcessingError(
                    "The expanded workbook is too large to process safely.", "unsafe-workbook"
                )
    except BadZipFile as exc:
        raise ProcessingError(
            "The workbook is unreadable or password protected.", "unreadable-workbook"
        ) from exc


def load_dataframe(
    payload: bytes,
    filename: str,
    *,
    requested_sheet_name: str | None = None,
    header_row: int = 1,
) -> tuple[pd.DataFrame, str | None, list[str], int]:
    suffix = Path(filename).suffix.lower()
    source = io.BytesIO(payload)
    sheet_name: str | None = None
    available_sheets: list[str] = []
    pandas_header = header_row - 1

    try:
        if suffix == ".csv":
            frame = pd.read_csv(source, header=pandas_header)
        elif suffix == ".tsv":
            frame = pd.read_csv(source, sep="\t", header=pandas_header)
        elif suffix == ".txt":
            frame = pd.read_csv(source, sep=None, engine="python", header=pandas_header)
        elif suffix == ".json":
            parsed = json.loads(payload.decode("utf-8-sig"))
            frame = pd.DataFrame(parsed)
        else:
            _validate_xlsx_archive(payload)
            workbook = pd.ExcelFile(source, engine="openpyxl")
            if not workbook.sheet_names:
                raise ProcessingError("The workbook has no worksheets.", "empty-workbook")
            available_sheets = workbook.sheet_names
            sheet_name = requested_sheet_name or workbook.sheet_names[0]
            if sheet_name not in workbook.sheet_names:
                raise ProcessingError(
                    f"The workbook has no worksheet named '{sheet_name}'.",
                    "sheet-not-found",
                )
            frame = workbook.parse(sheet_name=sheet_name, header=pandas_header)
    except ProcessingError:
        raise
    except UnicodeDecodeError as exc:
        raise ProcessingError(
            "The text file is not valid UTF-8 data.", "invalid-text-encoding"
        ) from exc
    except Exception as exc:
        raise ProcessingError(
            f"Could not read '{Path(filename).name}' as tabular data.", "unreadable-file"
        ) from exc

    if frame.columns.empty:
        raise ProcessingError("The file does not contain any columns.", "empty-table")
    if frame.empty:
        raise ProcessingError("The selected table contains no data rows.", "empty-table")
    if len(frame.columns) > 500:
        raise ProcessingError(
            "The table contains more than 500 columns. Split it before uploading.",
            "too-many-columns",
        )

    column_names = [
        str(column).strip() or f"Column {index + 1}" for index, column in enumerate(frame.columns)
    ]
    duplicates = [name for name, count in Counter(column_names).items() if count > 1]
    if duplicates:
        raise ProcessingError(
            f"Column names must be unique after trimming: {', '.join(sorted(duplicates))}",
            "duplicate-columns",
        )
    frame.columns = column_names
    return frame, sheet_name, available_sheets, header_row


def _column_label_and_unit(column: str) -> tuple[str, str | None]:
    match = UNIT_PATTERN.match(column)
    if not match:
        return column, None
    label = match.group(1).strip() or column
    unit = (match.group(2) or match.group(3) or "").strip() or None
    return label, unit


def _column_kind(series: pd.Series[Any]) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def _json_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, list | dict | tuple | set):
        return json.dumps(value, ensure_ascii=False, default=str)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, np.floating | float):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp | datetime | date):
        return value.isoformat()
    return str(value)


def build_preview(project_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    preview_row_limit = min(
        MAX_PREVIEW_ROWS,
        max(20, MAX_PREVIEW_CELLS // max(1, len(frame.columns))),
    )
    if len(frame) <= preview_row_limit:
        sampled_frame = frame
        sampled = False
    else:
        positions = np.linspace(0, len(frame) - 1, num=preview_row_limit, dtype=int)
        sampled_frame = frame.iloc[positions]
        sampled = True

    columns = []
    for column in frame.columns:
        label, unit = _column_label_and_unit(str(column))
        columns.append(
            {
                "field": str(column),
                "label": label,
                "kind": _column_kind(frame[column]),
                "unit": unit,
                "nullable": bool(frame[column].isna().any()),
            }
        )

    rows: list[dict[str, Any]] = []
    for index, row in sampled_frame.iterrows():
        item: dict[str, Any] = {"rowId": int(index) + 1 if isinstance(index, int) else str(index)}
        for column in frame.columns:
            item[str(column)] = _json_value(row[column])
        rows.append(item)

    return {
        "apiVersion": "v1",
        "projectId": project_id,
        "columns": columns,
        "rows": rows,
        "totalRows": len(frame),
        "sampled": sampled,
        "sampleStrategy": "evenly-distributed" if sampled else "none",
    }


def _finding_rows(mask: pd.Series[Any]) -> tuple[list[int], int, bool]:
    positions = np.flatnonzero(mask.to_numpy())
    affected_count = int(len(positions))
    row_ids = [int(position) + 1 for position in positions[:MAX_FINDING_ROW_IDS]]
    return row_ids, affected_count, affected_count > len(row_ids)


def _finding_message_fields(
    *,
    summary_code: str,
    summary_params: dict[str, JsonScalar] | None = None,
    reason_code: str,
    reason_params: dict[str, JsonScalar] | None = None,
) -> dict[str, Any]:
    """Return stable message identifiers while retaining readable v1 fallbacks."""

    return {
        "summaryCode": summary_code,
        "summaryParams": summary_params or {},
        "reasonCode": reason_code,
        "reasonParams": reason_params or {},
    }


def build_quality_report(
    project_id: str,
    frame: pd.DataFrame,
    valid_ranges: dict[str, tuple[float | None, float | None]] | None = None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    issue_mask = np.zeros(len(frame), dtype=bool)
    suspicious_mask = np.zeros(len(frame), dtype=bool)

    for column in frame.columns:
        missing_mask = frame[column].isna()
        if missing_mask.any():
            rows, affected_count, truncated = _finding_rows(missing_mask)
            issue_mask |= missing_mask.to_numpy()
            findings.append(
                {
                    "id": f"missing:{column}",
                    "kind": "missing",
                    "severity": "warning",
                    "column": str(column),
                    "rowIds": rows,
                    "affectedCount": affected_count,
                    "rowIdsTruncated": truncated,
                    "summary": (
                        f"{affected_count} missing value{'s' if affected_count != 1 else ''}"
                    ),
                    "reason": (
                        "These cells are empty. LabViz will not replace or remove them "
                        "without your decision."
                    ),
                    **_finding_message_fields(
                        summary_code="quality.missing.summary",
                        summary_params={"count": affected_count},
                        reason_code="quality.missing.reason",
                    ),
                }
            )

        series = frame[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_empty = series.dropna().astype(str).str.strip()
            if not non_empty.empty:
                numeric = pd.to_numeric(non_empty, errors="coerce")
                numeric_ratio = float(numeric.notna().mean())
                conflict_mask = series.notna() & pd.to_numeric(series, errors="coerce").isna()
                if 0.8 <= numeric_ratio < 1 and conflict_mask.any():
                    rows, affected_count, truncated = _finding_rows(conflict_mask)
                    mask_values = conflict_mask.to_numpy()
                    issue_mask |= mask_values
                    suspicious_mask |= mask_values
                    findings.append(
                        {
                            "id": f"type-conflict:{column}",
                            "kind": "type-conflict",
                            "severity": "warning",
                            "column": str(column),
                            "rowIds": rows,
                            "affectedCount": affected_count,
                            "rowIdsTruncated": truncated,
                            "summary": (
                                f"{affected_count} value"
                                f"{'s' if affected_count != 1 else ''} "
                                "do not match the numeric column"
                            ),
                            "reason": (
                                "Most values in this column are numeric, but these cells "
                                "contain other text."
                            ),
                            **_finding_message_fields(
                                summary_code="quality.type-conflict.summary",
                                summary_params={"count": affected_count},
                                reason_code="quality.type-conflict.reason",
                            ),
                        }
                    )

    duplicate_mask = frame.duplicated(keep=False)
    if duplicate_mask.any():
        rows, affected_count, truncated = _finding_rows(duplicate_mask)
        issue_mask |= duplicate_mask.to_numpy()
        findings.append(
            {
                "id": "duplicate:rows",
                "kind": "duplicate",
                "severity": "warning",
                "column": None,
                "rowIds": rows,
                "affectedCount": affected_count,
                "rowIdsTruncated": truncated,
                "summary": f"{affected_count} rows belong to duplicate groups",
                "reason": (
                    "These rows repeat all values. Confirm whether repetition is expected "
                    "before removing a cleaned copy."
                ),
                **_finding_message_fields(
                    summary_code="quality.duplicate.summary",
                    summary_params={"count": affected_count},
                    reason_code="quality.duplicate.reason",
                ),
            }
        )

    ranges = valid_ranges or {}
    unknown_ranges = sorted(column for column in ranges if column not in frame.columns)
    non_numeric_ranges = sorted(
        column
        for column in ranges
        if column in frame.columns and not pd.api.types.is_numeric_dtype(frame[column])
    )
    if unknown_ranges or non_numeric_ranges:
        invalid = [*unknown_ranges, *non_numeric_ranges]
        raise ProcessingError(
            f"Valid ranges require existing numeric columns: {', '.join(invalid)}",
            "invalid-quality-rule",
        )

    for column in frame.select_dtypes(include="number").columns:
        if column in ranges:
            minimum, maximum = ranges[column]
            outside_mask = pd.Series(False, index=frame.index)
            if minimum is not None:
                outside_mask |= frame[column] < minimum
            if maximum is not None:
                outside_mask |= frame[column] > maximum
            outside_mask &= frame[column].notna()
            if outside_mask.any():
                rows, affected_count, truncated = _finding_rows(outside_mask)
                mask_values = outside_mask.to_numpy()
                issue_mask |= mask_values
                suspicious_mask |= mask_values
                bounds = (
                    f"{minimum:g} to {maximum:g}"
                    if minimum is not None and maximum is not None
                    else f"at least {minimum:g}"
                    if minimum is not None
                    else f"at most {maximum:g}"
                )
                if minimum is not None and maximum is not None:
                    range_reason_code = "quality.outside-range.reason.both"
                    range_reason_params: dict[str, JsonScalar] = {
                        "minimum": minimum,
                        "maximum": maximum,
                    }
                elif minimum is not None:
                    range_reason_code = "quality.outside-range.reason.minimum"
                    range_reason_params = {"minimum": minimum}
                else:
                    range_reason_code = "quality.outside-range.reason.maximum"
                    range_reason_params = {"maximum": maximum}
                findings.append(
                    {
                        "id": f"outside-range:{column}",
                        "kind": "outside-range",
                        "severity": "warning",
                        "column": str(column),
                        "rowIds": rows,
                        "affectedCount": affected_count,
                        "rowIdsTruncated": truncated,
                        "summary": f"{affected_count} values are outside the valid range",
                        "reason": (
                            f"You defined {bounds} as valid for this column. "
                            "LabViz has only flagged the values for your review."
                        ),
                        "validMinimum": minimum,
                        "validMaximum": maximum,
                        **_finding_message_fields(
                            summary_code="quality.outside-range.summary",
                            summary_params={"count": affected_count},
                            reason_code=range_reason_code,
                            reason_params=range_reason_params,
                        ),
                    }
                )

        series = frame[column].dropna()
        if len(series) >= 8:
            first_quartile, third_quartile = series.quantile([0.25, 0.75])
            spread = third_quartile - first_quartile
            if math.isfinite(float(spread)) and spread > 0:
                lower = first_quartile - 1.5 * spread
                upper = third_quartile + 1.5 * spread
                extreme_mask = frame[column].notna() & (
                    (frame[column] < lower) | (frame[column] > upper)
                )
                if extreme_mask.any():
                    rows, affected_count, truncated = _finding_rows(extreme_mask)
                    mask_values = extreme_mask.to_numpy()
                    issue_mask |= mask_values
                    suspicious_mask |= mask_values
                    findings.append(
                        {
                            "id": f"extreme-value:{column}",
                            "kind": "extreme-value",
                            "severity": "warning",
                            "column": str(column),
                            "rowIds": rows,
                            "affectedCount": affected_count,
                            "rowIdsTruncated": truncated,
                            "summary": (
                                f"{affected_count} value"
                                f"{'s' if affected_count != 1 else ''} "
                                "are far from the middle range"
                            ),
                            "reason": (
                                "These values fall beyond 1.5 interquartile ranges. This is a "
                                "review flag, not proof that the measurements are wrong."
                            ),
                            **_finding_message_fields(
                                summary_code="quality.extreme-value.summary",
                                summary_params={"count": affected_count},
                                reason_code="quality.extreme-value.reason",
                            ),
                        }
                    )

        changes = frame[column].diff().abs()
        valid_changes = changes.dropna()
        if len(valid_changes) < 5:
            continue
        typical_change = float(valid_changes.median())
        deviation = float((valid_changes - typical_change).abs().median())
        threshold = max(
            typical_change + 6 * 1.4826 * deviation,
            typical_change * 6,
            1e-12,
        )
        sudden_change_mask = changes.notna() & (changes > threshold)
        if sudden_change_mask.any():
            rows, affected_count, truncated = _finding_rows(sudden_change_mask)
            mask_values = sudden_change_mask.to_numpy()
            issue_mask |= mask_values
            suspicious_mask |= mask_values
            findings.append(
                {
                    "id": f"sudden-change:{column}",
                    "kind": "sudden-change",
                    "severity": "warning",
                    "column": str(column),
                    "rowIds": rows,
                    "affectedCount": affected_count,
                    "rowIdsTruncated": truncated,
                    "summary": (
                        f"{affected_count} point"
                        f"{'s' if affected_count != 1 else ''} change "
                        "abruptly from the previous row"
                    ),
                    "reason": (
                        "The adjacent change is much larger than the typical change in this "
                        "column. Review the measurements before excluding anything."
                    ),
                    **_finding_message_fields(
                        summary_code="quality.sudden-change.summary",
                        summary_params={"count": affected_count},
                        reason_code="quality.sudden-change.reason",
                    ),
                }
            )

        trend_mask, trend_r_squared = _trend_inconsistent_mask(frame[column])
        if trend_mask.any():
            rows, affected_count, truncated = _finding_rows(
                pd.Series(trend_mask, index=frame.index)
            )
            issue_mask |= trend_mask
            suspicious_mask |= trend_mask
            findings.append(
                {
                    "id": f"trend-inconsistent:{column}",
                    "kind": "trend-inconsistent",
                    "severity": "warning",
                    "column": str(column),
                    "rowIds": rows,
                    "affectedCount": affected_count,
                    "rowIdsTruncated": truncated,
                    "summary": (
                        f"{affected_count} point"
                        f"{'s' if affected_count != 1 else ''} differ from the overall trend"
                    ),
                    "reason": (
                        "A strong linear trend was present across row order "
                        f"(R²={trend_r_squared:.3f}), and these residuals were unusually large. "
                        "This is a review flag, not proof that the measurements are wrong."
                    ),
                    **_finding_message_fields(
                        summary_code="quality.trend-inconsistent.summary",
                        summary_params={"count": affected_count},
                        reason_code="quality.trend-inconsistent.reason",
                        reason_params={"rSquared": f"{trend_r_squared:.3f}"},
                    ),
                }
            )

    return {
        "apiVersion": "v1",
        "projectId": project_id,
        "totalRows": len(frame),
        "validRows": max(0, len(frame) - int(np.count_nonzero(issue_mask))),
        "missingValues": int(frame.isna().sum().sum()),
        "duplicateRows": int(frame.duplicated().sum()),
        "suspiciousPoints": int(np.count_nonzero(suspicious_mask)),
        "findings": findings,
    }


def _trend_inconsistent_mask(series: pd.Series[Any]) -> tuple[np.ndarray[Any, Any], float]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna().to_numpy()
    positions = np.flatnonzero(valid).astype(float)
    values = numeric.iloc[np.flatnonzero(valid)].to_numpy(dtype=float)
    result = np.zeros(len(series), dtype=bool)
    if len(values) < 8 or len(np.unique(positions)) < 2:
        return result, 0.0
    coefficients = np.polyfit(positions, values, 1)
    fitted = np.polyval(coefficients, positions)
    total_variation = float(np.sum((values - values.mean()) ** 2))
    if total_variation <= 0:
        return result, 0.0
    residuals = values - fitted
    r_squared = 1 - float(np.sum(residuals**2)) / total_variation
    if r_squared < 0.8:
        return result, r_squared
    residual_median = float(np.median(residuals))
    median_deviation = float(np.median(np.abs(residuals - residual_median)))
    if median_deviation <= 1e-12:
        return result, r_squared
    unusual = np.abs(residuals - residual_median) > 4.5 * 1.4826 * median_deviation
    result[np.flatnonzero(valid)] = unusual
    return result, r_squared


def default_chart_spec(frame: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [str(column) for column in frame.select_dtypes(include="number").columns]
    if len(numeric_columns) >= 2:
        x_field, y_field = numeric_columns[:2]
    elif numeric_columns:
        x_field = y_field = numeric_columns[0]
    else:
        x_field = y_field = str(frame.columns[0])
    x_label, x_unit = _column_label_and_unit(x_field)
    y_label, y_unit = _column_label_and_unit(y_field)
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": f"{y_label} over {x_label}",
        "xAxis": {"field": x_field, "title": x_label, "unit": x_unit or ""},
        "yAxis": {"field": y_field, "title": y_label, "unit": y_unit or ""},
        "series": [{"field": y_field, "label": y_label, "color": "#2563EB"}],
        "panelCount": 1,
        "export": {
            "format": "png",
            "dpi": 300,
            "sizePreset": "double-column",
            "grayscalePreview": False,
        },
    }


def serialize_dataframe(frame: pd.DataFrame) -> bytes:
    raw = frame.to_json(orient="split", date_format="iso").encode("utf-8")
    return zlib.compress(raw, level=6)


def deserialize_dataframe(payload: bytes) -> pd.DataFrame:
    parsed = json.loads(zlib.decompress(payload).decode("utf-8"))
    return pd.DataFrame(data=parsed["data"], columns=parsed["columns"], index=parsed["index"])


def apply_chart_decisions(
    frame: pd.DataFrame,
    quality: dict[str, Any],
    decisions: list[dict[str, str]],
    actions: set[str] | None = None,
) -> pd.DataFrame:
    excluded_actions = actions or {"exclude", "remove"}
    excluded_findings = {
        item["findingId"] for item in decisions if item["action"] in excluded_actions
    }
    excluded_mask = np.zeros(len(frame), dtype=bool)
    for finding in quality["findings"]:
        if finding["id"] in excluded_findings:
            excluded_mask |= _quality_finding_mask(frame, finding)
    if not excluded_mask.any():
        return frame
    return frame.iloc[np.flatnonzero(~excluded_mask)]


def _quality_finding_mask(frame: pd.DataFrame, finding: dict[str, Any]) -> np.ndarray[Any, Any]:
    finding_id = str(finding["id"])
    kind, _, fallback_column = finding_id.partition(":")
    column = str(finding.get("column") or fallback_column)
    if kind == "duplicate" and column == "rows":
        return cast(np.ndarray[Any, Any], frame.duplicated(keep=False).to_numpy())
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    series = frame[column]
    if kind == "missing":
        return cast(np.ndarray[Any, Any], series.isna().to_numpy())
    if kind == "type-conflict":
        return cast(
            np.ndarray[Any, Any],
            (series.notna() & pd.to_numeric(series, errors="coerce").isna()).to_numpy(),
        )
    if kind == "extreme-value":
        numeric = series.dropna()
        first_quartile, third_quartile = numeric.quantile([0.25, 0.75])
        spread = third_quartile - first_quartile
        if not math.isfinite(float(spread)) or spread <= 0:
            return np.zeros(len(frame), dtype=bool)
        lower = first_quartile - 1.5 * spread
        upper = third_quartile + 1.5 * spread
        return cast(
            np.ndarray[Any, Any],
            (series.notna() & ((series < lower) | (series > upper))).to_numpy(),
        )
    if kind == "sudden-change":
        changes = series.diff().abs()
        valid_changes = changes.dropna()
        if len(valid_changes) < 5:
            return np.zeros(len(frame), dtype=bool)
        typical_change = float(valid_changes.median())
        deviation = float((valid_changes - typical_change).abs().median())
        threshold = max(typical_change + 6 * 1.4826 * deviation, typical_change * 6, 1e-12)
        return cast(np.ndarray[Any, Any], (changes.notna() & (changes > threshold)).to_numpy())
    if kind == "outside-range":
        minimum = finding.get("validMinimum")
        maximum = finding.get("validMaximum")
        outside = pd.Series(False, index=frame.index)
        if minimum is not None:
            outside |= series < float(minimum)
        if maximum is not None:
            outside |= series > float(maximum)
        outside &= series.notna()
        return cast(np.ndarray[Any, Any], outside.to_numpy())
    if kind == "trend-inconsistent":
        return _trend_inconsistent_mask(series)[0]
    return np.zeros(len(frame), dtype=bool)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ProcessingError(
            f"Chart fields are missing: {', '.join(missing)}", "invalid-chart-fields"
        )
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ProcessingError(
            f"Numeric fields are required: {', '.join(non_numeric)}", "invalid-chart-fields"
        )


def validate_chart_fields(frame: pd.DataFrame, chart: ChartSpec) -> None:
    fields = {chart.x_axis.field, chart.y_axis.field}
    fields.update(item.field for item in chart.series)
    if chart.group_field:
        fields.add(chart.group_field)
    if chart.uncertainty.error_field:
        fields.add(chart.uncertainty.error_field)
    if chart.secondary_y_axis.field:
        fields.add(chart.secondary_y_axis.field)
    missing = sorted(field for field in fields if field not in frame.columns)
    if missing:
        raise ProcessingError(
            f"Chart fields are missing: {', '.join(missing)}", "invalid-chart-fields"
        )

    if chart.group_field and chart.type not in {"line", "scatter", "bar"}:
        raise ProcessingError(
            "Grouping is supported for line, scatter, and bar charts.",
            "invalid-group-field",
        )

    if chart.type in {"line", "scatter", "bar", "surface3d"} and any(
        item.field == chart.x_axis.field for item in chart.series
    ):
        raise ProcessingError(
            "The X field must be different from every response field.",
            "invalid-chart-fields",
        )

    numeric_fields = [item.field for item in chart.series]
    if chart.type == "surface3d":
        numeric_fields.append(chart.x_axis.field)
    if chart.fitting.model != "none" or chart.uncertainty.mode != "none":
        numeric_fields.append(chart.x_axis.field)
    if chart.uncertainty.error_field:
        numeric_fields.append(chart.uncertainty.error_field)
    _numeric(frame, sorted(set(numeric_fields)))


def _sample_positions(length: int, limit: int = MAX_ANALYSIS_POINTS) -> np.ndarray[Any, Any]:
    if length <= limit:
        return np.arange(length)
    return np.linspace(0, length - 1, num=limit, dtype=int)


def _fit_equation(model: str, coefficients: np.ndarray[Any, Any]) -> str:
    if model == "exponential":
        return f"y = {math.exp(float(coefficients[1])):.5g}·e^({coefficients[0]:.5g}x)"
    if model == "logarithmic":
        return f"y = {coefficients[0]:.5g}·ln(x) + {coefficients[1]:.5g}"
    if model == "power":
        return f"y = {math.exp(float(coefficients[1])):.5g}·x^{coefficients[0]:.5g}"
    degree = len(coefficients) - 1
    terms = []
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        if power == 0:
            terms.append(f"{coefficient:.5g}")
        elif power == 1:
            terms.append(f"{coefficient:.5g}x")
        else:
            terms.append(f"{coefficient:.5g}x^{power}")
    return "y = " + " + ".join(terms).replace("+ -", "− ")


def _fit_analysis(
    frame: pd.DataFrame,
    x_field: str,
    y_field: str,
    chart: ChartSpec,
) -> tuple[dict[str, Any] | None, list[str]]:
    fitting = chart.fitting
    if fitting.model == "none":
        return None, []
    _numeric(frame, [x_field, y_field])
    complete = frame[[x_field, y_field]].dropna()
    x = complete[x_field].to_numpy(dtype=float)
    y = complete[y_field].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    warnings: list[str] = []

    if fitting.model in {"exponential", "power"}:
        positive_y = y > 0
        if not positive_y.all():
            warnings.append("Non-positive Y values were excluded from this fit.")
        x, y = x[positive_y], y[positive_y]
    if fitting.model in {"logarithmic", "power"}:
        positive_x = x > 0
        if not positive_x.all():
            warnings.append("Non-positive X values were excluded from this fit.")
        x, y = x[positive_x], y[positive_x]

    degree = fitting.polynomial_order if fitting.model == "polynomial" else 1
    parameter_count = degree + 1
    if len(x) <= parameter_count or len(np.unique(x)) <= degree:
        raise ProcessingError(
            f"{fitting.model.title()} fitting needs more distinct complete points.",
            "fit-not-suitable",
        )

    transformed_x = np.log(x) if fitting.model in {"logarithmic", "power"} else x
    transformed_y = np.log(y) if fitting.model in {"exponential", "power"} else y
    design = np.vander(transformed_x, N=parameter_count, increasing=False)
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design, transformed_y, rcond=None)
    if rank < parameter_count:
        raise ProcessingError(
            "The selected points cannot support a stable fitted curve.", "fit-not-suitable"
        )

    x_grid = np.linspace(float(x.min()), float(x.max()), num=MAX_ANALYSIS_POINTS)
    transformed_grid = np.log(x_grid) if fitting.model in {"logarithmic", "power"} else x_grid
    grid_design = np.vander(transformed_grid, N=parameter_count, increasing=False)
    fitted_transformed = design @ coefficients
    grid_transformed = grid_design @ coefficients
    fitted_y = (
        np.exp(fitted_transformed)
        if fitting.model in {"exponential", "power"}
        else fitted_transformed
    )
    grid_y = (
        np.exp(grid_transformed) if fitting.model in {"exponential", "power"} else grid_transformed
    )
    total_variation = float(np.sum((y - y.mean()) ** 2))
    if total_variation <= 0:
        raise ProcessingError(
            "A fitted curve is not meaningful when all Y values are identical.",
            "fit-not-suitable",
        )
    r_squared = 1 - float(np.sum((y - fitted_y) ** 2)) / total_variation

    lower: np.ndarray[Any, Any] | None = None
    upper: np.ndarray[Any, Any] | None = None
    degrees_of_freedom = len(x) - parameter_count
    if fitting.confidence_band:
        residuals = transformed_y - fitted_transformed
        residual_variance = float(residuals @ residuals) / degrees_of_freedom
        covariance = residual_variance * np.linalg.pinv(design.T @ design)
        variance = np.einsum("ij,jk,ik->i", grid_design, covariance, grid_design)
        critical = float(
            student_t.ppf((1 + fitting.confidence_level / 100) / 2, degrees_of_freedom)
        )
        delta = critical * np.sqrt(np.maximum(variance, 0))
        if fitting.model in {"exponential", "power"}:
            lower, upper = np.exp(grid_transformed - delta), np.exp(grid_transformed + delta)
        else:
            lower, upper = grid_y - delta, grid_y + delta

    points = []
    for index, (x_value, y_value) in enumerate(zip(x_grid, grid_y, strict=True)):
        points.append(
            {
                "x": float(x_value),
                "y": float(y_value),
                "lower": float(lower[index]) if lower is not None else None,
                "upper": float(upper[index]) if upper is not None else None,
            }
        )
    return (
        {
            "model": fitting.model,
            "equation": _fit_equation(fitting.model, coefficients),
            "rSquared": r_squared,
            "points": points,
        },
        warnings,
    )


def _uncertainty_analysis(
    frame: pd.DataFrame,
    x_field: str,
    y_field: str,
    chart: ChartSpec,
) -> tuple[dict[str, Any] | None, list[str]]:
    uncertainty = chart.uncertainty
    if uncertainty.mode == "none":
        return None, []
    fields = [x_field, y_field]
    if uncertainty.mode == "column" and uncertainty.error_field:
        fields.append(uncertainty.error_field)
    _numeric(frame, fields)
    warnings: list[str] = []

    if uncertainty.mode == "column":
        error_field = uncertainty.error_field
        if error_field is None:
            raise ProcessingError("Choose an error column.", "invalid-error-field")
        complete = frame[[x_field, y_field, error_field]].dropna()
        positions = _sample_positions(len(complete))
        complete = complete.iloc[positions]
        points = [
            {
                "x": float(row[x_field]),
                "y": float(row[y_field]),
                "error": abs(float(row[error_field])),
            }
            for _, row in complete.iterrows()
            if all(math.isfinite(float(row[field])) for field in fields)
        ]
    else:
        complete = frame[[x_field, y_field]].dropna()
        grouped = complete.groupby(x_field, sort=True)[y_field].agg(["mean", "std", "count"])
        grouped = grouped[grouped["count"] >= 2].dropna()
        if grouped.empty:
            warnings.append(
                "At least two measurements at the same X value are needed "
                "for calculated uncertainty."
            )
            return None, warnings
        if uncertainty.mode == "standard-deviation":
            errors = grouped["std"]
        else:
            standard_error = grouped["std"] / np.sqrt(grouped["count"])
            if uncertainty.mode == "standard-error":
                errors = standard_error
            else:
                probability = (1 + uncertainty.confidence_level / 100) / 2
                critical = student_t.ppf(probability, grouped["count"] - 1)
                errors = standard_error * critical
        positions = _sample_positions(len(grouped))
        selected = grouped.iloc[positions]
        selected_errors = errors.iloc[positions]
        points = [
            {"x": float(index), "y": float(row["mean"]), "error": float(error)}
            for (index, row), error in zip(
                selected.iterrows(), selected_errors.to_numpy(), strict=True
            )
            if math.isfinite(float(error))
        ]

    if not points:
        warnings.append("No complete uncertainty values are available for this series.")
        return None, warnings
    return {"mode": uncertainty.mode, "points": points}, warnings


def analyze_chart(frame: pd.DataFrame, chart: ChartSpec) -> dict[str, Any]:
    validate_chart_fields(frame, chart)
    x_field = chart.x_axis.field
    analyses = []
    grouped_frames: list[tuple[Any, pd.DataFrame]] = [(None, frame)]
    if chart.group_field:
        grouped_frames = []
        complete_groups = frame[chart.group_field].dropna()
        group_values = sorted(complete_groups.unique().tolist(), key=lambda value: str(value))
        if len(group_values) > 12:
            raise ProcessingError(
                "The selected grouping field has more than 12 groups. "
                "Filter or combine groups first.",
                "too-many-groups",
            )
        for group_value in group_values:
            grouped_frames.append((group_value, frame[frame[chart.group_field] == group_value]))
    for series in chart.series:
        for group_value, series_frame in grouped_frames:
            fit: dict[str, Any] | None = None
            uncertainty: dict[str, Any] | None = None
            fit_warnings: list[str] = []
            uncertainty_warnings: list[str] = []
            if chart.type in {"line", "scatter", "bar"}:
                try:
                    fit, fit_warnings = _fit_analysis(series_frame, x_field, series.field, chart)
                except ProcessingError as exc:
                    fit_warnings = [str(exc)]
                try:
                    uncertainty, uncertainty_warnings = _uncertainty_analysis(
                        series_frame, x_field, series.field, chart
                    )
                except ProcessingError as exc:
                    uncertainty_warnings = [str(exc)]
            group_label = str(group_value) if group_value is not None else None
            analyses.append(
                {
                    "field": series.field,
                    "label": (
                        f"{series.label} · {group_label}"
                        if group_label is not None
                        else series.label
                    ),
                    "panel": series.panel,
                    "group": _json_value(group_value),
                    "points": _series_preview_points(series_frame, x_field, series.field),
                    "fit": fit,
                    "uncertainty": uncertainty,
                    "warnings": [*fit_warnings, *uncertainty_warnings],
                }
            )
    return {"series": analyses, "preview": _derived_chart_preview(frame, chart)}


def _series_preview_points(frame: pd.DataFrame, x_field: str, y_field: str) -> list[dict[str, Any]]:
    if x_field not in frame.columns or y_field not in frame.columns:
        return []
    if x_field == y_field:
        numeric = pd.to_numeric(frame[y_field], errors="coerce").dropna()
        numeric = numeric.iloc[_sample_positions(len(numeric))]
        return [
            {"x": _json_value(value), "y": float(value)}
            for value in numeric
            if math.isfinite(float(value))
        ]
    complete = frame[[x_field, y_field]].dropna()
    numeric_y = pd.to_numeric(complete[y_field], errors="coerce")
    complete = complete[numeric_y.notna()].copy()
    complete[y_field] = numeric_y[numeric_y.notna()]
    complete = complete.iloc[_sample_positions(len(complete))]
    return [
        {"x": _json_value(row[x_field]), "y": float(row[y_field])}
        for _, row in complete.iterrows()
        if math.isfinite(float(row[y_field]))
    ]


def _derived_chart_preview(frame: pd.DataFrame, chart: ChartSpec) -> dict[str, Any]:
    histograms: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    heatmaps: list[dict[str, Any]] = []
    surface_points: list[dict[str, Any]] = []

    if chart.type == "histogram":
        for item in chart.series:
            _numeric(frame, [item.field])
            values = frame[item.field].dropna().to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                raise ProcessingError("A histogram needs at least one value.", "empty-chart-data")
            counts, edges = np.histogram(values, bins="auto")
            histograms.append(
                {
                    "field": item.field,
                    "label": item.label,
                    "bins": [
                        {
                            "start": float(edges[index]),
                            "end": float(edges[index + 1]),
                            "count": int(count),
                        }
                        for index, count in enumerate(counts)
                    ],
                }
            )
    elif chart.type == "box":
        for item in chart.series:
            _numeric(frame, [item.field])
            values = frame[item.field].dropna().to_numpy(dtype=float)
            values = np.sort(values[np.isfinite(values)])
            if not len(values):
                raise ProcessingError("Each box needs at least one value.", "empty-chart-data")
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            spread = q3 - q1
            included = values[(values >= q1 - 1.5 * spread) & (values <= q3 + 1.5 * spread)]
            outliers = values[(values < included[0]) | (values > included[-1])]
            boxes.append(
                {
                    "field": item.field,
                    "label": item.label,
                    "minimum": float(included[0]),
                    "q1": float(q1),
                    "median": float(median),
                    "q3": float(q3),
                    "maximum": float(included[-1]),
                    "outliers": [
                        float(value) for value in outliers[_sample_positions(len(outliers), 100)]
                    ],
                }
            )
    elif chart.type == "heatmap":
        for panel in range(1, chart.panel_count + 1):
            items = [item for item in chart.series if item.panel == panel]
            fields = [item.field for item in items]
            _numeric(frame, fields)
            if len(fields) < 2:
                raise ProcessingError(
                    "Each correlation heatmap needs at least two numeric series.",
                    "invalid-chart-fields",
                )
            correlation = frame[fields].corr().to_numpy(dtype=float)
            heatmaps.append(
                {
                    "panel": panel,
                    "labels": [item.label for item in items],
                    "matrix": [
                        [float(value) if math.isfinite(value) else None for value in row]
                        for row in correlation
                    ],
                }
            )
    elif chart.type == "surface3d":
        for panel in range(1, chart.panel_count + 1):
            items = [item for item in chart.series if item.panel == panel]
            if len(items) < 2:
                raise ProcessingError(
                    "Each 3D surface needs Y and Z series in addition to the X field.",
                    "invalid-chart-fields",
                )
            y_field, z_field = items[0].field, items[1].field
            _numeric(frame, [chart.x_axis.field, y_field, z_field])
            complete = frame[[chart.x_axis.field, y_field, z_field]].dropna()
            complete = complete.iloc[_sample_positions(len(complete), 2_000)]
            surface_points.extend(
                {
                    "panel": panel,
                    "x": float(row[chart.x_axis.field]),
                    "y": float(row[y_field]),
                    "z": float(row[z_field]),
                }
                for _, row in complete.iterrows()
                if all(
                    math.isfinite(float(row[field]))
                    for field in (chart.x_axis.field, y_field, z_field)
                )
            )
    return {
        "histograms": histograms,
        "boxes": boxes,
        "heatmaps": heatmaps,
        "surfacePoints": surface_points,
    }


def render_chart(frame: pd.DataFrame, chart: ChartSpec) -> bytes:
    if frame.empty:
        raise ProcessingError("There are no rows available to export.", "empty-chart-data")
    export = chart.export_settings
    sizes = {
        "single-column": (3.35, 2.6),
        "double-column": (7.1, 4.4),
        "a4": (8.27, 11.69),
        "custom": (6.0, 4.0),
    }
    figure_size = sizes[export.size_preset]
    if export.size_preset == "custom":
        assert export.width is not None and export.height is not None
        unit_scale = {"in": 1.0, "cm": 1 / 2.54, "mm": 1 / 25.4}[export.unit]
        figure_size = (export.width * unit_scale, export.height * unit_scale)
    if figure_size[0] * figure_size[1] * export.dpi**2 > 50_000_000:
        raise ProcessingError(
            "The requested dimensions and DPI would use too much memory. Choose a smaller size.",
            "figure-too-large",
        )

    figure = Figure(figsize=figure_size, dpi=export.dpi)
    column_count = 1 if chart.panel_count == 1 else 2
    row_count = math.ceil(chart.panel_count / column_count)
    axes = [
        figure.add_subplot(
            row_count,
            column_count,
            index + 1,
            projection="3d" if chart.type == "surface3d" else None,
        )
        for index in range(chart.panel_count)
    ]
    colors = [
        chart_series_color(
            configured_color=item.color,
            grayscale=export.grayscale_preview,
            grouped=False,
            index=index,
        )
        for index, item in enumerate(chart.series)
    ]
    x_field = chart.x_axis.field
    derived = analyze_chart(frame, chart)
    analyses = derived["series"]
    preview = derived["preview"]

    for panel_index, axis in enumerate(axes, start=1):
        panel_series = [
            (series_index, item)
            for series_index, item in enumerate(chart.series)
            if item.panel == panel_index
        ]
        if not panel_series:
            axis.set_axis_off()
            continue
        secondary_axis: Any | None = None
        if (
            chart.secondary_y_axis.enabled
            and chart.type in {"line", "scatter", "bar"}
            and any(item.y_axis == "secondary" for _, item in panel_series)
        ):
            secondary_axis = axis.twinx()

        if chart.type in {"line", "scatter", "bar"}:
            if x_field not in frame.columns:
                raise ProcessingError(f"Chart field is missing: {x_field}", "invalid-chart-fields")
            panel_analyses = [
                (analysis_index, analysis)
                for analysis_index, analysis in enumerate(analyses)
                if analysis["panel"] == panel_index
            ]
            for order, (analysis_index, analysis) in enumerate(panel_analyses):
                item = next(
                    series for _, series in panel_series if series.field == analysis["field"]
                )
                points = analysis["points"]
                if not points:
                    continue
                target_axis = (
                    secondary_axis
                    if secondary_axis is not None and item.y_axis == "secondary"
                    else axis
                )
                color = chart_series_color(
                    configured_color=item.color,
                    grayscale=export.grayscale_preview,
                    grouped=bool(chart.group_field),
                    index=analysis_index,
                )
                label = analysis["label"]
                style = matplotlib_line_style(item.line_style)
                x_values = [point["x"] for point in points]
                y_values = [point["y"] for point in points]
                if chart.type == "line":
                    target_axis.plot(
                        x_values,
                        y_values,
                        label=label,
                        color=color,
                        linestyle=style,
                        linewidth=export.line_width,
                        marker="o",
                        markersize=export.marker_size,
                    )
                elif chart.type == "scatter":
                    target_axis.scatter(
                        x_values,
                        y_values,
                        label=label,
                        color=color,
                        alpha=0.8,
                        s=export.marker_size**2,
                    )
                else:
                    target_axis.bar(
                        x_values,
                        y_values,
                        label=label,
                        color=color,
                        alpha=max(0.45, 0.85 - order * 0.08),
                        linewidth=export.line_width,
                    )

                uncertainty = analysis["uncertainty"]
                if uncertainty:
                    points = uncertainty["points"]
                    target_axis.errorbar(
                        [point["x"] for point in points],
                        [point["y"] for point in points],
                        yerr=[point["error"] for point in points],
                        fmt="none",
                        ecolor=color,
                        elinewidth=export.line_width,
                        capsize=max(2, export.marker_size / 2),
                        alpha=0.9,
                    )
                fit = analysis["fit"]
                if fit:
                    fit_label = f"{label} fit"
                    if chart.fitting.show_equation:
                        fit_label += f" · {fit['equation']}"
                    if chart.fitting.show_r_squared:
                        fit_label += f" · R²={fit['rSquared']:.4f}"
                    points = fit["points"]
                    target_axis.plot(
                        [point["x"] for point in points],
                        [point["y"] for point in points],
                        color=color,
                        linestyle="--",
                        linewidth=max(export.line_width, 1),
                        label=fit_label,
                    )
                    if chart.fitting.confidence_band:
                        band = [
                            point
                            for point in points
                            if point["lower"] is not None and point["upper"] is not None
                        ]
                        if band:
                            target_axis.fill_between(
                                [point["x"] for point in band],
                                [point["lower"] for point in band],
                                [point["upper"] for point in band],
                                color=color,
                                alpha=CONFIDENCE_BAND_OPACITY,
                            )
            axis.set_xlabel(_axis_label(chart.x_axis.title, chart.x_axis.unit))
            axis.set_ylabel(_axis_label(chart.y_axis.title, chart.y_axis.unit))
            if secondary_axis is not None:
                secondary_axis.set_ylabel(
                    _axis_label(
                        chart.secondary_y_axis.title,
                        chart.secondary_y_axis.unit,
                    )
                )
        elif chart.type == "histogram":
            for series_index, item in panel_series:
                histogram = next(
                    value
                    for value in preview["histograms"]
                    if value["field"] == item.field and value["label"] == item.label
                )
                bins = histogram["bins"]
                axis.stairs(
                    [value["count"] for value in bins],
                    [bins[0]["start"], *[value["end"] for value in bins]],
                    fill=True,
                    alpha=0.55,
                    label=item.label,
                    color=colors[series_index],
                    linewidth=export.line_width,
                )
            axis.set_xlabel(_axis_label(chart.y_axis.title, chart.y_axis.unit))
            axis.set_ylabel("Frequency")
        elif chart.type == "box":
            stats = []
            for _, item in panel_series:
                box = next(
                    value
                    for value in preview["boxes"]
                    if value["field"] == item.field and value["label"] == item.label
                )
                stats.append(
                    {
                        "label": item.label,
                        "whislo": box["minimum"],
                        "q1": box["q1"],
                        "med": box["median"],
                        "q3": box["q3"],
                        "whishi": box["maximum"],
                        "fliers": box["outliers"],
                    }
                )
            axis.bxp(stats)
            axis.set_ylabel(_axis_label(chart.y_axis.title, chart.y_axis.unit))
        elif chart.type == "heatmap":
            heatmap = next(value for value in preview["heatmaps"] if value["panel"] == panel_index)
            correlations = np.array(
                [
                    [float("nan") if value is None else value for value in row]
                    for row in heatmap["matrix"]
                ],
                dtype=float,
            )
            image = axis.imshow(
                correlations,
                cmap="Greys" if export.grayscale_preview else "coolwarm",
                vmin=-1,
                vmax=1,
            )
            labels = heatmap["labels"]
            ticks = np.arange(len(labels))
            axis.set_xticks(ticks, labels, rotation=45, ha="right")
            axis.set_yticks(ticks, labels)
            figure.colorbar(image, ax=axis, label="Correlation")
        else:
            if len(panel_series) < 2:
                raise ProcessingError(
                    "Each 3D surface needs Y and Z series in addition to the X field.",
                    "invalid-chart-fields",
                )
            surface_points = [
                point for point in preview["surfacePoints"] if point["panel"] == panel_index
            ]
            if len(surface_points) < 3:
                raise ProcessingError(
                    "A 3D surface needs at least three points.", "empty-chart-data"
                )
            try:
                surface = cast(Any, axis).plot_trisurf(
                    [point["x"] for point in surface_points],
                    [point["y"] for point in surface_points],
                    [point["z"] for point in surface_points],
                    cmap="Greys" if export.grayscale_preview else "viridis",
                )
            except (RuntimeError, ValueError) as exc:
                raise ProcessingError(
                    "The selected 3D points do not span a surface.", "invalid-chart-fields"
                ) from exc
            axis.set_xlabel(_axis_label(chart.x_axis.title, chart.x_axis.unit))
            axis.set_ylabel(panel_series[0][1].label)
            cast(Any, axis).set_zlabel(panel_series[1][1].label)
            figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.1)

        if chart.panel_count == 1:
            axis.set_title(f"{chart.title}\n{chart.subtitle}" if chart.subtitle else chart.title)
        else:
            axis.set_title(f"Panel {panel_index}")
        if chart.type not in {"box", "heatmap", "surface3d"}:
            axis.grid(
                export.grid_visible,
                color=GRID_COLOR,
                linewidth=0.7,
                alpha=0.8,
            )
        _configure_legend(axis, export.legend_position)
        if secondary_axis is not None:
            _configure_legend(secondary_axis, export.legend_position)

    if chart.panel_count > 1:
        figure.suptitle(
            f"{chart.title}\n{chart.subtitle}" if chart.subtitle else chart.title,
            fontsize=export.font_size + 2,
        )
    for text in figure.findobj(match=lambda item: hasattr(item, "set_fontfamily")):
        styled_text = cast(Any, text)
        styled_text.set_fontfamily(export.font_family)
        if hasattr(styled_text, "get_fontsize") and styled_text.get_fontsize() == 10:
            styled_text.set_fontsize(export.font_size)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(
        output,
        format=export.format,
        dpi=export.dpi,
        bbox_inches="tight",
        facecolor=("none" if export.transparent_background else export.background_color),
        transparent=export.transparent_background,
    )
    return output.getvalue()


def _configure_legend(axis: Any, position: str) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if not handles or position == "none":
        return
    locations = {
        "auto": "best",
        "top": "upper center",
        "bottom": "lower center",
        "left": "center left",
        "right": "center right",
    }
    axis.legend(handles, labels, loc=locations[position])


def _axis_label(title: str, unit: str) -> str:
    return f"{title} ({unit})" if unit else title


def sample_csv_bytes() -> bytes:
    rows = ["Time (min),Response (mV),Temperature (°C)"]
    for index in range(120):
        time = index * 0.5
        response = 18 + 0.62 * time + 3.4 * math.sin(time / 4.8)
        if index == 37:
            response += 18
        response_value = "" if index == 71 else f"{response:.2f}"
        temperature = 23.2 + 0.25 * math.sin(time / 7)
        rows.append(f"{time:.1f},{response_value},{temperature:.2f}")
    return ("\n".join(rows) + "\n").encode("utf-8")
