"""Canonical Parquet v1 serialization for immutable DatasetVersion objects."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_SCHEMA_VERSION = 1
PARQUET_CONTENT_HASH = "sha256-parquet-bytes-v1"
SCHEMA_METADATA_KEY = b"labviz.schema"


class ParquetContractError(ValueError):
    """Raised when bytes do not satisfy the LabViz Parquet v1 contract."""


@dataclass(frozen=True)
class ParquetArtifact:
    payload: bytes
    sha256: str
    schema_document: dict[str, Any]
    row_count: int
    column_count: int


def _normalize_series(series: pd.Series[Any]) -> pd.Series[Any]:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.astype("Int64")
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.astype("Float64")
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return pd.to_datetime(series, utc=True).dt.floor("us")
    return series.astype("string")


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.empty:
        raise ParquetContractError("A DatasetVersion must contain at least one column.")
    normalized = frame.convert_dtypes().copy()
    normalized.columns = [str(column) for column in normalized.columns]
    if len(set(normalized.columns)) != len(normalized.columns):
        raise ParquetContractError("Parquet column names must be unique.")
    for column in normalized.columns:
        normalized[column] = _normalize_series(normalized[column])
    return normalized


def _schema_document(table: pa.Table, units: dict[str, str | None]) -> dict[str, Any]:
    columns = []
    for field in table.schema:
        columns.append(
            {
                "name": field.name,
                "arrowType": str(field.type),
                "nullable": field.nullable,
                "unit": units.get(field.name),
            }
        )
    return {
        "schemaVersion": PARQUET_SCHEMA_VERSION,
        "contentHash": PARQUET_CONTENT_HASH,
        "missingValues": "arrow-null",
        "timestampTimezone": "UTC",
        "columns": columns,
    }


def write_parquet(
    frame: pd.DataFrame,
    *,
    units: dict[str, str | None] | None = None,
) -> ParquetArtifact:
    """Serialize normalized data and hash the exact immutable Parquet bytes."""

    normalized = _canonical_frame(frame)
    table = pa.Table.from_pandas(normalized, preserve_index=False, safe=True)
    schema_document = _schema_document(table, units or {})
    metadata = dict(table.schema.metadata or {})
    metadata[SCHEMA_METADATA_KEY] = json.dumps(
        schema_document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    output = io.BytesIO()
    pq.write_table(
        table,
        output,
        compression="zstd",
        version="2.6",
        data_page_version="2.0",
        use_dictionary=False,
        write_statistics=True,
        coerce_timestamps="us",
        allow_truncated_timestamps=False,
    )
    payload = output.getvalue()
    return ParquetArtifact(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        schema_document=schema_document,
        row_count=len(normalized),
        column_count=len(normalized.columns),
    )


def read_parquet(payload: bytes, *, expected_sha256: str | None = None) -> pd.DataFrame:
    """Validate integrity and schema metadata before reconstructing a dataframe."""

    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ParquetContractError("Parquet content hash does not match metadata.")
    try:
        table = pq.read_table(io.BytesIO(payload))
    except Exception as exc:
        raise ParquetContractError("DatasetVersion object is not readable Parquet.") from exc
    metadata = table.schema.metadata or {}
    encoded_schema = metadata.get(SCHEMA_METADATA_KEY)
    if encoded_schema is None:
        raise ParquetContractError("Parquet object has no LabViz schema metadata.")
    schema_document = json.loads(encoded_schema)
    if schema_document.get("schemaVersion") != PARQUET_SCHEMA_VERSION:
        raise ParquetContractError("Unsupported LabViz Parquet schema version.")
    if schema_document.get("contentHash") != PARQUET_CONTENT_HASH:
        raise ParquetContractError("Unsupported Parquet content hash rule.")
    return table.to_pandas()
