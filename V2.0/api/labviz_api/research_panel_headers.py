"""Small immutable panel bindings; no full result decoding on ordinary layout saves."""

from __future__ import annotations

import json
from typing import Any


def install_headers(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS research_panel_headers (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            header_json TEXT NOT NULL
        )
    """)


def scientific_header(result: dict[str, Any], revision: str) -> dict[str, Any]:
    observations = [point["observation"] for point in result["points"]]
    units = {row.get("experimentalUnit") for row in observations}
    kind = result["spec"]["kind"]
    return {
        "revisionId": revision,
        "spec": result["spec"],
        "kind": kind,
        "title": result["spec"]["title"],
        "unit": result["unit"],
        "observations": len(observations),
        "independentUnits": None if None in units else len(units),
        "resultSha256": result["resultSha256"],
        "xSignature": ["paired", result["spec"]["pairTimes"]]
        if kind == "paired"
        else ["group", [group["label"] for group in result["groups"]]],
        "ySignature": ["numeric", observations[0]["measurementType"], result["unit"]],
    }


def source_header(source: dict[str, Any]) -> dict[str, Any]:
    units = {row.get("experimentalUnit") for row in source["rows"]}
    return {
        **{
            key: source[key]
            for key in (
                "revisionId",
                "kind",
                "title",
                "unit",
                "resultSha256",
                "xSignature",
                "ySignature",
            )
        },
        "observations": len(source["rows"]),
        "independentUnits": None if None in units else len(units),
    }


def remember_header(
    connection: Any, project_id: str, identifier: str, header: dict[str, Any]
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO research_panel_headers VALUES (?,?,?)",
        (identifier, project_id, json.dumps(header, ensure_ascii=False, allow_nan=False)),
    )
