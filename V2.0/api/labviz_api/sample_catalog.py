"""Versioned, synthetic, offline sample catalog for the V2.2 local workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .models import SampleCatalogResponse, SampleExample

SAMPLES_ROOT = Path(__file__).resolve().parents[1] / "samples" / "v22"
MANIFEST_PATH = SAMPLES_ROOT / "manifest.json"


class SampleCatalogError(ValueError):
    """Raised when a tracked sample manifest or payload is unavailable."""


def _manifest() -> dict[str, Any]:
    try:
        document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SampleCatalogError("The bundled V2.2 sample catalog is unavailable.") from exc
    if document.get("catalogVersion") != "v1" or document.get("syntheticOnly") is not True:
        raise SampleCatalogError("The bundled sample catalog failed its version or privacy check.")
    if not isinstance(document.get("examples"), list):
        raise SampleCatalogError("The bundled sample catalog has no examples list.")
    return cast(dict[str, Any], document)


def public_samples() -> list[SampleExample]:
    return [
        SampleExample.model_validate(entry)
        for entry in _manifest()["examples"]
        if entry.get("visibility") == "public"
    ]


def catalog_response() -> SampleCatalogResponse:
    document = _manifest()
    return SampleCatalogResponse(
        catalog_version=document["catalogVersion"],
        release=document.get("release", "V2.2"),
        synthetic_only=document["syntheticOnly"],
        examples=public_samples(),
    )


def sample_payload(slug: str) -> tuple[SampleExample, bytes]:
    entry = next(
        (
            item
            for item in _manifest()["examples"]
            if item.get("visibility") == "public" and item.get("slug") == slug
        ),
        None,
    )
    if entry is None:
        raise SampleCatalogError("That synthetic sample does not exist.")
    sample = SampleExample.model_validate(entry)
    payload_path = (SAMPLES_ROOT / sample.filename).resolve()
    if payload_path.parent != SAMPLES_ROOT.resolve():
        raise SampleCatalogError("The sample path is outside the bundled catalog directory.")
    try:
        payload = payload_path.read_bytes()
    except OSError as exc:
        raise SampleCatalogError("The bundled synthetic sample payload is unavailable.") from exc
    if not payload:
        raise SampleCatalogError("The bundled synthetic sample payload is empty.")
    return sample, payload
