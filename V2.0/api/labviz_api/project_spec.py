"""Authoritative Pydantic source for the versioned ProjectSpec contract."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from .models import ChartSpec, ContractModel


class ProjectSourceSpec(ContractModel):
    """Immutable identifiers and import selection for one project revision."""

    source_file_id: UUID
    dataset_id: UUID
    dataset_version_id: UUID
    sheet_name: str | None = None
    header_row: int | None = Field(default=None, ge=1)


class ProjectCleaningSpec(ContractModel):
    """Optional immutable cleaning decision-set reference."""

    decision_set_id: UUID
    revision: int = Field(ge=1)


class ProjectSpecV1(ContractModel):
    """Complete reproducible project snapshot stored by ProjectRevision."""

    schema_version: Literal[1] = 1
    project_id: UUID
    title: str = Field(min_length=1, max_length=200)
    description: str = ""
    source: ProjectSourceSpec
    cleaning: ProjectCleaningSpec | None = None
    chart: ChartSpec
