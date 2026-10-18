"""Prospective V2.3 sidecar contracts; not connected to persistence or public routes."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ResearchModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class Source(ResearchModel):
    id: Identifier
    sha256: Digest
    name: Annotated[str, Field(min_length=1, max_length=255)]


class Lineage(ResearchModel):
    source_id: Identifier
    sheet: str | None
    row: Annotated[int, Field(ge=1)]
    column: Annotated[str, Field(min_length=1)]


class Sample(ResearchModel):
    id: Identifier
    experimental_unit_id: Identifier | None
    parent_sample_id: Identifier | None
    group: str | None


class Acquisition(ResearchModel):
    id: Identifier
    batch: Identifier
    protocol: str | None


class Observation(ResearchModel):
    id: Identifier
    sample_id: Identifier | None
    acquisition_id: Identifier
    time_point: str | None
    technical_repeat: Annotated[int, Field(ge=1)] | None
    pair_key: Identifier | None
    unit: Annotated[str, Field(min_length=1)]
    state: Literal[
        "observed",
        "missing",
        "not_measured",
        "failed",
        "below_detection",
        "below_quantification",
        "out_of_range",
        "saturated",
    ]
    value: float | None
    reported_text: str
    limit: float | None
    lineage: Annotated[list[Lineage], Field(min_length=1)]


class ProcessingStep(ResearchModel):
    id: Identifier
    operation: Literal[
        "stack", "map_roles", "wide_to_long", "long_to_wide", "filter", "convert_unit"
    ]
    input_revision: Identifier
    output_revision: Identifier
    algorithm_version: Identifier
    parameters_json: str


class Exclusion(ResearchModel):
    observation_id: Identifier
    reason: Annotated[str, Field(min_length=1)]


class Analysis(ResearchModel):
    id: Identifier
    data_revision: Identifier
    method: Identifier
    method_version: Identifier
    parameters_json: str
    seed: Annotated[int, Field(ge=0, le=2147483647)]
    analysis_unit: Literal["experimental_unit", "sample", "observation"]
    input_observation_ids: Annotated[list[Identifier], Field(min_length=1)]
    excluded: list[Exclusion]
    independent_unit_count: Annotated[int, Field(ge=0)] | None
    sample_count: Annotated[int, Field(ge=0)]
    observation_count: Annotated[int, Field(ge=1)]
    result_json: str


class Panel(ResearchModel):
    id: Literal["A", "B", "C", "D"]
    chart_type: Literal[
        "line",
        "scatter",
        "bar",
        "histogram",
        "box",
        "heatmap",
        "strip",
        "paired",
        "violin",
    ]
    data_revision: Identifier
    analysis_id: Identifier | None
    x_field: Annotated[str, Field(min_length=1)]
    y_field: Annotated[str, Field(min_length=1)]
    x_unit: Annotated[str, Field(min_length=1)]
    y_unit: Annotated[str, Field(min_length=1)]
    jitter_seed: Annotated[int, Field(ge=0, le=2147483647)]


class AxisLink(ResearchModel):
    axis: Literal["x", "y"]
    panel_ids: Annotated[list[Literal["A", "B", "C", "D"]], Field(min_length=2, max_length=4)]


class Figure(ResearchModel):
    schema_version: Literal[1]
    id: Identifier
    revision: Identifier
    panels: Annotated[list[Panel], Field(min_length=1, max_length=4)]
    axis_links: list[AxisLink]
    style_id: Identifier


class ResearchContractV1(ResearchModel):
    schema_version: Literal[1]
    id: Identifier
    data_revision: Identifier
    sources: Annotated[list[Source], Field(min_length=1, max_length=5)]
    samples: list[Sample]
    acquisitions: Annotated[list[Acquisition], Field(min_length=1)]
    observations: Annotated[list[Observation], Field(min_length=1, max_length=100000)]
    processing: list[ProcessingStep]
    analyses: list[Analysis]
    figure: Figure

    @model_validator(mode="after")
    def validate_relationships(self) -> ResearchContractV1:
        collections = (
            self.sources,
            self.samples,
            self.acquisitions,
            self.observations,
            self.processing,
            self.analyses,
            self.figure.panels,
        )
        for collection in collections:
            ids = [item.id for item in collection]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate_id")
        sources = {item.id for item in self.sources}
        samples = {item.id: item for item in self.samples}
        acquisitions = {item.id for item in self.acquisitions}
        observations = {item.id: item for item in self.observations}
        analyses = {item.id: item for item in self.analyses}
        for sample in self.samples:
            seen = {sample.id}
            parent = sample.parent_sample_id
            while parent is not None:
                if parent not in samples or parent in seen:
                    raise ValueError("invalid_sample_parent")
                seen.add(parent)
                parent = samples[parent].parent_sample_id
        for observation in self.observations:
            if observation.sample_id is not None and observation.sample_id not in samples:
                raise ValueError("unknown_sample")
            if observation.acquisition_id not in acquisitions:
                raise ValueError("unknown_acquisition")
            if any(ref.source_id not in sources for ref in observation.lineage):
                raise ValueError("unknown_source")
            if observation.state == "observed" and observation.value is None:
                raise ValueError("observed_requires_value")
            if (
                observation.state in {"missing", "not_measured", "failed"}
                and observation.value is not None
            ):
                raise ValueError("absent_state_requires_null")
            if (
                observation.state in {"below_detection", "below_quantification"}
                and observation.limit is None
            ):
                raise ValueError("qualified_state_requires_limit")
        for analysis in self.analyses:
            included = analysis.input_observation_ids
            excluded = [item.observation_id for item in analysis.excluded]
            if (
                analysis.data_revision != self.data_revision
                or len(included) != len(set(included))
                or len(excluded) != len(set(excluded))
                or set(included) & set(excluded)
                or not (set(included) | set(excluded)) <= observations.keys()
            ):
                raise ValueError("invalid_analysis_inputs")
            selected = [observations[key] for key in included]
            if any(item.state != "observed" for item in selected):
                raise ValueError("unsupported_measurement_state")
            sample_ids = {item.sample_id for item in selected if item.sample_id is not None}
            units = {samples[key].experimental_unit_id for key in sample_ids}
            unknown = len(sample_ids) == 0 or any(item.sample_id is None for item in selected)
            unknown = unknown or None in units
            expected_units = None if unknown else len(units)
            if (
                analysis.sample_count != len(sample_ids)
                or analysis.observation_count != len(selected)
                or analysis.independent_unit_count != expected_units
            ):
                raise ValueError("incorrect_analysis_counts")
            if analysis.analysis_unit == "experimental_unit" and unknown:
                raise ValueError("unknown_experimental_unit")
        panels = self.figure.panels
        if [panel.id for panel in panels] != list("ABCD")[: len(panels)] or len(panels) == 3:
            raise ValueError("invalid_panel_layout")
        for panel in panels:
            if panel.data_revision != self.data_revision:
                raise ValueError("unknown_panel_revision")
            if panel.analysis_id is not None and panel.analysis_id not in analyses:
                raise ValueError("unknown_analysis")
        panel_map = {panel.id: panel for panel in panels}
        for link in self.figure.axis_links:
            if len(link.panel_ids) != len(set(link.panel_ids)):
                raise ValueError("duplicate_axis_panel")
            if any(key not in panel_map for key in link.panel_ids):
                raise ValueError("unknown_axis_panel")
            units = {getattr(panel_map[key], f"{link.axis}_unit") for key in link.panel_ids}
            if len(units) != 1:
                raise ValueError("incompatible_axis_units")
        return self
