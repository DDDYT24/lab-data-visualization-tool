"""Pydantic models shared by the LabViz API endpoints."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

API_VERSION: Literal["v1"] = "v1"
MAX_PROJECT_DESCRIPTION_UTF8_BYTES = 4_000
JsonScalar = str | int | float | bool | None
SurfaceDiagnosticStatus = Literal[
    "valid",
    "invalid-values",
    "insufficient-points",
    "collinear",
    "duplicate-coordinates",
    "missing-grid",
    "irregular-grid",
]


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )


class VersionedModel(ContractModel):
    api_version: Literal["v1"] = API_VERSION


class SourceFile(ContractModel):
    name: str
    size: int = Field(ge=0)
    media_type: str
    sheet_name: str | None = None
    available_sheets: list[str] = Field(default_factory=list)
    header_row: int | None = Field(default=None, ge=1)


class ProcessingJob(VersionedModel):
    id: str
    project_id: str
    stage: Literal["queued", "uploading", "parsing", "profiling", "ready", "failed"]
    progress: float = Field(ge=0, le=100)
    message: str
    error_code: str | None = None


class ExperimentInput(ContractModel):
    title: str = Field(min_length=1, max_length=200)
    run_label: str = Field(min_length=1, max_length=200)
    replicate_id: str | None = Field(default=None, min_length=1, max_length=100)
    batch_id: str | None = Field(default=None, min_length=1, max_length=100)


class ExperimentContext(ExperimentInput):
    experiment_id: str
    experiment_run_id: str


class ProjectSession(VersionedModel):
    project_id: str
    storage_mode: Literal["temporary-cloud", "saved-cloud", "local"]
    description: str
    current_revision_id: UUID | None
    source: SourceFile
    job: ProcessingJob | None = None
    expires_at: str | None = None
    experiment: ExperimentContext | None = None


class PreviewColumn(ContractModel):
    field: str
    label: str
    kind: Literal["number", "text", "datetime", "boolean"]
    unit: str | None = None
    nullable: bool


class DataPreview(VersionedModel):
    project_id: str
    columns: list[PreviewColumn]
    rows: list[dict[str, JsonScalar]]
    total_rows: int = Field(ge=0)
    sampled: bool
    sample_strategy: Literal["none", "evenly-distributed"]


class QualityFinding(ContractModel):
    id: str
    kind: Literal[
        "missing",
        "duplicate",
        "type-conflict",
        "extreme-value",
        "sudden-change",
        "outside-range",
        "trend-inconsistent",
    ]
    severity: Literal["info", "warning", "error"]
    column: str | None = None
    row_ids: list[str | int]
    affected_count: int = Field(default=0, ge=0)
    row_ids_truncated: bool = False
    summary: str
    reason: str
    # Optional for backwards compatibility with persisted v1 quality documents.
    # New processing responses always include stable locale-independent codes and
    # scalar parameters so clients can render the text in the user's language.
    summary_code: str | None = None
    summary_params: dict[str, JsonScalar] | None = None
    reason_code: str | None = None
    reason_params: dict[str, JsonScalar] | None = None


class QualityReport(VersionedModel):
    project_id: str
    total_rows: int = Field(ge=0)
    valid_rows: int = Field(ge=0)
    missing_values: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    suspicious_points: int = Field(ge=0)
    findings: list[QualityFinding]


class ValidRangeRule(ContractModel):
    field: str = Field(min_length=1)
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> ValidRangeRule:
        if self.minimum is None and self.maximum is None:
            raise ValueError("minimum or maximum is required")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot be greater than maximum")
        return self


class QualityRulesRequest(ContractModel):
    ranges: list[ValidRangeRule] = Field(max_length=50)


class AxisSpec(ContractModel):
    field: str
    title: str = Field(max_length=120)
    unit: str = Field(max_length=40)


class SeriesSpec(ContractModel):
    field: str
    label: str = Field(min_length=1, max_length=120)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    line_style: Literal["solid", "dashed", "dotted", "dashdot"] = "solid"
    panel: int = Field(default=1, ge=1, le=4)
    y_axis: Literal["primary", "secondary"] = "primary"


class FittingSpec(ContractModel):
    model: Literal["none", "linear", "polynomial", "exponential", "logarithmic", "power"] = "none"
    polynomial_order: Literal[1, 2, 3] = 2
    fit_method: Literal["ordinary-least-squares", "weighted-least-squares"] = (
        "ordinary-least-squares"
    )
    show_equation: bool = True
    show_r_squared: bool = True
    confidence_band: bool = False
    confidence_method: Literal["student-t", "bootstrap"] = "student-t"
    confidence_level: Literal[90, 95, 99] = 95


class UncertaintySpec(ContractModel):
    mode: Literal[
        "none",
        "standard-deviation",
        "standard-error",
        "confidence-interval",
        "column",
    ] = "none"
    error_field: str | None = None
    confidence_level: Literal[90, 95, 99] = 95

    @model_validator(mode="after")
    def require_error_column(self) -> UncertaintySpec:
        if self.mode == "column" and not self.error_field:
            raise ValueError("errorField is required when uncertainty mode is column")
        return self


class SecondaryYAxisSpec(ContractModel):
    enabled: bool = False
    field: str | None = None
    title: str = Field(default="", max_length=120)
    unit: str = Field(default="", max_length=40)

    @model_validator(mode="after")
    def require_secondary_field(self) -> SecondaryYAxisSpec:
        if self.enabled and not self.field:
            raise ValueError("field is required when the secondary Y axis is enabled")
        return self


class ExportSpec(ContractModel):
    format: Literal["png", "svg", "pdf"]
    dpi: Literal[300, 600]
    size_preset: Literal["single-column", "double-column", "a4", "custom"]
    grayscale_preview: bool
    width: float | None = Field(default=None, gt=0, le=2000)
    height: float | None = Field(default=None, gt=0, le=2000)
    unit: Literal["mm", "cm", "in"] = "in"
    font_family: Literal["Arial", "Times New Roman"] = "Arial"
    font_size: float = Field(default=10, ge=6, le=36)
    line_width: float = Field(default=1.5, ge=0.25, le=10)
    marker_size: float = Field(default=4, ge=1, le=20)
    legend_position: Literal["auto", "top", "bottom", "left", "right", "none"] = "auto"
    transparent_background: bool = False
    grid_visible: bool = True
    background_color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")

    @model_validator(mode="after")
    def require_custom_dimensions(self) -> ExportSpec:
        if self.size_preset == "custom" and (self.width is None or self.height is None):
            raise ValueError("width and height are required for a custom figure size")
        return self


class ChartSpec(ContractModel):
    schema_version: Literal[1]
    type: Literal["line", "scatter", "bar", "histogram", "box", "heatmap", "surface3d"]
    title: str = Field(max_length=200)
    subtitle: str = Field(default="", max_length=200)
    x_axis: AxisSpec
    y_axis: AxisSpec
    series: list[SeriesSpec] = Field(min_length=1, max_length=20)
    group_field: str | None = None
    panel_count: int = Field(ge=1, le=4)
    fitting: FittingSpec = Field(default_factory=FittingSpec)
    uncertainty: UncertaintySpec = Field(default_factory=UncertaintySpec)
    secondary_y_axis: SecondaryYAxisSpec = Field(default_factory=SecondaryYAxisSpec)
    export_settings: ExportSpec = Field(alias="export")

    @model_validator(mode="after")
    def validate_panel_assignments(self) -> ChartSpec:
        if any(item.panel > self.panel_count for item in self.series):
            raise ValueError("series panel cannot exceed panelCount")
        return self


class CleaningDecision(ContractModel):
    finding_id: str
    action: Literal["ignore", "exclude", "remove"]


class CleaningDecisionsRequest(ContractModel):
    decisions: list[CleaningDecision]


class CleaningDecisionsResponse(VersionedModel):
    project_id: str
    decisions: list[CleaningDecision]
    updated_at: str


class ChartRequest(ContractModel):
    chart: ChartSpec


class SavedChartResponse(VersionedModel):
    project_id: str
    chart: ChartSpec
    updated_at: str


class UpdateProjectDescriptionRequest(ContractModel):
    description: str
    expected_revision_id: UUID

    @field_validator("description")
    @classmethod
    def validate_plain_text_description(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("description cannot contain a null character")
        try:
            size = len(value.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("description must be valid UTF-8 text") from exc
        if size > MAX_PROJECT_DESCRIPTION_UTF8_BYTES:
            raise ValueError(
                f"description cannot exceed {MAX_PROJECT_DESCRIPTION_UTF8_BYTES} UTF-8 bytes"
            )
        return value


class ProjectDescriptionResponse(VersionedModel):
    project_id: str
    description: str
    revision_id: UUID
    revision_number: int = Field(ge=1)
    updated_at: str


class ProjectSummary(ContractModel):
    id: str
    title: str
    source_name: str
    chart_type: Literal["line", "scatter", "bar", "histogram", "box", "heatmap", "surface3d"]
    updated_at: str
    storage_mode: Literal["saved-cloud", "local"]
    thumbnail_url: str | None = None
    experiment: ExperimentContext | None = None


class ProjectList(VersionedModel):
    projects: list[ProjectSummary]


class CreateShareRequest(ContractModel):
    downloads_enabled: bool = False


class ShareLink(VersionedModel):
    token: str
    url: str
    downloads_enabled: bool
    created_at: str


class SharedDownloads(ContractModel):
    png: str | None = None
    svg: str | None = None
    pdf: str | None = None


class RequestEmailCode(ContractModel):
    email: EmailStr


class RequestEmailCodeResponse(VersionedModel):
    challenge_id: str
    expires_in_seconds: int = Field(gt=0)
    resend_after_seconds: int = Field(ge=0)
    delivery_mode: Literal["console", "email"]


class VerifyEmailCode(ContractModel):
    challenge_id: str
    code: str = Field(pattern=r"^\d{6}$")


class AuthenticatedUser(ContractModel):
    id: str
    email: EmailStr


class VerifyEmailCodeResponse(VersionedModel):
    authenticated: Literal[True] = True
    user: AuthenticatedUser


class AuthState(VersionedModel):
    authenticated: bool
    user: AuthenticatedUser | None = None


class ShareSummary(ContractModel):
    token: str
    url: str
    downloads_enabled: bool
    created_at: str


class UpdateShareRequest(ContractModel):
    downloads_enabled: bool


class ProjectWorkspace(VersionedModel):
    session: ProjectSession
    preview: DataPreview
    quality: QualityReport
    decisions: list[CleaningDecision]
    chart: ChartSpec
    shares: list[ShareSummary]


class FitPoint(ContractModel):
    x: float
    y: float
    lower: float | None = None
    upper: float | None = None


class FitAnalysis(ContractModel):
    model: Literal["linear", "polynomial", "exponential", "logarithmic", "power"]
    equation: str
    r_squared: float
    sample_size: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    fit_method: Literal["ordinary-least-squares", "weighted-least-squares"]
    confidence_method: Literal["none", "student-t", "bootstrap"]
    interval_kind: Literal["none", "pointwise-mean"]
    points: list[FitPoint]


class UncertaintyPoint(ContractModel):
    x: float
    y: float
    error: float


class UncertaintyAnalysis(ContractModel):
    mode: Literal["standard-deviation", "standard-error", "confidence-interval", "column"]
    points: list[UncertaintyPoint]


class ResidualDiagnostic(ContractModel):
    status: Literal["supported", "insufficient-data"]
    sample_size: int = Field(ge=0)
    mean_residual: float | None = None
    rmse: float | None = None
    mae: float | None = None
    max_abs_residual: float | None = None
    residual_trend: Literal["none", "possible-trend"] | None = None
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AnalysisDisclosure(ContractModel):
    method: Literal[
        "fit",
        "confidence-band",
        "residual-diagnostic",
        "prediction-band",
        "simultaneous-band",
        "robust-fitting",
        "weighted-fitting",
        "multiple-comparison",
    ]
    status: Literal["supported", "not-requested", "insufficient-data", "deferred"]
    sample_size: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class SeriesPoint(ContractModel):
    x: JsonScalar
    y: float


class SeriesAnalysis(ContractModel):
    field: str
    label: str
    panel: int
    group: JsonScalar = None
    points: list[SeriesPoint]
    fit: FitAnalysis | None = None
    uncertainty: UncertaintyAnalysis | None = None
    residual_diagnostic: ResidualDiagnostic | None = None
    disclosures: list[AnalysisDisclosure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HistogramBin(ContractModel):
    start: float
    end: float
    count: int = Field(ge=0)


class HistogramPreview(ContractModel):
    field: str
    label: str
    bins: list[HistogramBin]


class BoxPreview(ContractModel):
    field: str
    label: str
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    outliers: list[float]


class HeatmapPreview(ContractModel):
    panel: int
    labels: list[str]
    matrix: list[list[float | None]]


class SurfacePoint(ContractModel):
    panel: int
    x: float
    y: float
    z: float


class SurfaceDiagnostic(ContractModel):
    panel: int = Field(ge=1, le=4)
    x_field: str
    y_field: str
    z_field: str
    x_count: int = Field(ge=0)
    y_count: int = Field(ge=0)
    expected_points: int = Field(ge=0)
    usable_points: int = Field(ge=0)
    duplicate_coordinate_pairs: int = Field(ge=0)
    duplicate_coordinate_rows: int = Field(ge=0)
    missing_grid_cells: int = Field(ge=0)
    non_finite_points: int = Field(ge=0)
    uniform_x_spacing: bool
    uniform_y_spacing: bool
    collinear: bool
    status: SurfaceDiagnosticStatus


class ChartDerivedPreview(ContractModel):
    histograms: list[HistogramPreview] = Field(default_factory=list)
    boxes: list[BoxPreview] = Field(default_factory=list)
    heatmaps: list[HeatmapPreview] = Field(default_factory=list)
    surface_points: list[SurfacePoint] = Field(default_factory=list)
    surface_diagnostics: list[SurfaceDiagnostic] = Field(default_factory=list)


class ChartRecommendation(ContractModel):
    chart_type: Literal["surface3d", "heatmap", "scatter"]
    source: Literal["regular-grid", "grid-like"]
    x_field: str
    y_field: str
    z_field: str
    reason: str
    reason_code: str
    reason_params: dict[str, JsonScalar] = Field(default_factory=dict)


class ChartAnalysis(VersionedModel):
    project_id: str
    series: list[SeriesAnalysis]
    preview: ChartDerivedPreview
    recommendations: list[ChartRecommendation] = Field(default_factory=list)


class SharedChart(VersionedModel):
    token: str
    title: str
    description: str
    updated_at: str
    chart: ChartSpec
    preview: DataPreview
    analysis: ChartAnalysis
    downloads: SharedDownloads


class ExportJob(VersionedModel):
    id: str
    project_id: str
    status: Literal["queued", "rendering", "ready", "failed"]
    download_url: str | None = None
    expires_at: str | None = None
    message: str
    experiment: ExperimentContext | None = None


class HealthResponse(VersionedModel):
    status: Literal["ok"] = "ok"
