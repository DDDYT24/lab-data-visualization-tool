"""Every reason the research API refuses a request, and what each one carries.

The browser shows a refusal by its code in the user's language. `contracts/capabilities-v1.json`
carries the codes to it, and a test fails while a code has no text in zh.json or en.json.
"""

from __future__ import annotations

from typing import Any, Literal

RefusalCode = Literal[
    "absent-state-has-value",
    "aggregate-byte-limit",
    "annotation-3d",
    "annotation-outside",
    "axis-control-3d",
    "axis-control-category",
    "axis-control-invalid",
    "axis-log-bars",
    "axis-log-nonpositive",
    "axis-ticks-outside",
    "chart-ambiguous-repeat",
    "chart-axis-range",
    "chart-cjk-font",
    "chart-density-degenerate",
    "chart-density-numeric",
    "chart-density-range",
    "chart-density-resolution",
    "chart-density-small",
    "chart-many-groups",
    "chart-mixed-measurements",
    "chart-not-found",
    "chart-numeric-range",
    "chart-pair-identity",
    "chart-pair-incompatible",
    "chart-pair-key-reused",
    "chart-pair-partner",
    "chart-pair-time",
    "chart-pair-times",
    "chart-plan-missing",
    "chart-unknown-summary-unit",
    "column-limit",
    "conflicting-sample",
    "duplicate-field",
    "duplicate-or-missing-pair",
    "duplicate-pivot-key",
    "duplicate-source",
    "empty-input",
    "empty-selection",
    "figure-build-refused",
    "figure-build-state",
    "figure-canvas-too-large",
    "figure-capture-failed",
    "figure-chart-unavailable",
    "figure-files-missing",
    "figure-incompatible-axis",
    "figure-layer-axis",
    "figure-layer-repeated",
    "figure-layer-revision",
    "figure-layer-unplanned",
    "figure-layer-unsupported",
    "figure-mark",
    "figure-not-found",
    "figure-not-ready",
    "figure-palette-unknown",
    "figure-panel",
    "figure-panel-unplanned",
    "figure-plan-missing",
    "figure-row",
    "figure-sample-mapping",
    "figure-secondary-axis",
    "figure-shared-axis-control",
    "figure-size",
    "figure-source-changed",
    "figure-type",
    "file-count-limit",
    "incompatible-pair",
    "incompatible-unit",
    "invalid-columns",
    "invalid-csv",
    "invalid-exclusion",
    "invalid-number",
    "invalid-parent",
    "invalid-repeat",
    "invalid-research-plan",
    "invalid-role",
    "limit-required",
    "missing-field",
    "missing-pivot-key",
    "no-redo",
    "no-undo",
    "observed-requires-value",
    "output-not-found",
    "output-review",
    "pair-key-required",
    "pair-times",
    "plan-invalid",
    "preview-expired",
    "research-busy",
    "research-conflict",
    "research-input-required",
    "research-local-only",
    "reserved-value-field",
    "reshape-collision",
    "reuse-figure-unsupported",
    "row-limit",
    "row-width",
    "schema-mismatch",
    "single-file-limit",
    "source-encoding",
    "strip-duplicate-id",
    "strip-empty",
    "strip-incomplete-input",
    "strip-invalid-value",
    "strip-mixed-units",
    "strip-non-observed",
    "template-blocked",
    "template-field",
    "template-mapping",
    "template-not-found",
    "template-unit",
    "unconverted-limit",
    "unit-mapping-mismatch",
    "unit-without-sample",
    "unknown-exclusion",
    "unknown-measurement-state",
    "unsupported-research-format",
    "unsupported-unit",
]

# Values a text needs, by the name it uses for them. The other codes carry none.
PARAMS: dict[RefusalCode, tuple[str, ...]] = {
    "conflicting-sample": ("sample",),
    "figure-capture-failed": ("detail",),
    "figure-incompatible-axis": ("axis",),
    "figure-shared-axis-control": ("axis",),
    "incompatible-unit": ("found", "target"),
    "invalid-csv": ("file",),
    "invalid-number": ("field",),
    "invalid-research-plan": ("detail",),
    "missing-field": ("fields",),
    "row-width": ("file", "row"),
    "single-file-limit": ("file",),
    "template-field": ("field",),
    "template-unit": ("expected", "found"),
}

DEFAULT_STATUS = 422
STATUS: dict[RefusalCode, int] = {
    "chart-not-found": 404,
    "chart-plan-missing": 404,
    "figure-files-missing": 404,
    "figure-not-found": 404,
    "figure-plan-missing": 404,
    "output-not-found": 404,
    "template-not-found": 404,
    "figure-build-state": 409,
    "figure-not-ready": 409,
    "preview-expired": 409,
    "research-conflict": 409,
    "research-busy": 429,
    "research-local-only": 501,
}


class PreparationError(ValueError):
    """A refusal. `message` is the English text for logs and for callers without the codes."""

    def __init__(self, code: RefusalCode, message: str, **params: str | int) -> None:
        self.code: RefusalCode = code
        self.params = params
        super().__init__(message)

    @property
    def status(self) -> int:
        return STATUS.get(self.code, DEFAULT_STATUS)

    def body(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "params": self.params}
