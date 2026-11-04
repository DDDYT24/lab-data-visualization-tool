import type { ChartSpec } from "@/domain/chart-spec";

import type { IssueAction } from "./workspace-store";

/**
 * The chart-analysis cache key, built only from fields the API actually reads.
 *
 * Appearance is deliberately excluded. Series colour and line style reach the
 * browser renderer and the Matplotlib export, never `analyze_chart`, so
 * including them re-ran every fit, interval and bootstrap resample whenever an
 * appearance control moved. Series label stays because the response echoes it.
 */
export function analysisCacheKey(
  spec: ChartSpec,
  issueActions: Record<string, IssueAction>,
): string {
  return JSON.stringify({
    fitting: spec.fitting,
    groupField: spec.groupField,
    issueActions,
    panelCount: spec.panelCount,
    secondaryYAxis: spec.secondaryYAxis,
    series: spec.series.map((series) => ({
      field: series.field,
      label: series.label,
      panel: series.panel,
      yAxis: series.yAxis,
    })),
    type: spec.type,
    uncertainty: spec.uncertainty,
    xAxis: spec.xAxis.field,
  });
}

/**
 * What the analysis response's RenderPlan draws, for charts that have one.
 *
 * These charts run no fit or resample, so restyling them only costs a plan.
 * Charts without a plan read their appearance from the spec, and stay out.
 */
export function planAppearanceKey(spec: ChartSpec): string {
  const planned =
    spec.type === "histogram" ||
    spec.type === "box" ||
    (spec.type === "bar" &&
      spec.fitting.model === "none" &&
      spec.uncertainty.mode === "none");
  if (!planned) return "";
  return JSON.stringify({
    colors: spec.series.map((series) => series.color),
    grayscale: spec.export.grayscalePreview,
    grid: spec.export.gridVisible,
    xAxis: [spec.xAxis.title, spec.xAxis.unit],
    yAxis: [spec.yAxis.title, spec.yAxis.unit],
  });
}
