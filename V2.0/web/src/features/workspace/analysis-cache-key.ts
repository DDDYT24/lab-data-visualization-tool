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
 * Restyling refetches only a plan: the API keeps the finished analysis, so no
 * fit or resample runs again. Heatmaps and surfaces read their appearance from
 * the spec and stay out.
 */
export function planAppearanceKey(spec: ChartSpec): string {
  if (spec.type === "heatmap" || spec.type === "surface3d") return "";
  return JSON.stringify({
    colors: spec.series.map((series) => [series.color, series.lineStyle]),
    grayscale: spec.export.grayscalePreview,
    grid: spec.export.gridVisible,
    lineWidth: spec.export.lineWidth,
    markerSize: spec.export.markerSize,
    xAxis: [spec.xAxis.title, spec.xAxis.unit],
    yAxis: [spec.yAxis.title, spec.yAxis.unit],
  });
}
