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
