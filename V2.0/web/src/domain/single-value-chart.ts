import type { DataPreview } from "@/domain/api-contract";
import type { ChartSpec } from "@/domain/chart-spec";

/** Recover the old default that used the sole numeric column for both axes. */
export function repairSingleValueChart(
  chart: ChartSpec,
  columns: DataPreview["columns"],
): ChartSpec {
  const numeric = columns.filter((column) => column.kind === "number");
  if (
    numeric.length !== 1 ||
    !["line", "scatter", "bar"].includes(chart.type) ||
    chart.xAxis.field !== numeric[0].field ||
    !chart.series.every((series) => series.field === numeric[0].field)
  ) return chart;

  return {
    ...chart,
    type: "histogram",
    title: chart.title === "Response over time" ? numeric[0].label : chart.title,
    groupField: null,
    fitting: {
      ...chart.fitting,
      model: "none",
      confidenceBand: false,
      fitMethod: "ordinary-least-squares",
      intervalKind: "pointwise-mean",
    },
    uncertainty: { ...chart.uncertainty, mode: "none", errorField: null },
    secondaryYAxis: { ...chart.secondaryYAxis, enabled: false, field: null },
  };
}
