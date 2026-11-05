import { describe, expect, it } from "vitest";

import { defaultChartSpec, type ChartSpec } from "@/domain/chart-spec";

import { analysisCacheKey, planAppearanceKey } from "./analysis-cache-key";
import type { IssueAction } from "./workspace-store";

const noDecisions: Record<string, IssueAction> = {};

function withSeries(overrides: Partial<ChartSpec["series"][number]>): ChartSpec {
  return {
    ...defaultChartSpec,
    series: [{ ...defaultChartSpec.series[0], ...overrides }],
  };
}

describe("analysisCacheKey", () => {
  it("ignores a series colour change, which only the renderer reads", () => {
    expect(analysisCacheKey(withSeries({ color: "#DC2626" }), noDecisions)).toBe(
      analysisCacheKey(defaultChartSpec, noDecisions),
    );
  });

  it("ignores a line style change", () => {
    expect(analysisCacheKey(withSeries({ lineStyle: "dashed" }), noDecisions)).toBe(
      analysisCacheKey(defaultChartSpec, noDecisions),
    );
  });

  it("ignores export and appearance settings", () => {
    const restyled: ChartSpec = {
      ...defaultChartSpec,
      title: "A different title",
      subtitle: "A different subtitle",
      export: {
        ...defaultChartSpec.export,
        dpi: 600,
        fontSize: 14,
        lineWidth: 3,
        markerSize: 8,
        legendPosition: "bottom",
        grayscalePreview: true,
      },
    };

    expect(analysisCacheKey(restyled, noDecisions)).toBe(
      analysisCacheKey(defaultChartSpec, noDecisions),
    );
  });

  it("separates every input the API actually reads", () => {
    const base = analysisCacheKey(defaultChartSpec, noDecisions);
    const variants: ChartSpec[] = [
      { ...defaultChartSpec, type: "scatter" },
      { ...defaultChartSpec, panelCount: 2 },
      { ...defaultChartSpec, groupField: "batch" },
      { ...defaultChartSpec, xAxis: { ...defaultChartSpec.xAxis, field: "other" } },
      { ...defaultChartSpec, fitting: { ...defaultChartSpec.fitting, model: "linear" } },
      {
        ...defaultChartSpec,
        uncertainty: { ...defaultChartSpec.uncertainty, mode: "standard-deviation" },
      },
      {
        ...defaultChartSpec,
        secondaryYAxis: { ...defaultChartSpec.secondaryYAxis, enabled: true, field: "second" },
      },
      withSeries({ field: "other" }),
      withSeries({ panel: 2 }),
      withSeries({ yAxis: "secondary" }),
    ];

    for (const variant of variants) {
      expect(analysisCacheKey(variant, noDecisions)).not.toBe(base);
    }
  });

  it("keeps the series label, because the response echoes it into the legend", () => {
    expect(analysisCacheKey(withSeries({ label: "Renamed" }), noDecisions)).not.toBe(
      analysisCacheKey(defaultChartSpec, noDecisions),
    );
  });

  it("separates cleaning decisions, which change the analysed rows", () => {
    expect(analysisCacheKey(defaultChartSpec, { "missing:response": "remove" })).not.toBe(
      analysisCacheKey(defaultChartSpec, noDecisions),
    );
  });
});

describe("planAppearanceKey", () => {
  const histogram: ChartSpec = { ...defaultChartSpec, type: "histogram" };

  it("separates everything a plan draws", () => {
    const base = planAppearanceKey(histogram);
    const variants: ChartSpec[] = [
      { ...histogram, series: [{ ...histogram.series[0], color: "#DC2626" }] },
      { ...histogram, xAxis: { ...histogram.xAxis, title: "Other title" } },
      { ...histogram, xAxis: { ...histogram.xAxis, unit: "other-x" } },
      { ...histogram, yAxis: { ...histogram.yAxis, title: "Other value" } },
      { ...histogram, yAxis: { ...histogram.yAxis, unit: "other-y" } },
      { ...histogram, export: { ...histogram.export, gridVisible: !histogram.export.gridVisible } },
      { ...histogram, export: { ...histogram.export, grayscalePreview: !histogram.export.grayscalePreview } },
    ];
    for (const variant of variants) expect(planAppearanceKey(variant)).not.toBe(base);
  });

  it("ignores what the page draws over the plan", () => {
    const restyled = { ...histogram, title: "Other", export: { ...histogram.export, fontSize: 14 } };
    expect(planAppearanceKey(restyled)).toBe(planAppearanceKey(histogram));
  });

  it("follows line, scatter and fitted charts, whose plans draw their style", () => {
    const fitted: ChartSpec = { ...defaultChartSpec, type: "bar", fitting: { ...defaultChartSpec.fitting, model: "linear" } };
    for (const spec of [{ ...defaultChartSpec, type: "line" as const }, fitted]) {
      const recoloured = { ...spec, series: [{ ...spec.series[0], color: "#DC2626" }] };
      const widened = { ...spec, export: { ...spec.export, lineWidth: spec.export.lineWidth + 1 } };
      expect(planAppearanceKey(recoloured)).not.toBe(planAppearanceKey(spec));
      expect(planAppearanceKey(widened)).not.toBe(planAppearanceKey(spec));
    }
  });

  it("leaves charts without a plan alone", () => {
    const heatmap: ChartSpec = { ...defaultChartSpec, type: "heatmap" };
    const recoloured = { ...heatmap, series: [{ ...heatmap.series[0], color: "#DC2626" }] };
    expect(planAppearanceKey(recoloured)).toBe(planAppearanceKey(heatmap));
  });
});
