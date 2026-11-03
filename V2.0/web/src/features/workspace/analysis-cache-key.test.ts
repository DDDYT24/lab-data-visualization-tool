import { describe, expect, it } from "vitest";

import { defaultChartSpec, type ChartSpec } from "@/domain/chart-spec";

import { analysisCacheKey } from "./analysis-cache-key";
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
