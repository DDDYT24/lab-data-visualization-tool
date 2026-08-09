import { describe, expect, it } from "vitest";

import type { ChartAnalysis, DataPreview } from "@/domain/api-contract";
import {
  chartLineType,
  chartRenderContract,
  chartSeriesColor,
} from "@/domain/chart-render-contract";
import { defaultChartSpec } from "@/domain/chart-spec";

import { buildChartOption } from "./chart-option";

describe("chart render contract", () => {
  it("keeps shared palettes and line styles versioned", () => {
    expect(chartRenderContract.contractVersion).toBe("chart-render-v1");
    expect(chartSeriesColor({
      configuredColor: "#FF0000",
      grayscale: true,
      grouped: true,
      index: 1,
    })).toBe("#5B616B");
    expect(chartSeriesColor({
      configuredColor: "#FF0000",
      grayscale: false,
      grouped: true,
      index: 1,
    })).toBe("#0F766E");
    expect(chartLineType("dashdot")).toEqual([8, 4, 2, 4]);
  });

  it("applies grayscale to grouped source, fit, and confidence-band series", () => {
    const preview: DataPreview = {
      apiVersion: "v1",
      projectId: "project-1",
      columns: [
        { field: "x", kind: "number", label: "X", nullable: false, unit: null },
        { field: "y", kind: "number", label: "Y", nullable: false, unit: null },
        { field: "group", kind: "text", label: "Group", nullable: false, unit: null },
      ],
      rows: [
        { rowId: 1, x: 1, y: 2, group: "A" },
        { rowId: 2, x: 2, y: 4, group: "B" },
      ],
      sampled: false,
      sampleStrategy: "none",
      totalRows: 2,
    };
    const fit = {
      equation: "y = 2x",
      model: "linear" as const,
      points: [{ lower: 1.8, upper: 2.2, x: 1, y: 2 }],
      rSquared: 1,
    };
    const analysis: ChartAnalysis = {
      apiVersion: "v1",
      projectId: "project-1",
      preview: { boxes: [], heatmaps: [], histograms: [], surfacePoints: [] },
      series: [
        {
          field: "y",
          fit,
          group: "A",
          label: "Y · A",
          panel: 1,
          points: [{ x: 1, y: 2 }],
          uncertainty: null,
          warnings: [],
        },
        {
          field: "y",
          fit,
          group: "B",
          label: "Y · B",
          panel: 1,
          points: [{ x: 2, y: 4 }],
          uncertainty: null,
          warnings: [],
        },
      ],
    };
    const option = buildChartOption({
      analysis,
      excludedFindingIds: [],
      findings: [],
      preview,
      spec: {
        ...defaultChartSpec,
        export: { ...defaultChartSpec.export, grayscalePreview: true },
        fitting: { ...defaultChartSpec.fitting, confidenceBand: true, model: "linear" },
        groupField: "group",
        series: [{ ...defaultChartSpec.series[0], field: "y" }],
        xAxis: { ...defaultChartSpec.xAxis, field: "x" },
      },
    });
    const series = option.series as Array<Record<string, Record<string, unknown>>>;

    expect(series[0].itemStyle.color).toBe("#20252D");
    expect(series[1].itemStyle.color).toBe("#5B616B");
    expect(series[2].lineStyle.color).toBe("#20252D");
    expect(series[4].areaStyle.color).toBe("#20252D");
    expect(series[5].lineStyle.color).toBe("#5B616B");
    expect(series[7].areaStyle.color).toBe("#5B616B");
  });
});
