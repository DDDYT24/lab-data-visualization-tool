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
  it.each([
    [{ start: 4, end: 5, count: 2 }, { start: 5, end: 7, count: 3 }],
    [{ start: -3, end: -1, count: 1 }, { start: -1, end: 2, count: 4 }],
    [{ start: 4.5, end: 5.5, count: 8 }],
  ])("draws full API histogram intervals without inventing gaps: %j", (...bins) => {
    const analysis: ChartAnalysis = {
      apiVersion: "v1", projectId: "histogram", recommendations: [], series: [],
      preview: { histograms: [{ field: "response", label: "Sample A", bins }],
        boxes: [], heatmaps: [], surfaceDiagnostics: [], surfacePoints: [] },
    };
    const option = buildChartOption({ analysis, findings: [], excludedFindingIds: [],
      preview: { apiVersion: "v1", projectId: "histogram", columns: [], rows: [],
        sampled: false, sampleStrategy: "none", totalRows: 0 },
      spec: { ...defaultChartSpec, type: "histogram" },
    });
    const series = (option.series as Array<{ type: string; data: number[][]; itemStyle: object }>)[0];
    expect(series.type).toBe("custom");
    expect(series.data).toEqual(bins.map((bin) => [bin.start, bin.end, bin.count]));
    expect(series.itemStyle).toMatchObject({ opacity: chartRenderContract.histogramFillOpacity });
    const min = Math.min(...bins.map((bin) => bin.start));
    const max = Math.max(...bins.map((bin) => bin.end));
    const padding = (max - min) * chartRenderContract.histogramAxisPadding;
    expect(option.xAxis).toMatchObject([{ min: min - padding, max: max + padding, scale: true }]);
    expect(option.yAxis).toMatchObject([{ min: 0, max: Math.max(...bins.map((bin) => bin.count)) * 1.05 }]);
  });

  it.each([false, true])("keeps a fixed correlation scale in grayscale=%s", (grayscalePreview) => {
    const option = buildChartOption({
      analysis: undefined, excludedFindingIds: [], findings: [],
      preview: { apiVersion: "v1", projectId: "heatmap", columns: [], rows: [],
        sampled: false, sampleStrategy: "none", totalRows: 0 },
      spec: { ...defaultChartSpec, type: "heatmap",
        export: { ...defaultChartSpec.export, grayscalePreview } },
    });
    expect(option.visualMap).toMatchObject({ min: -1, max: 1, calculable: false,
      inRange: { color: grayscalePreview ? chartRenderContract.surfaceGrayscalePalette
        : chartRenderContract.correlationPalette } });
  });
  it("preserves a rectangular grid, boundary points and full Z scale when reducing a surface", () => {
    const rows = Array.from({ length: 10201 }, (_, index) => {
      const x = index % 101;
      const y = Math.floor(index / 101);
      return { rowId: index, x, y, z: x === 1 && y === 1 ? 999 : x + y };
    });
    const option = buildChartOption({
      analysis: undefined, excludedFindingIds: [], findings: [], surfacePointLimit: 800,
      preview: { apiVersion: "v1", projectId: "grid", columns: [], rows,
        sampled: false, sampleStrategy: "none", totalRows: rows.length },
      spec: { ...defaultChartSpec, type: "surface3d",
        xAxis: { field: "x", title: "X", unit: "" },
        series: [
          { ...defaultChartSpec.series[0], field: "y", label: "Y" },
          { ...defaultChartSpec.series[0], field: "z", label: "Z" },
        ] },
    });
    const surface = (option.series as Array<{ data: number[][]; dataShape: number[] }>)[0];
    expect(surface.data.length).toBeLessThanOrEqual(800);
    expect(surface.data.length).toBe(surface.dataShape[0] * surface.dataShape[1]);
    expect(surface.data[0]).toEqual([0, 0, 0]);
    expect(surface.data.at(-1)).toEqual([100, 100, 200]);
    expect(option.visualMap).toMatchObject({ min: 0, max: 999, dimension: 2,
      inRange: { color: chartRenderContract.surfacePalette } });
  });
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

  it("keeps a configured color for an ungrouped chart in dark mode", () => {
    const preview: DataPreview = {
      apiVersion: "v1",
      projectId: "dark-color-project",
      columns: [
        { field: "x", kind: "number", label: "X", nullable: false, unit: null },
        { field: "y", kind: "number", label: "Y", nullable: false, unit: null },
      ],
      rows: [
        { rowId: 1, x: 1, y: 2 },
        { rowId: 2, x: 2, y: 4 },
      ],
      sampled: false,
      sampleStrategy: "none",
      totalRows: 2,
    };
    const option = buildChartOption({
      analysis: undefined,
      colorMode: "dark",
      excludedFindingIds: [],
      findings: [],
      preview,
      spec: {
        ...defaultChartSpec,
        type: "scatter",
        xAxis: { field: "x", title: "X", unit: "" },
        yAxis: { field: "y", title: "Y", unit: "" },
        series: [{ ...defaultChartSpec.series[0], color: "#FF0000", field: "y" }],
      },
    });
    const series = option.series as Array<Record<string, Record<string, unknown>>>;

    expect(series[0].itemStyle.color).toBe("#FF0000");
    expect(series[0].lineStyle.color).toBe("#FF0000");
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
      preview: {
        boxes: [],
        heatmaps: [],
        histograms: [],
        surfaceDiagnostics: [],
        surfacePoints: [],
      },
      recommendations: [],
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

  it("builds a camera-enabled 3D surface from analyzed X, Y, and Z points", () => {
    const preview: DataPreview = {
      apiVersion: "v1",
      projectId: "surface-project",
      columns: [
        { field: "x", kind: "number", label: "X", nullable: false, unit: null },
        { field: "y", kind: "number", label: "Y", nullable: false, unit: null },
        { field: "z", kind: "number", label: "Z", nullable: false, unit: null },
      ],
      rows: [{ rowId: 1, x: 0, y: 0, z: 1 }],
      sampled: true,
      sampleStrategy: "evenly-distributed",
      totalRows: 441,
    };
    const analysis: ChartAnalysis = {
      apiVersion: "v1",
      projectId: "surface-project",
      preview: {
        boxes: [],
        heatmaps: [],
        histograms: [],
        surfaceDiagnostics: [
          {
            collinear: false,
            duplicateCoordinatePairs: 0,
            duplicateCoordinateRows: 0,
            expectedPoints: 441,
            missingGridCells: 0,
            nonFinitePoints: 0,
            panel: 1,
            status: "valid",
            uniformXSpacing: true,
            uniformYSpacing: true,
            usablePoints: 441,
            xCount: 21,
            xField: "x",
            yCount: 21,
            yField: "y",
            zField: "z",
          },
        ],
        surfacePoints: [
          { panel: 1, x: 0, y: 0, z: 1 },
          { panel: 1, x: 1, y: 0, z: 2 },
          { panel: 1, x: 0, y: 1, z: 3 },
          { panel: 1, x: 1, y: 1, z: 4 },
        ],
      },
      recommendations: [],
      series: [],
    };
    const option = buildChartOption({
      analysis,
      excludedFindingIds: [],
      findings: [],
      preview,
      spec: {
        ...defaultChartSpec,
        type: "surface3d",
        xAxis: { field: "x", title: "X", unit: "" },
        yAxis: { field: "y", title: "Y", unit: "" },
        series: [
          { ...defaultChartSpec.series[0], field: "y", label: "Y" },
          { ...defaultChartSpec.series[0], color: "#DC6B2F", field: "z", label: "Z" },
        ],
      },
    });

    expect(option).toMatchObject({
      grid3D: {
        viewControl: {
          panSensitivity: 1,
          projection: "perspective",
          rotateSensitivity: 1,
          zoomSensitivity: 1,
        },
      },
      xAxis3D: { name: "X", type: "value" },
      yAxis3D: { name: "Y", type: "value" },
      zAxis3D: { name: "Z", type: "value" },
    });
    expect(option.series).toEqual([
      {
        data: [
          [0, 0, 1],
          [1, 0, 2],
          [0, 1, 3],
          [1, 1, 4],
        ],
        dataShape: [2, 2],
        shading: "color",
        type: "surface",
      },
    ]);
  });
});
