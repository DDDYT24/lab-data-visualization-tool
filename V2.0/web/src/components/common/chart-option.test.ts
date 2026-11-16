import { describe, expect, it } from "vitest";

import type { ChartAnalysis } from "@/domain/api-contract";
import { chartLineType, chartRenderContract, chartSeriesColor } from "@/domain/chart-render-contract";
import { defaultChartSpec, type ChartSpec } from "@/domain/chart-spec";
import type { RenderPlan, RenderPlanMark, RenderPlanPanel } from "@/domain/render-plan";

import { buildChartOption } from "./chart-option";

type Option = any; // eslint-disable-line @typescript-eslint/no-explicit-any
const axis = (ticks: number[], domain: [number, number] = [0, 4]) => ({
  scale: "linear" as const, domain, title: "", niceRule: "nice-1-2-5-v1" as const, grid: true,
  ticks: ticks.map((value) => ({ value, label: String(value) })),
});
const panel = (marks: RenderPlanMark[], extra: Partial<RenderPlanPanel> = {}): RenderPlanPanel => ({
  id: "1", title: "Chart", legend: [], marks,
  axes: { x: axis([0, 2, 4]), y: axis([0, 2, 4]), y2: null, z: null }, ...extra,
});
const plan = (kind: RenderPlan["source"]["kind"], panels: RenderPlanPanel[]): RenderPlan => ({
  schemaVersion: 1, contract: "render-plan-v1", planHash: "0".repeat(64),
  source: { kind, resultSha256: "1".repeat(64), algorithmVersion: "classic-charts-v1" },
  geometry: { contract: "chart-render-v2", axis: "nice-1-2-5-v1", jitter: null, density: null, histogram: null, quantile: null, decimation: null },
  rendering: { pointRendering: "auto", rasterizePoints: false }, disclosures: [], panels,
});
const analysisOf = (planned: RenderPlan | null): ChartAnalysis => ({
  apiVersion: "v1", projectId: "p", recommendations: [], series: [], plan: planned,
  preview: { histograms: [], boxes: [], heatmaps: [], surfaceDiagnostics: [], surfacePoints: [] },
} as ChartAnalysis);
const spec = (type: ChartSpec["type"]): ChartSpec => ({ ...defaultChartSpec, type });
const line: RenderPlanMark = {
  kind: "polyline", id: "line:0", group: "A", color: "#1768ac", width: 1.5, lineStyle: "solid",
  markerSize: 3, axis: "y", x: [0, 1, 2], y: [1, 3, 2], decimation: null,
};

describe("classic chart option", () => {
  it("draws nothing until the analysis brings a plan", () => {
    expect(buildChartOption({ spec: spec("line") })).toBeNull();
    // An invalid surface grid is analysed but has no plan.
    expect(buildChartOption({ analysis: analysisOf(null), spec: spec("surface3d") })).toBeNull();
  });

  it("draws a line plan with one tooltip per x and the plan's legend", () => {
    const planned = plan("line", [panel([line], { legend: [{ label: "A", color: "#1768ac", symbol: "line" }] })]);
    const option = buildChartOption({ analysis: analysisOf(planned), spec: spec("line") }) as Option;
    expect(option.series[0]).toMatchObject({ type: "line", data: [[0, 1], [1, 3], [2, 2]] });
    expect(option.tooltip.trigger).toBe("axis");
    expect(option.legend.data).toEqual([{ name: "A", icon: "roundRect" }]);
    expect(option.dataZoom).toEqual([]);
    expect(option.title.text).toBe(defaultChartSpec.title);
  });

  it("hides the legend a box plan does not have", () => {
    const box: RenderPlanMark = { kind: "box", id: "box:0", group: "A", color: "#000000", medianColor: "#ff7f0e", width: 1, position: 1, halfWidth: 0.25, low: 1, q1: 2, median: 3, q3: 4, high: 5, outliers: [] };
    const option = buildChartOption({ analysis: analysisOf(plan("box", [panel([box])])), spec: spec("box") }) as Option;
    expect(option.legend).toEqual({ show: false });
  });

  it("keeps a heatmap's room for its colour scale and lets the viewer hide the scale", () => {
    const cells: RenderPlanMark = {
      kind: "mesh", id: "heatmap:1", group: "Correlation", shape: "cells", x: [0, 1], y: [0, 1],
      values: [[1, 0.2], [0.2, 1]], colorDomain: [-1, 1], palette: ["#3b4cc0", "#dddddd", "#b40426"],
      colorTitle: "Correlation", decimation: null,
    };
    const planned = plan("heatmap", [panel([cells], { axes: { x: axis([0, 1], [-0.5, 1.5]), y: axis([0, 1], [1.5, -0.5]), y2: null, z: null } })]);
    const shown = buildChartOption({ analysis: analysisOf(planned), spec: spec("heatmap") }) as Option;
    expect(shown.grid[0].right).toBe(96);
    expect(shown.visualMap).toEqual([expect.objectContaining({ min: -1, max: 1, show: true })]);
    const hidden = buildChartOption({ analysis: analysisOf(planned), showLegend: false, spec: spec("heatmap") }) as Option;
    expect(hidden.visualMap[0].show).toBe(false);
  });

  it("passes the camera and the low-cost limit to a surface plan", () => {
    const grid = [0, 1, 2, 3, 4];
    const surface: RenderPlanMark = {
      kind: "mesh", id: "surface:1", group: "Z", shape: "surface", x: grid, y: grid,
      values: grid.map((y) => grid.map((x) => x + y)), colorDomain: [0, 8], palette: ["#440154", "#fde725"],
      colorTitle: "", decimation: null,
    };
    const planned = plan("surface3d", [panel([surface], { axes: { x: axis([0, 2, 4]), y: axis([0, 2, 4]), y2: null, z: axis([0, 4, 8], [-0.4, 8.4]) } })]);
    const option = buildChartOption({
      analysis: analysisOf(planned), spec: spec("surface3d"), surfacePointLimit: 9,
      surfaceView: { alpha: 10, beta: 60, distance: 150 },
    }) as Option;
    expect(option.grid3D.viewControl).toMatchObject({ alpha: 10, beta: 60, distance: 150 });
    expect(option.series[0]).toMatchObject({ type: "surface", dataShape: [3, 3] });
    expect(option.visualMap).toMatchObject({ min: 0, max: 8 });
  });

  it("follows the page's dark theme without changing the plan's colours", () => {
    const option = buildChartOption({ analysis: analysisOf(plan("line", [panel([line])])), colorMode: "dark", spec: spec("line") }) as Option;
    expect(option.backgroundColor).toBe("#2F2F2F");
    expect(option.textStyle.color).toBe("#ECECEC");
    expect(option.series[0].lineStyle.color).toBe("#1768ac");
  });

  it("names plan statistics in the reader's language", () => {
    const summary: RenderPlanMark = { kind: "summary", id: "summary:0", group: "A", color: "#1768ac", width: 2, x0: 0, x1: 1, y: 1, statistic: "mean", summaryUnit: "sample", n: 2 };
    const analysis = analysisOf(plan("bar", [panel([summary])]));
    const tooltip = (zh: boolean) =>
      (buildChartOption({ analysis, spec: spec("bar"), zh }) as Option).tooltip.formatter({ seriesIndex: 0, dataIndex: 0 });
    expect(tooltip(true)).toContain("均值");
    expect(tooltip(false)).toContain("Mean");
  });
});

describe("chart render contract", () => {
  it("keeps shared palettes and line styles versioned", () => {
    expect(chartRenderContract.contractVersion).toBe("chart-render-v1");
    expect(chartSeriesColor({ configuredColor: "#FF0000", grayscale: true, grouped: true, index: 1 })).toBe("#5B616B");
    expect(chartSeriesColor({ configuredColor: "#FF0000", grayscale: false, grouped: true, index: 1 })).toBe("#0F766E");
    expect(chartLineType("dashdot")).toEqual([8, 4, 2, 4]);
  });
});
