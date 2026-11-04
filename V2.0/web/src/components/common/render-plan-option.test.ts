import { describe, expect, it } from "vitest";

import type { RenderPlan, RenderPlanMark } from "@/domain/render-plan";

import { zoomedWindow } from "./render-plan-chart";
import { buildPlanOption, planSelection, planTooltip } from "./render-plan-option";

const text = { mean: "Mean", median: "Median" };
const axis = (ticks: [number, string][], scale: "linear" | "category") => ({
  scale, domain: [-0.5, 4.5] as [number, number], title: scale === "linear" ? "mV" : "",
  niceRule: scale === "linear" ? "nice-1-2-5-v1" as const : null,
  ticks: ticks.map(([value, label]) => ({ value, label })),
});
const base = { group: "A <b>", color: "#1768ac" };
const marks: RenderPlanMark[] = [
  { ...base, kind: "violin", id: "density:0", opacity: 0.3, position: 0, grid: [1, 2, 3], halfWidths: [0.1, 0.4, 0.1], bandwidth: 0.5, bandwidthRule: "scott-sample-sd", scale: "equal_width", n: 3, sourceRefs: ["o1", "o2", "o3"] },
  { ...base, kind: "segments", id: "pairs:0", width: 1.2, opacity: 0.65, x0: [0], y0: [1], x1: [1], y1: [3], sourceRefs: [["o1", "o2"]] },
  { ...base, kind: "points", id: "points:0", marker: "v", size: 4, opacity: 0.7, x: [0.1, 1.2], y: [1, 3], sourceRefs: ["o1", "o2"] },
  { ...base, kind: "summary", id: "summary:0", width: 2.4, x0: -0.25, x1: 0.25, y: 2, statistic: "mean", summaryUnit: "sample", n: 2 },
];
const plan: RenderPlan = {
  schemaVersion: 1, contract: "render-plan-v1", planHash: "0".repeat(64),
  source: { kind: "grouped_strip", resultSha256: "1".repeat(64), algorithmVersion: "scientific-charts-v1" },
  geometry: { contract: "chart-render-v2", axis: "nice-1-2-5-v1", jitter: "sha256-uniform-v1", density: null },
  rendering: { pointRendering: "auto", rasterizePoints: false },
  disclosures: [],
  panels: [{
    id: "A", title: "Chart", marks,
    legend: [{ label: "A <b>", color: "#1768ac", symbol: "line" }],
    axes: {
      x: axis([[0, "A\nobs=2"], [1, "B"]], "category"),
      y: axis([[0, "0"], [2, "2"], [4, "4"]], "linear"),
    },
  }],
};

type Series = any; // eslint-disable-line @typescript-eslint/no-explicit-any
const option = (extra: Partial<Parameters<typeof buildPlanOption>[0]> = {}) => buildPlanOption({ plan, text, ...extra });
const series = (index: number) => option().series[index] as Series;

describe("render plan adapter", () => {
  it("draws axes exactly as the plan states them", () => {
    const { xAxis, yAxis } = option();
    expect([yAxis.min, yAxis.max]).toEqual([-0.5, 4.5]);
    expect(yAxis.axisLabel.customValues).toEqual([0, 2, 4]);
    expect(yAxis.axisTick.customValues).toEqual([0, 2, 4]);
    expect(yAxis.axisLabel.formatter(2)).toBe("2");
    expect(yAxis.name).toBe("mV");
    expect(xAxis.axisLabel.formatter(0)).toBe("A\nobs=2");
    expect(xAxis.splitLine.show).toBe(false);
  });

  it("keeps category ticks but lets a zoomed value axis label its window", () => {
    const { xAxis, yAxis } = option({ zoomed: true });
    expect(xAxis.axisLabel.customValues).toEqual([0, 1]);
    expect(yAxis.axisLabel.customValues).toBeUndefined();
    expect(yAxis.axisLabel.formatter(-1.23456789e-8)).toBe("−1.23457e−8");
  });

  it("draws a violin as the outline of its grid and half widths", () => {
    const violin = series(0);
    expect(violin.type).toBe("custom");
    const drawn = violin.renderItem({}, { coord: (point: number[]) => point });
    expect(drawn.shape.points).toEqual([
      [-0.1, 1], [-0.4, 2], [-0.1, 3], [0.1, 3], [0.4, 2], [0.1, 1],
    ]);
    expect(drawn.style).toMatchObject({ fill: "#1768ac", opacity: 0.3 });
  });

  it("draws one line per segment", () => {
    expect(series(1).type).toBe("lines");
    expect(series(1).data).toEqual([{ coords: [[0, 1], [1, 3]] }]);
    expect(series(1).lineStyle).toEqual({ color: "#1768ac", width: 1.2, opacity: 0.65 });
  });

  it("draws points with the plan's marker, colour and coordinates", () => {
    expect(series(2)).toMatchObject({
      type: "scatter", symbol: "triangle", symbolRotate: 180, symbolSize: 8,
      data: [[0.1, 1], [1.2, 3]],
      itemStyle: { color: "#1768ac", opacity: 0.7 },
    });
  });

  it("draws a summary above the points", () => {
    expect(series(3).data).toEqual([{ coords: [[-0.25, 2], [0.25, 2]] }]);
    expect(series(3).z).toBeGreaterThan(series(2).z);
  });

  it("links a drawn item to its source observations", () => {
    expect(planSelection(plan, 2, 1)).toEqual(["o2"]);
    expect(planSelection(plan, 1, 0)).toEqual(["o1", "o2"]);
    expect(planSelection(plan, 0, 0)).toEqual([]);
    expect(planSelection(plan, 3, 0)).toEqual([]);
    expect(planSelection(plan, 9, 0)).toEqual([]);
  });

  it("marks selected observations on the chart", () => {
    const selection = option({ selected: ["o2", "gone"] }).series[4] as Series;
    expect(selection.id).toBe("selection");
    expect(selection.data).toEqual([[1.2, 3]]);
    expect((option().series[4] as Series).data).toEqual([]);
  });

  it("escapes data in tooltips", () => {
    expect(planTooltip(plan, 2, 1, text)).toBe("A &#60;b&#62;<br/>3");
    expect(planTooltip(plan, 1, 0, text)).toBe("A &#60;b&#62;<br/>1 → 3");
    expect(planTooltip(plan, 3, 0, text)).toBe("A &#60;b&#62;<br/>Mean = 2<br/>n = 2");
    expect(planTooltip(plan, 0, 0, text)).toContain("n = 3<br/>h = 0.5 (scott-sample-sd)");
    expect(planTooltip(plan, 9, 0, text)).toBe("");
  });

  it("hides the legend when the plan has none", () => {
    const bare = { ...plan, panels: [{ ...plan.panels[0]!, legend: [] }] };
    expect(buildPlanOption({ plan: bare, text }).legend.show).toBe(false);
  });
});

describe("zoom window", () => {
  it("narrows and widens around its middle and stays inside the data", () => {
    expect(zoomedWindow({ start: 0, end: 100 }, -25)).toEqual({ start: 12.5, end: 87.5 });
    expect(zoomedWindow({ start: 12.5, end: 87.5 }, 25)).toEqual({ start: 0, end: 100 });
    expect(zoomedWindow({ start: 0, end: 10 }, 25)).toEqual({ start: 0, end: 35 });
    expect(zoomedWindow({ start: 49, end: 51 }, -25)).toEqual({ start: 49, end: 51 });
  });
});
