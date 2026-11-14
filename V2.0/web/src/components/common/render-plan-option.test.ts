import { describe, expect, it } from "vitest";

import type { RenderPlan, RenderPlanMark } from "@/domain/render-plan";

import { zoomedWindow } from "./render-plan-chart";
import { buildPlanOption, planAxisTooltip, planSelection, planTooltip } from "./render-plan-option";

const text = { mean: "Mean", median: "Median" };
const axis = (ticks: [number, string][], scale: "linear" | "category") => ({
  scale, domain: [-0.5, 4.5] as [number, number], title: scale === "linear" ? "mV" : "",
  niceRule: scale === "linear" ? "nice-1-2-5-v1" as const : null, grid: scale === "linear",
  ticks: ticks.map(([value, label]) => ({ value, label })),
});
const base = { group: "A <b>", color: "#1768ac" };
const marks: RenderPlanMark[] = [
  { ...base, kind: "violin", id: "density:0", opacity: 0.3, position: 0, grid: [1, 2, 3], halfWidths: [0.1, 0.4, 0.1], bandwidth: 0.5, bandwidthRule: "scott-sample-sd", scale: "equal_width", n: 3, sourceRefs: ["o1", "o2", "o3"] },
  { ...base, kind: "segments", id: "pairs:0", width: 1.2, opacity: 0.65, x0: [0], y0: [1], x1: [1], y1: [3], sourceRefs: [["o1", "o2"]] },
  { ...base, kind: "points", id: "points:0", marker: "v", size: 4, opacity: 0.7, axis: "y", x: [0.1, 1.2], y: [1, 3], sourceRefs: ["o1", "o2"], decimation: null },
  { ...base, kind: "summary", id: "summary:0", width: 2.4, x0: -0.25, x1: 0.25, y: 2, statistic: "mean", summaryUnit: "sample", n: 2 },
];
const plan: RenderPlan = {
  schemaVersion: 1, contract: "render-plan-v1", planHash: "0".repeat(64),
  source: { kind: "grouped_strip", resultSha256: "1".repeat(64), algorithmVersion: "scientific-charts-v1" },
  geometry: { contract: "chart-render-v2", axis: "nice-1-2-5-v1", jitter: "sha256-uniform-v1", density: null, histogram: null, quantile: null, decimation: null },
  rendering: { pointRendering: "auto", rasterizePoints: false },
  disclosures: [],
  panels: [{
    id: "A", title: "Chart", marks,
    legend: [{ label: "A <b>", color: "#1768ac", symbol: "line" }],
    axes: {
      x: axis([[0, "A\nobs=2"], [1, "B"]], "category"),
      y: axis([[0, "0"], [2, "2"], [4, "4"]], "linear"),
      y2: null,
    },
  }],
};

type Series = any; // eslint-disable-line @typescript-eslint/no-explicit-any
const option = (extra: Partial<Parameters<typeof buildPlanOption>[0]> = {}) => buildPlanOption({ plan, text, ...extra });
const series = (index: number) => option().series[index] as Series;

const classicMarks: RenderPlanMark[] = [
  { ...base, kind: "polyline", id: "line:0", width: 1.5, lineStyle: "dashed", markerSize: 3, axis: "y", x: [0, 1], y: [1, 2], decimation: null },
  { ...base, kind: "polyline", id: "fit:0", width: 1, lineStyle: "dashed", markerSize: 0, axis: "y2", x: [0, 1], y: [5, 6], decimation: null },
  { ...base, kind: "band", id: "band:0", opacity: 0.15, axis: "y", x: [0, 1], lower: [0, 1], upper: [2, 3] },
  { ...base, kind: "errorbar", id: "error:0", width: 1, capSize: 2, opacity: 0.9, axis: "y", x: [1], y: [2], error: [0.5] },
];
const classic: RenderPlan = {
  ...plan,
  panels: [{
    ...plan.panels[0]!, marks: classicMarks,
    legend: [{ label: "fit", color: "#1768ac", symbol: "dashed" }],
    axes: { ...plan.panels[0]!.axes, y2: { ...axis([[5, "5"], [6, "6"]], "linear"), grid: false } },
  }],
};

describe("classic marks", () => {
  const built = buildPlanOption({ plan: classic, text }) as Series;

  it("draws a polyline with its style and marks hidden when it has none", () => {
    expect(built.series[0]).toMatchObject({
      type: "line", showSymbol: true, symbolSize: 6, data: [[0, 1], [1, 2]], yAxisIndex: 0,
      lineStyle: { color: "#1768ac", width: 1.5, type: "dashed" },
    });
    expect(built.series[1]).toMatchObject({ showSymbol: false, silent: true, yAxisIndex: 1 });
  });

  it("gives each panel a right-hand axis only when it has one", () => {
    expect(built.yAxis).toHaveLength(2);
    expect(built.yAxis[1]).toMatchObject({ position: "right", min: -0.5, max: 4.5 });
    expect(built.yAxis[1].splitLine.show).toBe(false);
    expect(built.dataZoom[1].yAxisIndex).toEqual([0, 1]);
  });

  it("outlines a band between its limits", () => {
    const drawn = built.series[2].renderItem({}, { coord: (point: number[]) => point });
    expect(drawn.shape.points).toEqual([[0, 2], [1, 3], [1, 1], [0, 0]]);
    expect(drawn.style).toMatchObject({ fill: "#1768ac", opacity: 0.15 });
  });

  it("draws an error bar with two caps", () => {
    const drawn = built.series[3].renderItem({ dataIndex: 0 }, { coord: (point: number[]) => point });
    expect(drawn.children[0].shape).toEqual({ x1: 1, y1: 2.5, x2: 1, y2: 1.5 });
    expect(drawn.children[1].shape).toEqual({ x1: -2, y1: 2.5, x2: 4, y2: 2.5 });
  });

  it("gives fits a dashed legend symbol", () => {
    expect(built.legend.data[0].icon).toContain("path://");
  });

  it("selects nothing for points that have no observations", () => {
    const points: RenderPlanMark = { ...base, kind: "points", id: "scatter:0", marker: "o", size: 4, opacity: 0.8, axis: "y", x: [1], y: [2], sourceRefs: [], decimation: null };
    const scatter = { ...plan, panels: [{ ...plan.panels[0]!, marks: [points] }] };
    expect(planSelection(scatter, 0, 0)).toEqual([]);
  });

  it("lists every series at one x in the shared tooltip", () => {
    const tooltip = planAxisTooltip(plan)([{ seriesName: "A <b>", value: [1, 2] }]);
    expect(tooltip).toBe("B<br/>A &#60;b&#62;: 2");
  });
});

describe("render plan adapter", () => {
  it("draws axes exactly as the plan states them", () => {
    const { xAxis: [xAxis], yAxis: [yAxis] } = option() as Series;
    expect([yAxis.min, yAxis.max]).toEqual([-0.5, 4.5]);
    expect(yAxis.axisLabel.customValues).toEqual([0, 2, 4]);
    expect(yAxis.axisTick.customValues).toEqual([0, 2, 4]);
    expect(yAxis.axisLabel.formatter(2)).toBe("2");
    expect(yAxis.name).toBe("mV");
    expect(xAxis.axisLabel.formatter(0)).toBe("A\nobs=2");
    expect(xAxis.splitLine.show).toBe(false);
    expect(yAxis.splitLine.show).toBe(true);
  });

  it("draws a grid only on the axes whose plan asks for one", () => {
    const [panel] = plan.panels;
    const { x, y } = panel!.axes;
    const swapped = { ...plan, panels: [{ ...panel!, axes: { x: { ...x, grid: true }, y: { ...y, grid: false }, y2: null } }] };
    const { xAxis: [xAxis], yAxis: [yAxis] } = buildPlanOption({ plan: swapped, text }) as Series;
    expect([xAxis.splitLine.show, yAxis.splitLine.show]).toEqual([true, false]);
  });

  it("keeps category ticks but lets a zoomed value axis label its window", () => {
    const { xAxis: [xAxis], yAxis: [yAxis] } = option({ zoomed: true }) as Series;
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

  it("draws rects as the plan's intervals, whichever way they grow", () => {
    const rects: RenderPlanMark = { ...base, kind: "rects", id: "bar:0", opacity: 0.5, width: 1, shape: "bars", axis: "y", x0: [0, 2], x1: [1, 3], y0: [0, 0], y1: [4, -2], decimation: null };
    const drawn = buildPlanOption({ plan: { ...plan, panels: [{ ...plan.panels[0]!, marks: [rects] }] }, text }).series[0] as Series;
    const coord = (point: number[]) => [point[0]! * 10, 100 - point[1]! * 10];
    expect(drawn.renderItem({ dataIndex: 0 }, { coord }).shape).toEqual({ x: 0, y: 60, width: 10, height: 40 });
    expect(drawn.renderItem({ dataIndex: 1 }, { coord }).shape).toEqual({ x: 20, y: 100, width: 10, height: 20 });
    expect(drawn.renderItem({ dataIndex: 0 }, { coord }).style).toEqual({ fill: "#1768ac", opacity: 0.5 });
    expect(planTooltip({ ...plan, panels: [{ ...plan.panels[0]!, marks: [rects] }] }, 0, 1, text)).toBe("A &#60;b&#62;<br/>\u22122");
  });

  it("draws a box from the API's five numbers and its outliers", () => {
    const box: RenderPlanMark = { ...base, kind: "box", id: "box:0", color: "#000000", medianColor: "#ff7f0e", width: 1, position: 1, halfWidth: 0.25, low: 1, q1: 2, median: 3, q3: 4, high: 5, outliers: [9] };
    const boxed = { ...plan, panels: [{ ...plan.panels[0]!, marks: [box] }] };
    const drawn = buildPlanOption({ plan: boxed, text }).series[0] as Series;
    const { children } = drawn.renderItem({}, { coord: (point: number[]) => point });
    expect(children[0].shape).toEqual({ x: 0.75, y: 2, width: 0.5, height: 2 });
    expect(children.at(-2)).toMatchObject({ type: "line", shape: { y1: 3, y2: 3 }, style: { stroke: "#ff7f0e" } });
    expect(children.at(-1)).toMatchObject({ type: "circle", shape: { cx: 1, cy: 9 } });
    expect(planTooltip(boxed, 0, 0, text)).toBe("A &#60;b&#62;<br/>1 \u00b7 2 \u00b7 3 \u00b7 4 \u00b7 5");
  });

  it("gives each panel its own grid and axes", () => {
    const panels = [plan.panels[0]!, { ...plan.panels[0]!, id: "B", marks: [marks[2]!] }];
    const built = buildPlanOption({ plan: { ...plan, panels }, text }) as Series;
    expect(built.grid).toHaveLength(2);
    expect(built.xAxis.map((item: Series) => item.gridIndex)).toEqual([0, 1]);
    expect(built.series[4].yAxisIndex).toBe(1);
    expect(built.dataZoom[1].yAxisIndex).toEqual([0, 1]);
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
