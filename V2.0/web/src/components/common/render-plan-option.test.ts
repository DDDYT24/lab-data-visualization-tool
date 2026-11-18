import { describe, expect, it } from "vitest";

import placement from "../../../../contracts/fixtures/v24/plan-placement.json";
import { axisPosition, axisValue } from "@/domain/render-plan";

import type { RenderPlan, RenderPlanMark } from "@/domain/render-plan";

import { zoomedWindow } from "./render-plan-chart";
import {
  buildPlanOption, buildSurfacePlanOption, planAxisTooltip, planSelection, planTooltip,
} from "./render-plan-option";

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
      z: null,
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
    const swapped = { ...plan, panels: [{ ...panel!, axes: { x: { ...x, grid: true }, y: { ...y, grid: false }, y2: null, z: null } }] };
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

describe("mesh marks", () => {
  const labels = axis([[0, "A"], [1, "B"]], "category");
  const cells: RenderPlanMark = {
    kind: "mesh", id: "heatmap:1", group: "Correlation", shape: "cells", x: [0, 1], y: [0, 1],
    values: [[1, -0.5], [null, 1]], colorDomain: [-1, 1], palette: ["#3b4cc0", "#dddddd", "#b40426"],
    colorTitle: "Correlation", decimation: null,
  };
  const heatmap: RenderPlan = {
    ...plan, source: { ...plan.source, kind: "heatmap" },
    panels: [{
      ...plan.panels[0]!, legend: [], marks: [cells],
      axes: { x: { ...labels, domain: [-0.5, 1.5] }, y: { ...labels, domain: [1.5, -0.5] }, y2: null, z: null },
    }],
  };

  it("draws heatmap cells with the plan's colour domain and its first row on top", () => {
    const built = buildPlanOption({ plan: heatmap, text }) as Series;
    expect(built.series[0].data).toEqual([[0, 0, 1], [1, 0, -0.5], [1, 1, 1]]);
    expect(built.visualMap).toEqual([expect.objectContaining({
      seriesIndex: 0, dimension: 2, min: -1, max: 1, inRange: { color: cells.palette }, text: ["1", "−1"],
    })]);
    expect(built.yAxis[0]).toMatchObject({ min: -0.5, max: 1.5, inverse: true });
    expect(built.xAxis[0]).toMatchObject({ inverse: false, axisLabel: { rotate: 45 } });
    const coord = (point: number[]) => [point[0]! * 10, point[1]! * 10];
    const value = (dimension: number) => [1, 0, -0.5][dimension]!;
    const item = built.series[0].renderItem({}, { coord, value, visual: () => "#dddddd" });
    expect(item).toMatchObject({ shape: { x: 5, y: -5, width: 10, height: 10 }, style: { fill: "#dddddd" } });
    expect(planTooltip(heatmap, 0, 1, text)).toBe("Correlation<br/>−0.5");
  });

  // 101 x 101 with a peak at (1, 1) that the low-cost grid leaves out.
  const side = 101;
  const grid = Array.from({ length: side }, (_, index) => index);
  const surface: RenderPlan = {
    ...plan, source: { ...plan.source, kind: "surface3d" },
    panels: [{
      ...plan.panels[0]!, legend: [],
      marks: [{
        kind: "mesh", id: "surface:1", group: "Z", shape: "surface", x: grid, y: grid,
        values: grid.map((y) => grid.map((x) => (x === 1 && y === 1 ? 999 : x + y))),
        colorDomain: [0, 999], palette: ["#440154", "#fde725"], colorTitle: "", decimation: null,
      }],
      axes: {
        x: axis([[0, "0"], [100, "100"]], "linear"), y: axis([[0, "0"], [100, "100"]], "linear"), y2: null,
        z: { ...axis([[0, "0"], [1000, "1000"]], "linear"), domain: [-50, 1050], title: "Z" },
      },
    }],
  };
  const view = { alpha: 25, beta: 40, distance: 200 };

  it("draws every surface point unless the browser asks for fewer", () => {
    const full = buildSurfacePlanOption({ plan: surface, view }) as Series;
    expect(full.series[0].data).toHaveLength(side * side);
    expect(full.series[0].dataShape).toEqual([side, side]);
    expect(full.zAxis3D).toMatchObject({ min: -50, max: 1050, name: "Z" });
    expect([0, 1000, -50, 1050].map((value) => full.zAxis3D.axisLabel.formatter(value))).toEqual(["0", "1000", "", ""]);
    expect(full.grid3D.viewControl).toMatchObject(view);
  });

  it("keeps a rectangular grid, its edges and the full colour scale on the low-cost limit", () => {
    const reduced = buildSurfacePlanOption({ plan: surface, pointLimit: 800, view }) as Series;
    const [{ data, dataShape }] = reduced.series;
    expect(data.length).toBeLessThanOrEqual(800);
    expect(data.length).toBe(dataShape[0] * dataShape[1]);
    expect(data[0]).toEqual([0, 0, 0]);
    expect(data.at(-1)).toEqual([100, 100, 200]);
    expect(data.some((point: number[]) => point[2] === 999)).toBe(false);
    expect(reduced.visualMap).toMatchObject({ min: 0, max: 999, dimension: 2, inRange: { color: ["#440154", "#fde725"] } });
  });

  it("hides axis names and the colour scale when the viewer turns them off", () => {
    const bare = buildSurfacePlanOption({ plan: surface, showAxes: false, showLegend: false, view }) as Series;
    expect([bare.xAxis3D.name, bare.visualMap.show]).toEqual(["", false]);
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

describe("log and symlog axes", () => {
  const scaledAxis = (scale: "log10" | "symlog", domain: [number, number], ticks: number[]) => ({
    scale, domain, title: "mV", niceRule: null, grid: true, ticks: ticks.map((value) => ({ value, label: String(value) })),
  });
  const points: RenderPlanMark = { ...base, kind: "points", id: "points:0", marker: "o", size: 4, opacity: 1, axis: "y", x: [1, 2], y: [1, 1000], sourceRefs: ["o1", "o2"], decimation: null };
  const error: RenderPlanMark = { ...base, kind: "errorbar", id: "error:0", width: 1, capSize: 2, opacity: 1, axis: "y", x: [1], y: [100], error: [90] };
  const segments: RenderPlanMark = { ...base, kind: "segments", id: "pairs:0", width: 1, opacity: 1, x0: [1], y0: [10], x1: [2], y1: [100], sourceRefs: [["o1", "o2"]] };
  const logged = (scale: "log10" | "symlog", list: RenderPlanMark[], domain: [number, number] = [0.1, 10000]): RenderPlan => ({
    ...plan,
    panels: [{
      ...plan.panels[0]!, marks: list,
      axes: { x: axis([[1, "1"], [2, "2"]], "linear"), y: scaledAxis(scale, domain, [1, 10, 100, 1000]), y2: null, z: null },
    }],
  });

  it("states a log axis in decades with the plan's labels", () => {
    const { yAxis: [yAxis] } = buildPlanOption({ plan: logged("log10", [points]), text, zoomed: true }) as Series;
    expect(yAxis.type).toBe("value");
    expect([yAxis.min, yAxis.max]).toEqual([-1, 4]);
    expect(yAxis.axisLabel.customValues).toEqual([0, 1, 2, 3]);
    expect(yAxis.axisLabel.formatter(2)).toBe("100");
    expect(yAxis.axisLabel.formatter(0.5)).toBe("3.16228");
  });

  it("places points, segments and both error bar ends by the scale", () => {
    const built = buildPlanOption({ plan: logged("log10", [points, error, segments]), text, selected: ["o2"] });
    const [dots, bar, lines, selection] = built.series as Series[];
    expect(dots.data).toEqual([[1, 0], [2, 3]]);
    expect(lines.data).toEqual([{ coords: [[1, 1], [2, 2]] }]);
    expect(selection.data).toEqual([[2, 3]]);
    const { children } = bar.renderItem({ dataIndex: 0 }, { coord: (point: number[]) => point });
    expect(children[0].shape.y1).toBeCloseTo(Math.log10(190));
    expect(children[0].shape.y2).toBeCloseTo(1);
  });

  it("keeps real values in the tooltip", () => {
    expect(planTooltip(logged("log10", [points]), 0, 1, text)).toContain("1000");
  });

  it("shows real values in the shared tooltip, which is fed the drawn positions", () => {
    const noted = logged("log10", [points]);
    const [dots] = buildPlanOption({ plan: noted, text }).series as Series[];
    const tooltip = planAxisTooltip(noted)([{ seriesName: "A", seriesIndex: 0, value: dots.data[1] }]);
    expect(tooltip).toBe("2<br/>A: 1000");
  });

  it("is linear near zero on a symlog axis and mirrors negatives", () => {
    const mirrored: RenderPlanMark = { ...points, y: [-9, 9] };
    const built = buildPlanOption({ plan: logged("symlog", [mirrored], [-99, 99]), text });
    const [dots] = built.series as Series[];
    expect(dots.data[0][1]).toBeCloseTo(-1);
    expect(dots.data[1][1]).toBeCloseTo(1);
    const { yAxis: [yAxis] } = built as Series;
    expect(yAxis.min).toBeCloseTo(-2);
    expect(yAxis.max).toBeCloseTo(2);
  });

  it("leaves a linear plan's series untouched", () => {
    expect(series(2).data).toEqual([[0.1, 1], [1.2, 3]]);
  });
});

describe("annotations", () => {
  const frame = { x: 100, y: 20, width: 400, height: 300 };
  const identity = { coord: (point: number[]) => point };
  const notes: NonNullable<RenderPlan["panels"][number]["notes"]> = [
    { kind: "text", x: 1, y: 2, text: "peak" },
    { kind: "hline", value: 3, label: "limit" },
    { kind: "vline", value: 2, label: "" },
    { kind: "rect", x: [0, 1], y: [1, 3], label: "window", labelAt: [0, 3] },
    { kind: "bracket", x: [0, 2], y: 4, foot: 3.9, label: "p < 0.01", labelAt: [1, 4] },
  ];
  const noted: RenderPlan = { ...plan, panels: [{ ...plan.panels[0]!, notes }] };
  const built = buildPlanOption({ plan: noted, text }).series as Series[];
  const under = built.find((series) => series.id === "notes-under-A")!;
  const over = built.find((series) => series.id === "notes-A")!;
  const drawn = (index: number) => over.renderItem({ dataIndex: index, coordSys: frame }, identity).children;

  it("adds nothing to a plan without notes", () => {
    const ids = (buildPlanOption({ plan, text }).series as Series[]).map((series) => series.id);
    expect(ids.filter((id: string) => id.startsWith("notes"))).toEqual([]);
  });

  it("keeps every mark at its plan index and stays out of the tooltip", () => {
    expect(built.map((series) => series.id)).toEqual([...marks.map((mark) => mark.id), "notes-under-A", "notes-A", "selection"]);
    expect([under.silent, over.silent, under.tooltip.show, over.tooltip.show]).toEqual([true, true, false, false]);
    expect(under.z).toBeLessThan(Math.min(...built.slice(0, marks.length).map((series) => series.z)));
    expect(over.z).toBeGreaterThan(Math.max(...built.slice(0, marks.length).map((series) => series.z)));
  });

  it("draws text centred on its point", () => {
    const [label] = drawn(0);
    expect(label.style).toMatchObject({ text: "peak", x: 1, y: 2, align: "center", verticalAlign: "middle", fontSize: 12 });
  });

  it("runs reference lines across the whole panel", () => {
    const [across, label] = drawn(1);
    expect(across.shape).toEqual({ x1: 100, y1: 3, x2: 500, y2: 3 });
    expect(across.style.lineDash).toBe("dashed");
    expect(label.style).toMatchObject({ text: "limit", x: 496, y: -1, align: "right", verticalAlign: "bottom" });
    const up = drawn(2);
    expect(up).toHaveLength(1);
    expect(up[0].shape).toEqual({ x1: 2, y1: 20, x2: 2, y2: 320 });
  });

  it("shades a range under the marks and labels its top left corner", () => {
    const shade = under.renderItem({ dataIndex: 0 }, identity);
    expect(shade.shape).toEqual({ x: 0, y: 1, width: 1, height: 2 });
    expect(shade.style.opacity).toBe(0.12);
    expect(drawn(3)[0].style).toMatchObject({ text: "window", x: 4, y: 7, align: "left", verticalAlign: "top" });
  });

  it("draws a bracket down to the planned feet with its label above the middle", () => {
    const [line, label] = drawn(4);
    expect(line.shape.points).toEqual([[0, 3.9], [0, 4], [2, 4], [2, 3.9]]);
    expect(label.style).toMatchObject({ text: "p < 0.01", x: 1, y: 0, align: "center", verticalAlign: "bottom" });
  });

  it("places notes by the axis scale", () => {
    const logAxis = {
      scale: "log10" as const, domain: [1, 1000] as [number, number], title: "mV", niceRule: null, grid: true,
      ticks: [1, 10, 100, 1000].map((value) => ({ value, label: String(value) })),
    };
    const logged: RenderPlan = {
      ...plan,
      panels: [{
        ...plan.panels[0]!, marks: [], axes: { ...plan.panels[0]!.axes, y: logAxis },
        notes: [{ kind: "bracket", x: [0, 2], y: 100, foot: 10, label: "", labelAt: [1, 100] }],
      }],
    };
    const [series] = buildPlanOption({ plan: logged, text }).series as Series[];
    expect(series.data).toEqual([[0, 2]]);
    const [line] = series.renderItem({ dataIndex: 0, coordSys: frame }, identity).children;
    expect(line.shape.points).toEqual([[0, 1], [0, 2], [2, 2], [2, 1]]);
  });
});

describe("placement fixture shared with the API tests", () => {
  const fixture = placement as unknown as {
    positions: { scale: "linear" | "log10" | "symlog"; value: number; position: number }[];
    plan: RenderPlan;
    placed: { points: number[][]; bracket: number[][]; bracketLabel: number[]; rectLabel: number[] };
    logHistogram: { plan: RenderPlan; placed: number[][] };
    layered: { plan: RenderPlan; drawOrder: string[] };
  };

  it("draws a layered panel's marks in the order the export draws them, notes above all", () => {
    const { plan: layeredPlan, drawOrder } = fixture.layered;
    const marks = layeredPlan.panels[0]!.marks;
    const series = buildPlanOption({ plan: layeredPlan, text }).series as (Series & { z: number })[];
    const order = marks.map((mark, index) => ({ id: mark.id, z: series[index]!.z, index }))
      .sort((a, b) => a.z - b.z || a.index - b.index).map((item) => item.id);
    expect(order).toEqual(drawOrder);
    const noted = { ...layeredPlan, panels: [{ ...layeredPlan.panels[0]!, notes: [{ kind: "hline" as const, value: 5, label: "" }] }] };
    const withNotes = buildPlanOption({ plan: noted, text }).series as (Series & { z: number })[];
    const top = withNotes.find((item) => item.id === `notes-${layeredPlan.panels[0]!.id}`)!;
    expect(top.z).toBeGreaterThan(Math.max(...withNotes.slice(0, marks.length).map((item) => item.z)));
  });
  const close = (got: number[][], want: number[][]) => got.forEach((point, i) =>
    point.forEach((value, j) => expect(value).toBeCloseTo(want[i]![j]!, 9)));

  it("draws a log-count histogram's bins from the axis floor, as the API places them", () => {
    const { plan: histogram, placed } = fixture.logHistogram;
    const series = buildPlanOption({ plan: histogram, text }).series as Series[];
    const bins = series.find((item) => String(item.id).startsWith("histogram:"))!;
    const drawn = placed.map((_, index) => bins.renderItem(
      { dataIndex: index }, { coord: (point: number[]) => point },
    ).shape as { x: number; y: number; width: number; height: number });
    close(drawn.map((rect) => [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height]), placed);
    expect(drawn.filter((rect) => rect.height === 0).length).toBeGreaterThan(0);
  });

  it("places sample values where the API places them, and back", () => {
    for (const { scale, value, position } of fixture.positions) {
      expect(axisPosition(scale, value)).toBeCloseTo(position, 12);
      expect(axisValue(scale, position)).toBeCloseTo(value, 6 - Math.ceil(Math.log10(Math.abs(value) + 1)));
    }
  });

  it("draws the fixture plan's points and notes at the positions the API states", () => {
    const series = buildPlanOption({ plan: fixture.plan, text }).series as Series[];
    const dots = series.find((item) => item.type === "scatter" && item.id !== "selection")!;
    close(dots.data, fixture.placed.points);
    const notes = series.find((item) => String(item.id).startsWith("notes-") && !String(item.id).startsWith("notes-under"))!;
    const frame = { x: 0, y: 0, width: 1, height: 1 };
    const draw = (index: number) => notes.renderItem({ dataIndex: index, coordSys: frame }, { coord: (point: number[]) => point }).children;
    const [line, label] = draw(4);
    close(line.shape.points, fixture.placed.bracket);
    expect(label.style.x).toBeCloseTo(fixture.placed.bracketLabel[0]!, 9);
    const [window] = draw(3);
    expect(window.style.x - 4).toBeCloseTo(fixture.placed.rectLabel[0]!, 9);
    expect(window.style.y - 4).toBeCloseTo(fixture.placed.rectLabel[1]!, 9);
  });
});
