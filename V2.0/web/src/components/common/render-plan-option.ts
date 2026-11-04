import { gridColor } from "@/domain/chart-render-contract";
import type { RenderPlan, RenderPlanMark, RenderPlanPanel } from "@/domain/render-plan";

// The adapter only draws what the plan states: it computes no statistics, axes or geometry.
const TEXT_COLOR = "#20252d";
const POINT_PIXELS_PER_SIZE = 2;
const LABEL_PIXELS = 8;
const LONG_AXIS_TICKS = 4;

const SYMBOLS = {
  o: { symbol: "circle", symbolRotate: 0 },
  s: { symbol: "rect", symbolRotate: 0 },
  "^": { symbol: "triangle", symbolRotate: 0 },
  D: { symbol: "diamond", symbolRotate: 0 },
  v: { symbol: "triangle", symbolRotate: 180 },
} as const;

type PlanAxis = RenderPlanPanel["axes"]["x"];
type Coordinate = [number, number];
type RenderApi = { coord: (point: number[]) => number[] };
export type PlanText = { mean: string; median: string };
export const planText = (zh: boolean): PlanText =>
  zh ? { mean: "均值", median: "中位数" } : { mean: "Mean", median: "Median" };
export type PlanColors = { text: string; grid: string };
const DEFAULT_COLORS: PlanColors = { text: TEXT_COLOR, grid: gridColor };
export const PLAN_ZOOM_IDS = ["zoom-x", "zoom-y"] as const;

const escapeHtml = (value: string) => value.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const shown = (value: number) =>
  Number(value.toPrecision(6)).toString().replace(/-/g, "−");

function tooltipLines(mark: RenderPlanMark, index: number, text: PlanText) {
  switch (mark.kind) {
    case "points":
      return [mark.group, shown(mark.y[index]!)];
    case "segments":
      return [mark.group, `${shown(mark.y0[index]!)} → ${shown(mark.y1[index]!)}`];
    case "violin":
      return [mark.group, `n = ${mark.n}`, `h = ${shown(mark.bandwidth)} (${mark.bandwidthRule})`];
    case "summary":
      return [mark.group, `${text[mark.statistic]} = ${shown(mark.y)}`, `n = ${mark.n}`];
    case "rects":
      return mark.shape === "stairs"
        ? [mark.group, `${shown(mark.x0[index]!)} \u2013 ${shown(mark.x1[index]!)}`, shown(mark.y1[index]!)]
        : [mark.group, shown(mark.y1[index]!)];
    case "box":
      return [mark.group, [mark.low, mark.q1, mark.median, mark.q3, mark.high].map(shown).join(" \u00b7 ")];
  }
}

const planMarks = (plan: RenderPlan) => plan.panels.flatMap((panel) => panel.marks);

export function planTooltip(plan: RenderPlan, seriesIndex: number, index: number, text: PlanText) {
  const mark = planMarks(plan)[seriesIndex];
  return mark ? tooltipLines(mark, index, text).map(escapeHtml).join("<br/>") : "";
}

/** The observations one drawn item stands for; violins and summaries have no single row. */
export function planSelection(plan: RenderPlan, seriesIndex: number, index: number): string[] {
  const mark = planMarks(plan)[seriesIndex];
  if (mark?.kind === "points") return [mark.sourceRefs[index]!];
  if (mark?.kind === "segments") return [...mark.sourceRefs[index]!];
  return [];
}

function selectedCoordinates(plan: RenderPlan, selected: readonly string[]) {
  const wanted = new Set(selected);
  const found: Coordinate[] = [];
  for (const mark of planMarks(plan)) {
    if (mark.kind !== "points") continue;
    mark.sourceRefs.forEach((id, index) => {
      if (wanted.has(id)) found.push([mark.x[index]!, mark.y[index]!]);
    });
  }
  return found;
}

function lineSeries(
  mark: { id: string; group: string; color: string; width: number },
  coords: [Coordinate, Coordinate][],
  opacity: number,
  z: number,
) {
  return {
    id: mark.id, name: mark.group, type: "lines", coordinateSystem: "cartesian2d", z,
    data: coords.map((pair) => ({ coords: pair })),
    lineStyle: { color: mark.color, width: mark.width, opacity },
  };
}

function rectItem(start: number[], end: number[], fill: string, opacity: number) {
  return {
    type: "rect",
    shape: {
      x: Math.min(start[0]!, end[0]!), y: Math.min(start[1]!, end[1]!),
      width: Math.abs(end[0]! - start[0]!), height: Math.abs(end[1]! - start[1]!),
    },
    style: { fill, opacity },
  };
}

function boxItem(mark: Extract<RenderPlanMark, { kind: "box" }>, api: RenderApi) {
  const left = mark.position - mark.halfWidth;
  const right = mark.position + mark.halfWidth;
  const line = (a: number[], b: number[], stroke = mark.color) => {
    const [x1, y1] = api.coord(a) as [number, number];
    const [x2, y2] = api.coord(b) as [number, number];
    return { type: "line", shape: { x1, y1, x2, y2 }, style: { stroke, lineWidth: mark.width } };
  };
  const box = rectItem(api.coord([left, mark.q3]), api.coord([right, mark.q1]), "transparent", 1);
  return {
    type: "group",
    children: [
      { ...box, style: { ...box.style, stroke: mark.color, lineWidth: mark.width } },
      line([mark.position, mark.q3], [mark.position, mark.high]),
      line([mark.position, mark.q1], [mark.position, mark.low]),
      line([left, mark.high], [right, mark.high]),
      line([left, mark.low], [right, mark.low]),
      line([left, mark.median], [right, mark.median], mark.medianColor),
      ...mark.outliers.map((value) => {
        const [cx, cy] = api.coord([mark.position, value]) as [number, number];
        return { type: "circle", shape: { cx, cy, r: 3 }, style: { fill: "transparent", stroke: mark.color } };
      }),
    ],
  };
}

function markSeries(mark: RenderPlanMark) {
  switch (mark.kind) {
    case "rects":
      return {
        id: mark.id, name: mark.group, type: "custom", coordinateSystem: "cartesian2d", clip: true, z: 2,
        data: mark.x0.map((x, index) => [x, mark.y1[index]!]),
        itemStyle: { color: mark.color, opacity: mark.opacity },
        renderItem: (params: { dataIndex: number }, api: RenderApi) => rectItem(
          api.coord([mark.x0[params.dataIndex]!, mark.y1[params.dataIndex]!]),
          api.coord([mark.x1[params.dataIndex]!, mark.y0[params.dataIndex]!]),
          mark.color, mark.opacity,
        ),
      };
    case "box":
      return {
        id: mark.id, name: mark.group, type: "custom", coordinateSystem: "cartesian2d", clip: true, z: 2,
        data: [[mark.position, mark.median]],
        itemStyle: { color: mark.color },
        renderItem: (_params: unknown, api: RenderApi) => boxItem(mark, api),
      };
    case "points":
      return {
        id: mark.id, name: mark.group, type: "scatter", z: 4, ...SYMBOLS[mark.marker],
        symbolSize: mark.size * POINT_PIXELS_PER_SIZE,
        data: mark.x.map((x, index) => [x, mark.y[index]!]),
        itemStyle: { color: mark.color, opacity: mark.opacity },
        emphasis: { scale: 1.6 },
      };
    case "segments":
      return lineSeries(
        mark,
        mark.x0.map((x, index) => [[x, mark.y0[index]!], [mark.x1[index]!, mark.y1[index]!]]),
        mark.opacity,
        2,
      );
    case "summary":
      return lineSeries(mark, [[[mark.x0, mark.y], [mark.x1, mark.y]]], 1, 5);
    case "violin": {
      const left = mark.grid.map((y, index) => [mark.position - mark.halfWidths[index]!, y]);
      const right = mark.grid.map((y, index) => [mark.position + mark.halfWidths[index]!, y]);
      const outline = [...left, ...right.reverse()];
      return {
        id: mark.id, name: mark.group, type: "custom", coordinateSystem: "cartesian2d", clip: true,
        z: 1, data: [[mark.position, mark.grid[0]!]],
        renderItem: (_params: unknown, api: RenderApi) => ({
          type: "polygon",
          shape: { points: outline.map((point) => api.coord(point)) },
          style: { fill: mark.color, opacity: mark.opacity, stroke: mark.color, lineWidth: 1 },
        }),
      };
    }
  }
}

function axisOption(axis: PlanAxis, zoomed: boolean, nameGap: number, gridIndex: number, colors: PlanColors) {
  const names = new Map(axis.ticks.map((tick) => [tick.value, tick.label]));
  // Plan ticks hold for the full view. A zoomed value axis has to label what is on screen.
  const customValues = zoomed && axis.scale === "linear"
    ? undefined : axis.ticks.map((tick) => tick.value);
  return {
    type: "value", gridIndex, min: axis.domain[0], max: axis.domain[1], name: axis.title,
    nameLocation: "middle", nameGap, nameTextStyle: { color: colors.text },
    axisLine: { onZero: false, show: true, lineStyle: { color: colors.text } },
    axisTick: { show: true, customValues },
    axisLabel: {
      color: colors.text, customValues, showMinLabel: true, showMaxLabel: true,
      rotate: axis.scale === "category" && axis.ticks.length > LONG_AXIS_TICKS ? 30 : 0,
      formatter: (value: number) => names.get(value) ?? shown(value),
    },
    splitLine: { show: axis.grid, lineStyle: { color: colors.grid } },
  };
}

function panelGrid(panel: RenderPlanPanel, index: number, count: number) {
  const widest = Math.max(...panel.axes.y.ticks.map((tick) => tick.label.length));
  const left = widest * LABEL_PIXELS + 36;
  if (count === 1) return { left, right: 24, top: 64, bottom: 72 };
  const rows = Math.ceil(count / 2);
  return {
    left: `${5 + (index % 2) * 49}%`, top: `${16 + Math.floor(index / 2) * (76 / rows)}%`,
    width: "42%", height: `${62 / rows}%`, containLabel: true,
  };
}

export function buildPlanOption({ colors = DEFAULT_COLORS, plan, selected = [], text, zoomed = false }: {
  colors?: PlanColors; plan: RenderPlan; selected?: readonly string[]; text: PlanText; zoomed?: boolean;
}) {
  const { panels } = plan;
  const panel = panels[0]!;
  const grid = panels.map((item, index) => panelGrid(item, index, panels.length));
  const axisIndexes = (count: number) => Array.from({ length: count }, (_, index) => index);
  const seriesIn = panels.flatMap((item, index) => item.marks.map((mark) => ({ mark, index })));
  return {
    animation: false,
    backgroundColor: "#ffffff",
    textStyle: { color: colors.text },
    title: { text: panel.title, left: "center", textStyle: { color: colors.text, fontSize: 15 } },
    legend: {
      show: panels.some((item) => item.legend.length > 0), top: 28, right: 16, textStyle: { color: colors.text },
      data: panels.flatMap((item) => item.legend).map((entry) => ({
        name: entry.label, icon: entry.symbol === "line" ? "roundRect" : "circle",
      })),
    },
    grid,
    tooltip: {
      trigger: "item", confine: true,
      formatter: (params: { seriesIndex: number; dataIndex: number }) =>
        planTooltip(plan, params.seriesIndex, params.dataIndex, text),
    },
    xAxis: panels.map((item, index) => axisOption(item.axes.x, zoomed, 48, index, colors)),
    yAxis: panels.map((item, index) => {
      const left = (grid[index] as { left: number | string }).left;
      return axisOption(item.axes.y, zoomed, typeof left === "number" ? left - 20 : 44, index, colors);
    }),
    dataZoom: PLAN_ZOOM_IDS.map((id, axis) => ({
      id, type: "inside", filterMode: "none",
      [axis ? "yAxisIndex" : "xAxisIndex"]: axisIndexes(panels.length),
      // A plain wheel keeps scrolling the page.
      zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false,
    })),
    series: [
      ...seriesIn.map(({ mark, index }) => ({ ...markSeries(mark), xAxisIndex: index, yAxisIndex: index })),
      {
        id: "selection", type: "scatter", silent: true, symbol: "circle", symbolSize: 16, z: 6,
        data: selectedCoordinates(plan, selected),
        itemStyle: { color: "transparent", borderColor: colors.text, borderWidth: 2 },
      },
    ],
  };
}
