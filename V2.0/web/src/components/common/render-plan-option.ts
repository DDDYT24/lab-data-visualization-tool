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
export type PlanText = { mean: string; median: string };
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
  }
}

export function planTooltip(plan: RenderPlan, seriesIndex: number, index: number, text: PlanText) {
  const mark = plan.panels[0]!.marks[seriesIndex];
  return mark ? tooltipLines(mark, index, text).map(escapeHtml).join("<br/>") : "";
}

/** The observations one drawn item stands for; violins and summaries have no single row. */
export function planSelection(plan: RenderPlan, seriesIndex: number, index: number): string[] {
  const mark = plan.panels[0]!.marks[seriesIndex];
  if (mark?.kind === "points") return [mark.sourceRefs[index]!];
  if (mark?.kind === "segments") return [...mark.sourceRefs[index]!];
  return [];
}

function selectedCoordinates(panel: RenderPlanPanel, selected: readonly string[]) {
  const wanted = new Set(selected);
  const found: Coordinate[] = [];
  for (const mark of panel.marks) {
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

function markSeries(mark: RenderPlanMark) {
  switch (mark.kind) {
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
        renderItem: (_params: unknown, api: { coord: (point: number[]) => number[] }) => ({
          type: "polygon",
          shape: { points: outline.map((point) => api.coord(point)) },
          style: { fill: mark.color, opacity: mark.opacity, stroke: mark.color, lineWidth: 1 },
        }),
      };
    }
  }
}

function axisOption(axis: PlanAxis, zoomed: boolean, nameGap: number) {
  const names = new Map(axis.ticks.map((tick) => [tick.value, tick.label]));
  // Plan ticks hold for the full view. A zoomed value axis has to label what is on screen.
  const customValues = zoomed && axis.scale === "linear"
    ? undefined : axis.ticks.map((tick) => tick.value);
  return {
    type: "value", min: axis.domain[0], max: axis.domain[1], name: axis.title,
    nameLocation: "middle", nameGap, nameTextStyle: { color: TEXT_COLOR },
    axisLine: { onZero: false, show: true, lineStyle: { color: TEXT_COLOR } },
    axisTick: { show: true, customValues },
    axisLabel: {
      color: TEXT_COLOR, customValues, showMinLabel: true, showMaxLabel: true,
      rotate: axis.scale === "category" && axis.ticks.length > LONG_AXIS_TICKS ? 30 : 0,
      formatter: (value: number) => names.get(value) ?? shown(value),
    },
    splitLine: { show: axis.scale === "linear", lineStyle: { color: gridColor } },
  };
}

export function buildPlanOption({ plan, selected = [], text, zoomed = false }: {
  plan: RenderPlan; selected?: readonly string[]; text: PlanText; zoomed?: boolean;
}) {
  const panel = plan.panels[0]!;
  const { x, y } = panel.axes;
  const widest = Math.max(...y.ticks.map((tick) => tick.label.length));
  const left = widest * LABEL_PIXELS + 36;
  return {
    animation: false,
    backgroundColor: "#ffffff",
    textStyle: { color: TEXT_COLOR },
    title: { text: panel.title, left: "center", textStyle: { color: TEXT_COLOR, fontSize: 15 } },
    legend: {
      show: panel.legend.length > 0, top: 28, right: 16, textStyle: { color: TEXT_COLOR },
      data: panel.legend.map((entry) => ({
        name: entry.label, icon: entry.symbol === "line" ? "roundRect" : "circle",
      })),
    },
    grid: { left, right: 24, top: 64, bottom: 72 },
    tooltip: {
      trigger: "item", confine: true,
      formatter: (params: { seriesIndex: number; dataIndex: number }) =>
        planTooltip(plan, params.seriesIndex, params.dataIndex, text),
    },
    xAxis: axisOption(x, zoomed, 48),
    yAxis: axisOption(y, zoomed, left - 20),
    dataZoom: PLAN_ZOOM_IDS.map((id, axis) => ({
      id, type: "inside", filterMode: "none", [axis ? "yAxisIndex" : "xAxisIndex"]: 0,
      // A plain wheel keeps scrolling the page.
      zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false,
    })),
    series: [
      ...panel.marks.map(markSeries),
      {
        id: "selection", type: "scatter", silent: true, symbol: "circle", symbolSize: 16, z: 6,
        data: selectedCoordinates(panel, selected),
        itemStyle: { color: "transparent", borderColor: TEXT_COLOR, borderWidth: 2 },
      },
    ],
  };
}
