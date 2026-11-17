import { chartLineType, geometryContract, gridColor } from "@/domain/chart-render-contract";
import {
  axisPosition, axisValue, type RenderPlan, type RenderPlanMark, type RenderPlanNote, type RenderPlanPanel,
} from "@/domain/render-plan";

// The adapter only draws what the plan states: it computes no statistics, axes or geometry.
const TEXT_COLOR = "#20252d";
const POINT_PIXELS_PER_SIZE = 2;
const LABEL_PIXELS = 8;
const LONG_AXIS_TICKS = 4;
// Two dashes; ECharts has no built-in dashed legend icon.
const DASHED_LEGEND_ICON = "path://M0 0h6v2H0zM10 0h6v2h-6z";
const LEGEND_ICONS = { line: "roundRect", marker: "circle", dashed: DASHED_LEGEND_ICON };
const ERRORBAR_CAP_PIXELS = 1.5;
// Annotation sizes are in points, as the export draws them.
const NOTE = geometryContract.annotation;
const PIXELS_PER_POINT = 4 / 3;
const NOTE_GAP = NOTE.labelGap * PIXELS_PER_POINT;

const SYMBOLS = {
  o: { symbol: "circle", symbolRotate: 0 },
  s: { symbol: "rect", symbolRotate: 0 },
  "^": { symbol: "triangle", symbolRotate: 0 },
  D: { symbol: "diamond", symbolRotate: 0 },
  v: { symbol: "triangle", symbolRotate: 180 },
} as const;

type PlanAxis = RenderPlanPanel["axes"]["x"];
type MeshMark = Extract<RenderPlanMark, { kind: "mesh" }>;
type Coordinate = [number, number];
type RenderApi = {
  coord: (point: number[]) => number[];
  value: (dimension: number) => number | string;
  visual: (key: "color") => string;
};
// Heatmap column names run at this angle, as the export draws them.
const CELL_LABEL_ROTATION = 45;
export type PlanText = { mean: string; median: string };
export const planText = (zh: boolean): PlanText =>
  zh ? { mean: "均值", median: "中位数" } : { mean: "Mean", median: "Median" };
export type PlanColors = { text: string; grid: string };
const DEFAULT_COLORS: PlanColors = { text: TEXT_COLOR, grid: gridColor };
export const PLAN_ZOOM_IDS = ["zoom-x", "zoom-y"] as const;

const escapeHtml = (value: string) => value.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const shown = (value: number) =>
  Number(value.toPrecision(6)).toString().replace(/-/g, "−");

/** Each cell that has a value, as [x, y, value]. */
const meshCells = (mark: MeshMark) =>
  mark.values.flatMap((row, j) =>
    row.flatMap((value, i) => (value === null ? [] : [[mark.x[i]!, mark.y[j]!, value]])));

function tooltipLines(mark: RenderPlanMark, index: number, text: PlanText) {
  switch (mark.kind) {
    case "mesh":
      return [mark.colorTitle || mark.group, shown(meshCells(mark)[index]![2]!)];
    case "points":
    case "polyline":
      return [mark.group, shown(mark.y[index]!)];
    case "band":
    case "errorbar":
      return [mark.group];
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

/** The shared tooltip of line and scatter charts: one x and each series' value there. */
export function planAxisTooltip(plan: RenderPlan) {
  const names = new Map(plan.panels.flatMap((panel) => panel.axes.x.ticks).map((tick) => [tick.value, tick.label]));
  return (params: { seriesName: string; value: [number, number] }[]) => {
    const [first] = params;
    if (!first) return "";
    const x = first.value[0];
    return [names.get(x) ?? shown(x), ...params.map((item) => `${item.seriesName}: ${shown(item.value[1])}`)]
      .map(escapeHtml).join("<br/>");
  };
}

/** The observations one drawn item stands for; violins and summaries have no single row. */
export function planSelection(plan: RenderPlan, seriesIndex: number, index: number): string[] {
  const mark = planMarks(plan)[seriesIndex];
  if (mark?.kind === "points") return mark.sourceRefs.length ? [mark.sourceRefs[index]!] : [];
  if (mark?.kind === "segments") return [...mark.sourceRefs[index]!];
  return [];
}

type Place = (point: number[]) => number[];
const scaled = (axis: PlanAxis) => axis.scale === "log10" || axis.scale === "symlog";

// A log or symlog panel is drawn on linear ECharts axes: every coordinate goes through the plan's scale.
function placement(panel: RenderPlanPanel, mark?: RenderPlanMark): Place | null {
  const { x } = panel.axes;
  const y = mark && "axis" in mark && mark.axis === "y2" && panel.axes.y2 ? panel.axes.y2 : panel.axes.y;
  if (!scaled(x) && !scaled(y)) return null;
  return (point) => [axisPosition(x.scale, point[0]!), axisPosition(y.scale, point[1]!), ...point.slice(2)];
}

type PlacedSeries = {
  data: unknown[];
  renderItem?: (params: never, api: RenderApi) => unknown;
};
function placed<T extends PlacedSeries>(series: T, at: Place | null): T {
  if (!at) return series;
  const draw = series.renderItem;
  return {
    ...series,
    data: series.data.map((item) => Array.isArray(item)
      ? at(item as number[])
      : { ...(item as object), coords: (item as { coords: number[][] }).coords.map(at) }),
    ...(draw ? {
      renderItem: (params: never, api: RenderApi) => draw(params, {
        coord: (point) => api.coord(at(point)), value: (dimension) => api.value(dimension), visual: (key) => api.visual(key),
      }),
    } : {}),
  };
}

function selectedCoordinates(plan: RenderPlan, selected: readonly string[]) {
  const wanted = new Set(selected);
  const found: number[][] = [];
  for (const panel of plan.panels) for (const mark of panel.marks) {
    if (mark.kind !== "points") continue;
    const at = placement(panel, mark) ?? ((point: number[]) => point);
    mark.sourceRefs.forEach((id, index) => {
      if (wanted.has(id)) found.push(at([mark.x[index]!, mark.y[index]!]));
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

function errorbarItem(mark: Extract<RenderPlanMark, { kind: "errorbar" }>, index: number, api: RenderApi) {
  const x = mark.x[index]!;
  const [left, top] = api.coord([x, mark.y[index]! + mark.error[index]!]) as [number, number];
  const [, bottom] = api.coord([x, mark.y[index]! - mark.error[index]!]) as [number, number];
  const cap = mark.capSize * ERRORBAR_CAP_PIXELS;
  const style = { stroke: mark.color, lineWidth: mark.width, opacity: mark.opacity };
  const line = (x1: number, y1: number, x2: number, y2: number) => ({ type: "line", shape: { x1, y1, x2, y2 }, style });
  return {
    type: "group",
    children: [
      line(left, top, left, bottom),
      line(left - cap, top, left + cap, top),
      line(left - cap, bottom, left + cap, bottom),
    ],
  };
}

function markSeries(mark: RenderPlanMark) {
  switch (mark.kind) {
    case "mesh":
      // Colour comes from the panel's visual map, which maps the third value onto the palette.
      return {
        id: mark.id, name: mark.colorTitle || mark.group, type: "custom", coordinateSystem: "cartesian2d",
        clip: true, z: 1, data: meshCells(mark), encode: { x: 0, y: 1, tooltip: 2 },
        renderItem: (_params: unknown, api: RenderApi) => {
          const x = Number(api.value(0));
          const y = Number(api.value(1));
          return rectItem(api.coord([x - 0.5, y - 0.5]), api.coord([x + 0.5, y + 0.5]), api.visual("color"), 1);
        },
      };
    case "polyline":
      return {
        id: mark.id, name: mark.group, type: "line", z: 3, animation: false,
        showSymbol: mark.markerSize > 0, symbolSize: mark.markerSize * POINT_PIXELS_PER_SIZE,
        data: mark.x.map((x, index) => [x, mark.y[index]!]),
        itemStyle: { color: mark.color },
        lineStyle: { color: mark.color, width: mark.width, type: chartLineType(mark.lineStyle) },
        silent: mark.markerSize === 0,
        ...(mark.markerSize === 0 ? { tooltip: { show: false } } : {}),
      };
    case "band": {
      const upper = mark.x.map((x, index) => [x, mark.upper[index]!]);
      const lower = mark.x.map((x, index) => [x, mark.lower[index]!]).reverse();
      const outline = [...upper, ...lower];
      return {
        id: mark.id, name: `${mark.group} band`, type: "custom", coordinateSystem: "cartesian2d", clip: true,
        z: 1, silent: true, tooltip: { show: false }, data: [[mark.x[0]!, mark.lower[0]!]],
        renderItem: (_params: unknown, api: RenderApi) => ({
          type: "polygon",
          shape: { points: outline.map((point) => api.coord(point)) },
          style: { fill: mark.color, opacity: mark.opacity },
        }),
      };
    }
    case "errorbar":
      return {
        id: mark.id, name: `${mark.group} error`, type: "custom", coordinateSystem: "cartesian2d", clip: true,
        z: 3, silent: true, tooltip: { show: false }, data: mark.x.map((x, index) => [x, mark.y[index]!]),
        renderItem: (params: { dataIndex: number }, api: RenderApi) => errorbarItem(mark, params.dataIndex, api),
      };
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

type Frame = { x: number; y: number; width: number; height: number };

/** The first point each note places, so a note is one item of its series. */
function noteAnchor(note: RenderPlanNote, panel: RenderPlanPanel): number[] {
  switch (note.kind) {
    case "text": return [note.x, note.y];
    case "hline": return [panel.axes.x.domain[0], note.value];
    case "vline": return [note.value, panel.axes.y.domain[0]];
    case "rect": return [note.x[0], note.y[0]];
    case "bracket": return [note.x[0], note.y];
  }
}

function noteLabel(text: string, at: number[], dx: number, dy: number, align: string, verticalAlign: string, color: string) {
  return text ? [{
    type: "text",
    style: { text, x: at[0]! + dx, y: at[1]! + dy, fill: color, fontSize: NOTE.fontSize * PIXELS_PER_POINT, align, verticalAlign },
  }] : [];
}

/** A note's lines and label, each placed where the plan says. */
function noteItem(note: RenderPlanNote, anchor: number[], api: RenderApi, frame: Frame, color: string) {
  const stroke = { stroke: color, lineWidth: NOTE.lineWidth, fill: "none" };
  const rule = { ...stroke, lineDash: chartLineType(NOTE.ruleStyle as Parameters<typeof chartLineType>[0]) };
  const at = api.coord(anchor);
  switch (note.kind) {
    case "text":
      return { type: "group", children: noteLabel(note.text, at, 0, 0, "center", "middle", color) };
    case "hline":
      return { type: "group", children: [
        { type: "line", shape: { x1: frame.x, y1: at[1], x2: frame.x + frame.width, y2: at[1] }, style: rule },
        ...noteLabel(note.label, [frame.x + frame.width, at[1]!], -NOTE_GAP, -NOTE_GAP, "right", "bottom", color),
      ] };
    case "vline":
      return { type: "group", children: [
        { type: "line", shape: { x1: at[0], y1: frame.y, x2: at[0], y2: frame.y + frame.height }, style: rule },
        ...noteLabel(note.label, [at[0]!, frame.y], NOTE_GAP, NOTE_GAP, "left", "top", color),
      ] };
    case "rect":
      return { type: "group", children: noteLabel(note.label, api.coord(note.labelAt), NOTE_GAP, NOTE_GAP, "left", "top", color) };
    case "bracket":
      return { type: "group", children: [
        {
          type: "polyline", style: stroke,
          shape: { points: [[note.x[0], note.foot], [note.x[0], note.y], [note.x[1], note.y], [note.x[1], note.foot]].map((point) => api.coord(point)) },
        },
        ...noteLabel(note.label, api.coord(note.labelAt), 0, -NOTE_GAP, "center", "bottom", color),
      ] };
  }
}

/** A panel's annotations: shaded ranges under the marks, everything else over them. */
function noteSeries(panel: RenderPlanPanel, color: string) {
  const notes = panel.notes ?? [];
  const rects = notes.filter((note): note is Extract<RenderPlanNote, { kind: "rect" }> => note.kind === "rect");
  const base = { type: "custom", coordinateSystem: "cartesian2d", clip: true, silent: true, tooltip: { show: false } };
  return [
    ...(rects.length ? [{
      ...base, id: `notes-under-${panel.id}`, z: 0, data: rects.map((note) => [note.x[0], note.y[0]]),
      renderItem: (params: { dataIndex: number }, api: RenderApi) => {
        const note = rects[params.dataIndex]!;
        return rectItem(api.coord([note.x[0], note.y[0]]), api.coord([note.x[1], note.y[1]]), color, NOTE.rectOpacity);
      },
    }] : []),
    ...(notes.length ? [{
      ...base, id: `notes-${panel.id}`, z: 7, data: notes.map((note) => noteAnchor(note, panel)),
      renderItem: (params: { dataIndex: number; coordSys: Frame }, api: RenderApi) => {
        const note = notes[params.dataIndex]!;
        return noteItem(note, noteAnchor(note, panel), api, params.coordSys, color);
      },
    }] : []),
  ];
}

function axisOption(
  axis: PlanAxis, zoomed: boolean, nameGap: number, gridIndex: number, colors: PlanColors, rotate?: number,
) {
  const at = (value: number) => axisPosition(axis.scale, value);
  const names = new Map(axis.ticks.map((tick) => [at(tick.value), tick.label]));
  // Plan ticks hold for the full view. A zoomed linear axis has to label what is on screen.
  const customValues = zoomed && axis.scale === "linear"
    ? undefined : axis.ticks.map((tick) => at(tick.value));
  const [start, end] = axis.domain.map(at) as [number, number];
  return {
    type: "value", gridIndex, min: Math.min(start, end), max: Math.max(start, end), inverse: start > end,
    name: axis.title,
    nameLocation: "middle", nameGap, nameTextStyle: { color: colors.text },
    axisLine: { onZero: false, show: true, lineStyle: { color: colors.text } },
    axisTick: { show: true, customValues },
    axisLabel: {
      color: colors.text, customValues, showMinLabel: true, showMaxLabel: true,
      rotate: rotate ?? (axis.scale === "category" && axis.ticks.length > LONG_AXIS_TICKS ? 30 : 0),
      formatter: (value: number) => names.get(value) ?? shown(axisValue(axis.scale, value)),
    },
    splitLine: { show: axis.grid, lineStyle: { color: colors.grid } },
  };
}

const cellMesh = (panel: RenderPlanPanel) =>
  panel.marks.find((mark): mark is MeshMark => mark.kind === "mesh" && mark.shape === "cells");

function panelGrid(panel: RenderPlanPanel, index: number, count: number) {
  const widest = Math.max(...panel.axes.y.ticks.map((tick) => tick.label.length));
  const left = widest * LABEL_PIXELS + 36;
  // Room for the colour scale beside a heatmap.
  if (count === 1) return { left, right: cellMesh(panel) ? 96 : 24, top: 64, bottom: 72 };
  const rows = Math.ceil(count / 2);
  return {
    left: `${5 + (index % 2) * 49}%`, top: `${16 + Math.floor(index / 2) * (76 / rows)}%`,
    width: "42%", height: `${62 / rows}%`, containLabel: true,
  };
}

const strided = (length: number, stride: number) =>
  Array.from({ length }, (_, index) => index).filter((index) => index % stride === 0 || index === length - 1);

export type SurfaceView = { alpha: number; beta: number; distance: number };

/**
 * A 3D surface plan. Camera, axis and legend toggles and the low-cost point limit are browser
 * state: the limit keeps every n-th grid line, and the plan's colour domain still spans every value.
 */
export function buildSurfacePlanOption({
  colors = DEFAULT_COLORS, plan, pointLimit, showAxes = true, showLegend = true, view,
}: {
  colors?: PlanColors & { axis?: string }; plan: RenderPlan; pointLimit?: number; showAxes?: boolean;
  showLegend?: boolean; view: SurfaceView;
}) {
  const panel = plan.panels[0]!;
  const mark = panel.marks.find((item): item is MeshMark => item.kind === "mesh")!;
  const limit = Math.max(4, pointLimit ?? mark.x.length * mark.y.length);
  let stride = 1;
  while (strided(mark.x.length, stride).length * strided(mark.y.length, stride).length > limit) stride += 1;
  const columns = strided(mark.x.length, stride);
  const rows = strided(mark.y.length, stride);
  const axisColor = colors.axis ?? colors.text;
  const axis3D = (axis: PlanAxis) => {
    // 3D axes take no tick list. They tick at round values as the plan does, and also at the
    // domain ends, which are not ticks and stay unlabelled.
    const span = Math.abs(axis.domain[1] - axis.domain[0]);
    const near = (a: number, b: number) => Math.abs(a - b) <= span * 1e-9;
    const label = (value: number) =>
      axis.ticks.find((tick) => near(tick.value, value))?.label
      ?? (axis.domain.some((end) => near(end, value)) ? "" : shown(value));
    return {
      type: "value", min: axis.domain[0], max: axis.domain[1], name: showAxes ? axis.title : "",
      axisLine: { show: showAxes }, nameTextStyle: { color: colors.text },
      axisLabel: { show: showAxes, color: axisColor, formatter: label },
      splitLine: { show: axis.grid, lineStyle: { color: colors.grid } },
    };
  };
  const [low, high] = mark.colorDomain;
  return {
    grid3D: {
      top: 45, height: "70%", boxDepth: 80, boxHeight: 80, boxWidth: 120,
      axisLabel: { show: showAxes, color: axisColor }, axisPointer: { show: showAxes },
      axisLine: { show: showAxes, lineStyle: { color: axisColor } },
      splitLine: { show: panel.axes.x.grid, lineStyle: { color: colors.grid } },
      viewControl: {
        alpha: view.alpha, autoRotate: false, beta: view.beta, distance: view.distance, panSensitivity: 1,
        projection: "perspective", rotateSensitivity: 1, zoomSensitivity: 1,
      },
    },
    visualMap: {
      show: showLegend, dimension: 2, calculable: false, orient: "horizontal", left: "center", bottom: 0,
      inRange: { color: mark.palette }, min: low, max: high, text: [shown(high), shown(low)],
      textStyle: { color: colors.text },
    },
    xAxis3D: axis3D(panel.axes.x),
    yAxis3D: axis3D(panel.axes.y),
    zAxis3D: axis3D(panel.axes.z!),
    series: [{
      type: "surface", shading: "color", dataShape: [rows.length, columns.length],
      data: rows.flatMap((j) => columns.map((i) => [mark.x[i]!, mark.y[j]!, mark.values[j]![i] ?? null])),
    }],
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
  // Each panel's y axes, in the order yAxis lists them below.
  const yAxisStart = panels.map((_, index) => panels.slice(0, index).reduce((sum, item) => sum + (item.axes.y2 ? 2 : 1), 0));
  const yAxisIndex = (mark: RenderPlanMark, index: number) =>
    yAxisStart[index]! + ("axis" in mark && mark.axis === "y2" ? 1 : 0);
  return {
    animation: false,
    backgroundColor: "#ffffff",
    textStyle: { color: colors.text },
    title: { text: panel.title, left: "center", textStyle: { color: colors.text, fontSize: 15 } },
    legend: {
      show: panels.some((item) => item.legend.length > 0), top: 28, right: 16, textStyle: { color: colors.text },
      data: panels.flatMap((item) => item.legend).map((entry) => ({ name: entry.label, icon: LEGEND_ICONS[entry.symbol] })),
    },
    grid,
    tooltip: {
      trigger: "item", confine: true,
      formatter: (params: { seriesIndex: number; dataIndex: number }) =>
        planTooltip(plan, params.seriesIndex, params.dataIndex, text),
    },
    xAxis: panels.map((item, index) =>
      axisOption(item.axes.x, zoomed, 48, index, colors, cellMesh(item) ? CELL_LABEL_ROTATION : undefined)),
    yAxis: panels.flatMap((item, index) => {
      const left = (grid[index] as { left: number | string }).left;
      const primary = axisOption(item.axes.y, zoomed, typeof left === "number" ? left - 20 : 44, index, colors);
      return item.axes.y2
        ? [primary, { ...axisOption(item.axes.y2, zoomed, 44, index, colors), position: "right" }]
        : [primary];
    }),
    dataZoom: PLAN_ZOOM_IDS.map((id, axis) => ({
      id, type: "inside", filterMode: "none",
      [axis ? "yAxisIndex" : "xAxisIndex"]: axisIndexes(axis ? yAxisStart.at(-1)! + (panels.at(-1)!.axes.y2 ? 2 : 1) : panels.length),
      // A plain wheel keeps scrolling the page.
      zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false,
    })),
    visualMap: seriesIn.flatMap(({ mark }, seriesIndex) => mark.kind === "mesh" ? [{
      type: "continuous", seriesIndex, dimension: 2, calculable: false, orient: "vertical", right: 8, top: "middle",
      min: mark.colorDomain[0], max: mark.colorDomain[1], inRange: { color: mark.palette },
      text: [shown(mark.colorDomain[1]), shown(mark.colorDomain[0])], textStyle: { color: colors.text },
    }] : []),
    series: [
      ...seriesIn.map(({ mark, index }) => ({
        ...placed(markSeries(mark) as PlacedSeries, placement(panels[index]!, mark)), xAxisIndex: index, yAxisIndex: yAxisIndex(mark, index),
      })),
      // After every mark, so a mark's series index is still its place in the plan.
      ...panels.flatMap((item, index) => noteSeries(item, colors.text).map((series) => ({
        ...placed(series as PlacedSeries, placement(item)), xAxisIndex: index, yAxisIndex: yAxisStart[index]!,
      }))),
      {
        id: "selection", type: "scatter", silent: true, symbol: "circle", symbolSize: 16, z: 6,
        data: selectedCoordinates(plan, selected),
        itemStyle: { color: "transparent", borderColor: colors.text, borderWidth: 2 },
      },
    ],
  };
}
