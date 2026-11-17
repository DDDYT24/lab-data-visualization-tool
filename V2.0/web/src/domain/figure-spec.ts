import { z } from "zod";
import generatedSpec from "../../../contracts/figure-spec-v2.schema.json";
import type { FigureBoardSpec } from "./research-figure-board";

type AxisControl = { scale: "linear" | "log10" | "symlog"; domain: [number, number] | null; ticks: number[] | null };
export type FigureAnnotation =
  | { kind: "text"; x: number; y: number; text: string }
  | { kind: "hline" | "vline"; value: number; label: string }
  | { kind: "rect"; x: [number, number]; y: [number, number]; label: string }
  | { kind: "bracket"; x: [number, number]; y: number; label: string };
export type FigureLayer = {
  kind: "chart" | "line" | "scatter" | "bar" | "histogram" | "box" | "violin" | "strip" | "paired"
    | "heatmap" | "contour" | "surface3d" | "fit" | "errorbar" | "band" | "reference";
  chartId: string | null;
  encoding: Record<"x" | "y" | "y2" | "z" | "group" | "error" | "weight" | "sampleId", string | null>;
  method: { id: string; params: Record<string, boolean | number | string | null>; seed: string | null } | null;
  style: {
    color: string | null; colorBy: string | null; marker: "o" | "s" | "^" | "D" | "v" | null; size: number | null;
    lineStyle: "solid" | "dashed" | "dotted" | null; width: number | null; opacity: number; zIndex: number;
  };
};
export type FigurePanel = {
  id: string; label: string; title: string; datasetRef: { datasetId: string; versionId: string } | null;
  axes: { x: AxisControl; y: AxisControl }; annotations: FigureAnnotation[]; layers: FigureLayer[];
};
export type FigureSpec = {
  schemaVersion: 2; title: string; fontSize: number; gridVisible: boolean;
  axisMode: FigureBoardSpec["axisMode"]; pointRendering: "auto" | "vector" | "raster";
  canvas: {
    sizePreset: "single_column" | "double_column" | "custom"; width: number | null; height: number | null;
    unit: "mm" | "in"; dpi: number; background: string; transparent: boolean; grayscale: boolean;
  };
  layout: { kind: "grid" | "mosaic"; rows: number; cols: number; mosaic: string[] | null; gutter: number; labelStyle: "upper" | "lower" | "none" };
  palette: { categorical: string; sequential: string; diverging: string };
  panels: FigurePanel[];
};

const increasing = (values: number[]) => values.every((value, index) => index === 0 || values[index - 1] < value);

// The rules the JSON schema cannot state; the API enforces the same ones.
function problems(figure: FigureSpec): string[] {
  const found: string[] = [];
  const labels = figure.panels.map((panel) => panel.label);
  if (new Set(labels).size !== labels.length || new Set(figure.panels.map((panel) => panel.id)).size !== labels.length)
    found.push("Panel ids and labels must be unique.");
  const { canvas, layout } = figure;
  const sized = canvas.width != null && canvas.height != null;
  if ((canvas.sizePreset === "custom") !== sized || (canvas.width == null) !== (canvas.height == null))
    found.push("Give width and height for a custom canvas, and only then.");
  if (layout.kind === "grid") {
    const cells = layout.rows * layout.cols;
    if (layout.mosaic != null || cells < labels.length || cells >= labels.length + layout.cols)
      found.push("The grid must hold every panel without an empty row.");
  } else {
    const rows = layout.mosaic ?? [];
    const placed = new Set(rows.join(""));
    if (rows.length !== layout.rows || rows.some((row) => row.length !== layout.cols))
      found.push("The mosaic must have rows x cols cells.");
    else if (placed.size !== labels.length || labels.some((label) => !placed.has(label)))
      found.push("The mosaic must place every panel and nothing else.");
    else for (const label of labels) {
      const cells = rows.flatMap((row, r) => [...row].flatMap((cell, c) => (cell === label ? [[r, c]] : [])));
      const span = (index: number) => Math.max(...cells.map((cell) => cell[index])) - Math.min(...cells.map((cell) => cell[index])) + 1;
      if (span(0) * span(1) !== cells.length) found.push("Each mosaic panel must cover one rectangle.");
    }
  }
  for (const panel of figure.panels) {
    for (const control of [panel.axes?.x, panel.axes?.y]) {
      if (!control) continue;
      if (control.domain && control.domain[0] >= control.domain[1]) found.push("An axis domain runs from low to high.");
      if (control.ticks && !increasing(control.ticks)) found.push("Axis ticks must increase.");
      if (control.scale === "log10" && [...(control.domain ?? []), ...(control.ticks ?? [])].some((value) => value <= 0))
        found.push("A log axis needs positive limits and ticks.");
    }
    for (const layer of panel.layers) {
      if ((layer.kind === "chart") !== (layer.chartId != null)) found.push("A chart layer names a saved chart; other layers do not.");
      if (layer.style?.color && layer.style.colorBy) found.push("Use a fixed colour or a colour field, not both.");
    }
  }
  return found;
}

const axis = (): AxisControl => ({ scale: "linear", domain: null, ticks: null });
const emptyEncoding = (): FigureLayer["encoding"] => ({ x: null, y: null, y2: null, z: null, group: null, error: null, weight: null, sampleId: null });
const plainStyle = (): FigureLayer["style"] => ({ color: null, colorBy: null, marker: null, size: null, lineStyle: null, width: null, opacity: 1, zIndex: 0 });
type Loose = Record<string, unknown>;

// Fills every default the API fills, in the API's key order, so both sides hold the same object.
export function completeFigure(raw: Loose): FigureSpec {
  const panels = (raw.panels as Loose[]).map((panel) => {
    const axes = (panel.axes ?? {}) as Loose;
    return {
      id: panel.id, label: panel.label, title: panel.title ?? "", datasetRef: panel.datasetRef ?? null,
      axes: { x: { ...axis(), ...(axes.x as object) }, y: { ...axis(), ...(axes.y as object) } },
      annotations: ((panel.annotations ?? []) as Loose[]).map((note) => (note.kind === "text" ? note : { ...note, label: note.label ?? "" })),
      layers: (panel.layers as Loose[]).map((layer) => ({
        kind: layer.kind, chartId: layer.chartId ?? null, encoding: { ...emptyEncoding(), ...(layer.encoding as object) },
        method: layer.method ? { id: (layer.method as Loose).id, params: (layer.method as Loose).params ?? {}, seed: (layer.method as Loose).seed ?? null } : null,
        style: { ...plainStyle(), ...(layer.style as object) },
      })),
    };
  });
  return {
    schemaVersion: 2, title: raw.title ?? "", fontSize: raw.fontSize ?? 10, gridVisible: raw.gridVisible ?? true,
    axisMode: raw.axisMode ?? "independent", pointRendering: raw.pointRendering ?? "auto",
    canvas: { sizePreset: "double_column", width: null, height: null, unit: "mm", dpi: 300, background: "#ffffff", transparent: false, grayscale: false, ...(raw.canvas as object) },
    layout: { kind: "grid", rows: 1, cols: 1, mosaic: null, gutter: 0.08, labelStyle: "upper", ...(raw.layout as object) },
    palette: { categorical: "blue_teal", sequential: "viridis", diverging: "blue_red", ...(raw.palette as object) },
    panels,
  } as FigureSpec;
}

const shape = z.fromJSONSchema(generatedSpec as Parameters<typeof z.fromJSONSchema>[0]);
export const figureSpecSchema = shape.transform((value, context) => {
  const figure = completeFigure(value as Loose);
  for (const message of problems(figure)) context.addIssue({ code: "custom", message });
  return figure;
});
export const parseFigureSpec = (value: unknown): FigureSpec => figureSpecSchema.parse(value);

const GRID: Record<number, [number, number]> = { 1: [1, 1], 2: [1, 2], 4: [2, 2] };
/** The v2 figure that draws exactly what a v1 board draws. */
export function upgradeBoard(board: FigureBoardSpec): FigureSpec {
  const [rows, cols] = GRID[board.panelChartIds.length];
  return completeFigure({
    schemaVersion: 2, title: board.title, fontSize: board.fontSize, gridVisible: board.gridVisible,
    axisMode: board.axisMode, pointRendering: board.pointRendering,
    layout: { rows, cols }, palette: { categorical: board.palette ?? "blue_teal" },
    panels: board.panelChartIds.map((chartId, index) => ({
      id: `p${index + 1}`, label: "ABCD"[index], title: board.panelTitles?.[index] ?? "",
      layers: [{ kind: "chart", chartId }],
    })),
  });
}

/** The v1 view of a figure, or null when the figure says more than v1 can. */
export function projectBoard(figure: FigureSpec): FigureBoardSpec | null {
  const charts = figure.panels.flatMap((panel) => (panel.layers.length === 1 && panel.layers[0].chartId ? [panel.layers[0].chartId] : []));
  const palette = figure.palette.categorical;
  if (charts.length !== figure.panels.length || !(charts.length in GRID) || (palette !== "blue_teal" && palette !== "grayscale")) return null;
  const titles = figure.panels.map((panel) => panel.title);
  const board: FigureBoardSpec = {
    schemaVersion: 1, pointRendering: figure.pointRendering, title: figure.title, axisMode: figure.axisMode, palette,
    fontSize: figure.fontSize, gridVisible: figure.gridVisible, panelTitles: titles.some(Boolean) ? titles : [], panelChartIds: charts,
  };
  // Both sides are completed, so a figure built in another key order still compares equal.
  return JSON.stringify(upgradeBoard(board)) === JSON.stringify(completeFigure(figure)) ? board : null;
}
