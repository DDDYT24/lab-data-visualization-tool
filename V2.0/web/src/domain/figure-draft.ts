import type { RenderPlan } from "./render-plan";
import { completeFigure, projectBoard, upgradeBoard, type FigureAnnotation, type FigureFont, type FigurePanel, type FigureSpec } from "./figure-spec";
import type { FigureBoardSpec } from "./research-figure-board";
import geometry from "../../../contracts/chart-render-v2.json";

export const MAX_PANELS = 6;
export const PANEL_LABELS = "ABCDEF";
export const MAX_LAYERS = geometry.layers.max;

export type FigureLayout = Pick<FigureSpec["layout"], "kind" | "rows" | "cols" | "mosaic">;
export type AxisDraft = FigurePanel["axes"]["x"];
/**
 * `chartId` is the panel's own chart: it names the panel and is listed first in the saved spec.
 * `layers` holds every chart the panel draws, bottom first, once it draws more than its own; it is
 * empty otherwise.
 */
export type PanelDraft = { chartId: string; title: string; axes: { x: AxisDraft; y: AxisDraft }; notes: FigureAnnotation[]; layers: string[] };
/** What the figure editor holds: always six panel slots, of which the first `count` are used. */
export type FigureDraft = {
  title: string; axisMode: FigureBoardSpec["axisMode"]; palette: "blue_teal" | "grayscale"; fontSize: number;
  grid: boolean; pointRendering: "auto" | "vector" | "raster"; labelStyle: FigureSpec["layout"]["labelStyle"];
  fontFamily: FigureFont; categoryCounts: boolean; provenanceOnFigure: boolean;
  count: number; layout: FigureLayout; panels: PanelDraft[];
  /** Settings the editor does not show (an API client may set them); a re-save keeps them. */
  kept: { canvas?: FigureSpec["canvas"]; gutter?: number; sequential?: string; diverging?: string };
};
const DRAFT_KEYS = [
  "title", "axisMode", "palette", "fontSize", "grid", "pointRendering", "labelStyle", "fontFamily", "categoryCounts",
  "provenanceOnFigure", "count", "layout", "panels", "kept",
] as const;

const grid = (rows: number, cols: number): FigureLayout => ({ kind: "grid", rows, cols, mosaic: null });
const mosaic = (...rows: string[]): FigureLayout => ({ kind: "mosaic", rows: rows.length, cols: rows[0]!.length, mosaic: rows });
/** The arrangements offered per panel count; the first is the default, and for 1, 2 and 4 it is the v1 grid. */
export const ARRANGEMENTS: Record<number, FigureLayout[]> = {
  1: [grid(1, 1)],
  2: [grid(1, 2), grid(2, 1)],
  3: [grid(1, 3), grid(3, 1), mosaic("AB", "AC"), mosaic("AA", "BC")],
  4: [grid(2, 2), grid(1, 4), mosaic("AAA", "BCD")],
  5: [grid(2, 3), mosaic("AAB", "CDE"), mosaic("AB", "CD", "EE")],
  6: [grid(2, 3), grid(3, 2), mosaic("AAB", "CDE", "CDF")],
};
export const sameLayout = (a: FigureLayout, b: FigureLayout) => JSON.stringify(a) === JSON.stringify(b);

/** The cells a panel covers, as [row, col, rowSpan, colSpan], the way the API lays them out. */
export function layoutCells(layout: FigureLayout, count: number): [number, number, number, number][] {
  if (layout.kind === "grid") return Array.from({ length: count }, (_, i) => [Math.floor(i / layout.cols), i % layout.cols, 1, 1]);
  return [...PANEL_LABELS.slice(0, count)].map((label) => {
    const cells = layout.mosaic!.flatMap((row, r) => [...row].flatMap((cell, c) => (cell === label ? [[r, c]] : [])));
    const rows = cells.map((cell) => cell[0]!), cols = cells.map((cell) => cell[1]!);
    return [Math.min(...rows), Math.min(...cols), Math.max(...rows) - Math.min(...rows) + 1, Math.max(...cols) - Math.min(...cols) + 1];
  });
}

const linear = (): AxisDraft => ({ scale: "linear", domain: null, ticks: null });
export const shares = (mode: FigureDraft["axisMode"], dimension: "x" | "y") => mode === "shared_xy" || mode === `shared_${dimension}`;
const emptyPanel = (): PanelDraft => ({ chartId: "", title: "", axes: { x: linear(), y: linear() }, notes: [], layers: [] });
const pad = (panels: PanelDraft[]) => [...panels, ...Array.from({ length: MAX_PANELS - panels.length }, emptyPanel)];

export function emptyDraft(): FigureDraft {
  return {
    title: "", axisMode: "independent", palette: "blue_teal", fontSize: 10, grid: true, pointRendering: "auto",
    labelStyle: "upper", fontFamily: "DejaVu Sans", categoryCounts: true, provenanceOnFigure: false,
    count: 1, layout: ARRANGEMENTS[1]![0]!, panels: pad([]), kept: {},
  };
}

/** A new count keeps the arrangement when it still fits, otherwise takes the count's default. */
export function withCount(draft: FigureDraft, count: number): FigureDraft {
  const fits = ARRANGEMENTS[count]!.some((layout) => sameLayout(layout, draft.layout));
  return { ...draft, count, layout: fits ? draft.layout : ARRANGEMENTS[count]![0]! };
}

export function figureOf(spec: FigureBoardSpec | FigureSpec): FigureSpec {
  return spec.schemaVersion === 1 ? upgradeBoard(spec) : spec;
}

export function chartIdsOf(spec: FigureBoardSpec | FigureSpec): string[] {
  return figureOf(spec).panels.map((panel) => panel.layers[0]?.chartId ?? "");
}

export function draftFromSpec(spec: FigureBoardSpec | FigureSpec): FigureDraft {
  const figure = figureOf(spec);
  const palette = figure.palette.categorical === "grayscale" ? "grayscale" : "blue_teal";
  // The editor names panels A–F by position; a figure saved with other letters is renamed,
  // its mosaic with it, so it can be saved again.
  const rename = new Map(figure.panels.map((panel, index) => [panel.label, PANEL_LABELS[index]!]));
  const mosaic = figure.layout.mosaic?.map((row) => [...row].map((cell) => rename.get(cell) ?? cell).join("")) ?? null;
  return {
    title: figure.title, axisMode: figure.axisMode, palette, fontSize: figure.fontSize, grid: figure.gridVisible,
    pointRendering: figure.pointRendering, labelStyle: figure.layout.labelStyle, fontFamily: figure.fontFamily,
    categoryCounts: figure.categoryCounts, provenanceOnFigure: figure.provenanceOnFigure, count: figure.panels.length,
    layout: { kind: figure.layout.kind, rows: figure.layout.rows, cols: figure.layout.cols, mosaic },
    panels: pad(figure.panels.map((panel) => {
      // Drawn by zIndex, then by place in the list, as the API orders them.
      const drawn = panel.layers.map((layer, index) => ({ layer, index }))
        .sort((a, b) => a.layer.style.zIndex - b.layer.style.zIndex || a.index - b.index)
        .map(({ layer }) => layer.chartId ?? "");
      return {
        chartId: panel.layers[0]?.chartId ?? "", title: panel.title, axes: panel.axes, notes: panel.annotations,
        layers: drawn.length > 1 ? drawn : [],
      };
    })),
    kept: { canvas: figure.canvas, gutter: figure.layout.gutter, sequential: figure.palette.sequential, diverging: figure.palette.diverging },
  };
}

/** The figure to save: v1 when v1 says the same thing, so templates and older readers keep working. */
export function specFromDraft(draft: FigureDraft): FigureBoardSpec | FigureSpec {
  const figure = completeFigure({
    schemaVersion: 2, title: draft.title, fontSize: draft.fontSize, gridVisible: draft.grid,
    axisMode: draft.axisMode, pointRendering: draft.pointRendering, fontFamily: draft.fontFamily,
    categoryCounts: draft.categoryCounts, provenanceOnFigure: draft.provenanceOnFigure,
    canvas: draft.kept.canvas,
    layout: { ...draft.layout, labelStyle: draft.labelStyle, ...(draft.kept.gutter === undefined ? {} : { gutter: draft.kept.gutter }) },
    palette: {
      categorical: draft.palette,
      ...(draft.kept.sequential ? { sequential: draft.kept.sequential } : {}),
      ...(draft.kept.diverging ? { diverging: draft.kept.diverging } : {}),
    },
    panels: draft.panels.slice(0, draft.count).map((panel, index) => ({
      id: `p${index + 1}`, label: PANEL_LABELS[index], title: panel.title,
      // A shared axis is framed across the panels, so a panel's own control on it is not sent;
      // the draft keeps it for when the axis is no longer shared.
      axes: {
        x: shares(draft.axisMode, "x") ? linear() : panel.axes.x,
        y: shares(draft.axisMode, "y") ? linear() : panel.axes.y,
      },
      annotations: panel.notes, layers: savedLayers(panel),
    })),
  });
  return projectBoard(figure) ?? figure;
}

/** The panel's own chart first, as it names the panel; each zIndex is its place in the drawing order. */
function savedLayers(panel: PanelDraft) {
  if (!panel.layers.length) return [{ kind: "chart", chartId: panel.chartId }];
  const others = panel.layers.filter((id) => id !== panel.chartId);
  return [panel.chartId, ...others].map((chartId) => ({ kind: "chart", chartId, style: { zIndex: panel.layers.indexOf(chartId) } }));
}

/** The charts a panel draws, bottom first. */
export const panelLayers = (panel: PanelDraft): string[] => panel.layers.length ? panel.layers : panel.chartId ? [panel.chartId] : [];
const stacked = (panel: PanelDraft, layers: string[]): PanelDraft => ({ ...panel, layers: layers.length > 1 ? layers : [] });

/** A new layer goes on top. */
export function addLayer(panel: PanelDraft, chartId: string): PanelDraft {
  const layers = panelLayers(panel);
  return layers.includes(chartId) || layers.length >= MAX_LAYERS ? panel : stacked(panel, [...layers, chartId]);
}

/** Moves a layer one step up (towards the top, +1) or down in the drawing order. */
export function moveLayer(panel: PanelDraft, chartId: string, delta: 1 | -1): PanelDraft {
  const layers = [...panelLayers(panel)];
  const index = layers.indexOf(chartId), other = index + delta;
  if (index < 0 || other < 0 || other >= layers.length) return panel;
  [layers[index], layers[other]] = [layers[other]!, layers[index]!];
  return stacked(panel, layers);
}

/** Removes an added layer; the panel's own chart is changed or cleared on the panel card. */
export function removeLayer(panel: PanelDraft, chartId: string): PanelDraft {
  return chartId === panel.chartId ? panel : stacked(panel, panelLayers(panel).filter((id) => id !== chartId));
}

/** A new chart for the panel takes the old one's place among the layers; `keeps` says which added layers still fit it. */
export function withChart(panel: PanelDraft, chartId: string, keeps: (layer: string) => boolean): PanelDraft {
  if (!chartId) return { ...panel, chartId, layers: [] };
  const layers = panelLayers(panel).flatMap((id) => (id === panel.chartId ? [chartId] : id !== chartId && keeps(id) ? [id] : []));
  return stacked({ ...panel, chartId }, layers);
}

/** Drafts stored by the desktop session before six panels existed held slots and titles. */
export function storedDraft(value: unknown): FigureDraft {
  const stored = value as Partial<FigureDraft> & { slots?: string[]; panelTitles?: string[] };
  // Only the draft's own fields: an old session object also held its undo stacks and view state.
  const known = Object.fromEntries(DRAFT_KEYS.flatMap((key) => (stored[key] === undefined ? [] : [[key, stored[key]]]))) as Partial<FigureDraft>;
  // Panels stored before layers existed have no `layers`.
  if (stored.panels) return { ...emptyDraft(), ...known, panels: pad(stored.panels.slice(0, MAX_PANELS).map((panel) => ({ ...emptyPanel(), ...panel }))) };
  const panels = (stored.slots ?? []).map((chartId, index) => ({ ...emptyPanel(), chartId, title: stored.panelTitles?.[index] ?? "" }));
  const count = stored.count ?? 1;
  return { ...emptyDraft(), ...known, count, layout: ARRANGEMENTS[count]?.[0] ?? ARRANGEMENTS[1]![0]!, labelStyle: "upper", panels: pad(panels) };
}

export function movePanel(draft: FigureDraft, index: number, delta: number): FigureDraft {
  const panels = [...draft.panels];
  [panels[index], panels[index + delta]] = [panels[index + delta]!, panels[index]!];
  return { ...draft, panels };
}

/** Copies a panel into the first empty slot, growing the figure when every used slot is full. */
export function duplicatePanel(draft: FigureDraft, index: number): { draft: FigureDraft; target: number } | null {
  const empty = draft.panels.slice(0, draft.count).findIndex((panel) => !panel.chartId);
  const target = empty >= 0 ? empty : draft.count;
  if (target >= MAX_PANELS) return null;
  const panels = draft.panels.map((panel, i) => (i === target ? structuredClone(draft.panels[index]!) : panel));
  const grown = target >= draft.count ? withCount(draft, target + 1) : draft;
  return { draft: { ...grown, panels }, target };
}

type PlanAxis = RenderPlan["panels"][number]["axes"]["x"];

/** Starting values for a new annotation, placed on round values of the source chart's axes. */
export function noteDefaults(kind: FigureAnnotation["kind"], axes: { x: PlanAxis; y: PlanAxis } | null, scales: PanelDraft["axes"]): FigureAnnotation {
  const values = (axis: PlanAxis | undefined, scale: AxisDraft) => {
    const ticks = (axis?.ticks ?? []).map((tick) => tick.value).filter((value) => scale.scale !== "log10" || value > 0);
    return ticks.length ? ticks : scale.scale === "log10" ? [1, 10] : [0, 1];
  };
  const xs = values(axes?.x, scales.x), ys = values(axes?.y, scales.y);
  const at = (list: number[], share: number) => list[Math.min(list.length - 1, Math.round((list.length - 1) * share))]!;
  const pair = (list: number[]): [number, number] => {
    const low = at(list, 0.25), high = at(list, 0.75);
    return low < high ? [low, high] : [list[0]!, list.at(-1)! > list[0]! ? list.at(-1)! : list[0]! + 1];
  };
  switch (kind) {
    case "text": return { kind, x: at(xs, 0.5), y: at(ys, 0.75), text: "" };
    case "hline": return { kind, value: at(ys, 0.5), label: "" };
    case "vline": return { kind, value: at(xs, 0.5), label: "" };
    case "rect": return { kind, x: pair(xs), y: pair(ys), label: "" };
    // Not the top tick: a log axis is framed by the data and may end below it.
    case "bracket": return { kind, x: [xs[0]!, xs[1] ?? xs[0]! + 1], y: at(ys, 0.75), label: "" };
  }
}

export type DraftProblem = { panel: number; kind: "limits-half" | "limits-order" | "log-limit" | "text-empty" };

/** What keeps a draft from saving, per used panel; a control on a shared axis is not sent, so not checked. */
export function draftProblems(draft: FigureDraft): DraftProblem[] {
  return draft.panels.slice(0, draft.count).flatMap((panel, index) => {
    const found: DraftProblem["kind"][] = [];
    for (const dimension of ["x", "y"] as const) {
      if (shares(draft.axisMode, dimension)) continue;
      const { scale, domain } = panel.axes[dimension];
      if (!domain) continue;
      const [low, high] = domain as [number | null, number | null];
      if (low === null || high === null) found.push("limits-half");
      else if (low >= high) found.push("limits-order");
      else if (scale === "log10" && low <= 0) found.push("log-limit");
    }
    if (panel.notes.some((note) => note.kind === "text" && !note.text.trim())) found.push("text-empty");
    return [...new Set(found)].map((kind) => ({ panel: index, kind }));
  });
}

/** The source chart's own limits, when the panel keeps its scale; another scale is framed by the server. */
export function autoLimits(axis: PlanAxis | undefined, scale: AxisDraft["scale"]): [number, number] | null {
  return axis && axis.scale === scale ? axis.domain : null;
}
