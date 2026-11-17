import type { RenderPlan } from "./render-plan";
import { completeFigure, projectBoard, upgradeBoard, type FigureAnnotation, type FigurePanel, type FigureSpec } from "./figure-spec";
import type { FigureBoardSpec } from "./research-figure-board";

export const MAX_PANELS = 6;
export const PANEL_LABELS = "ABCDEF";

export type FigureLayout = Pick<FigureSpec["layout"], "kind" | "rows" | "cols" | "mosaic">;
export type AxisDraft = FigurePanel["axes"]["x"];
export type PanelDraft = { chartId: string; title: string; axes: { x: AxisDraft; y: AxisDraft }; notes: FigureAnnotation[] };
/** What the figure editor holds: always six panel slots, of which the first `count` are used. */
export type FigureDraft = {
  title: string; axisMode: FigureBoardSpec["axisMode"]; palette: "blue_teal" | "grayscale"; fontSize: number;
  grid: boolean; pointRendering: "auto" | "vector" | "raster"; labelStyle: FigureSpec["layout"]["labelStyle"];
  count: number; layout: FigureLayout; panels: PanelDraft[];
};

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
const emptyPanel = (): PanelDraft => ({ chartId: "", title: "", axes: { x: linear(), y: linear() }, notes: [] });
const pad = (panels: PanelDraft[]) => [...panels, ...Array.from({ length: MAX_PANELS - panels.length }, emptyPanel)];

export function emptyDraft(): FigureDraft {
  return {
    title: "", axisMode: "independent", palette: "blue_teal", fontSize: 10, grid: true, pointRendering: "auto",
    labelStyle: "upper", count: 1, layout: ARRANGEMENTS[1]![0]!, panels: pad([]),
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
  return {
    title: figure.title, axisMode: figure.axisMode, palette, fontSize: figure.fontSize, grid: figure.gridVisible,
    pointRendering: figure.pointRendering, labelStyle: figure.layout.labelStyle, count: figure.panels.length,
    layout: { kind: figure.layout.kind, rows: figure.layout.rows, cols: figure.layout.cols, mosaic: figure.layout.mosaic },
    panels: pad(figure.panels.map((panel) => ({
      chartId: panel.layers[0]?.chartId ?? "", title: panel.title, axes: panel.axes, notes: panel.annotations,
    }))),
  };
}

/** The figure to save: v1 when v1 says the same thing, so templates and older readers keep working. */
export function specFromDraft(draft: FigureDraft): FigureBoardSpec | FigureSpec {
  const figure = completeFigure({
    schemaVersion: 2, title: draft.title, fontSize: draft.fontSize, gridVisible: draft.grid,
    axisMode: draft.axisMode, pointRendering: draft.pointRendering,
    layout: { ...draft.layout, labelStyle: draft.labelStyle },
    palette: { categorical: draft.palette },
    panels: draft.panels.slice(0, draft.count).map((panel, index) => ({
      id: `p${index + 1}`, label: PANEL_LABELS[index], title: panel.title,
      // A shared axis is framed across the panels, so a panel's own control on it is not sent;
      // the draft keeps it for when the axis is no longer shared.
      axes: {
        x: shares(draft.axisMode, "x") ? linear() : panel.axes.x,
        y: shares(draft.axisMode, "y") ? linear() : panel.axes.y,
      },
      annotations: panel.notes, layers: [{ kind: "chart", chartId: panel.chartId }],
    })),
  });
  return projectBoard(figure) ?? figure;
}

/** Drafts stored by the desktop session before six panels existed held slots and titles. */
export function storedDraft(value: unknown): FigureDraft {
  const stored = value as Partial<FigureDraft> & { slots?: string[]; panelTitles?: string[] };
  if (stored.panels) return { ...emptyDraft(), ...stored, panels: pad(stored.panels.slice(0, MAX_PANELS)) };
  const panels = (stored.slots ?? []).map((chartId, index) => ({ ...emptyPanel(), chartId, title: stored.panelTitles?.[index] ?? "" }));
  const count = stored.count ?? 1;
  return { ...emptyDraft(), ...stored, count, layout: ARRANGEMENTS[count]?.[0] ?? ARRANGEMENTS[1]![0]!, labelStyle: "upper", panels: pad(panels) };
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

/** The source chart's own limits, when the panel keeps its scale; another scale is framed by the server. */
export function autoLimits(axis: PlanAxis | undefined, scale: AxisDraft["scale"]): [number, number] | null {
  return axis && axis.scale === scale ? axis.domain : null;
}
