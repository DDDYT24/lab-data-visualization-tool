import { describe, expect, it } from "vitest";
import matrix from "../../../contracts/fixtures/v24/figure-spec.json";
import {
  addLayer, ARRANGEMENTS, draftFromSpec, draftProblems, duplicatePanel, emptyDraft, layoutCells, MAX_LAYERS, MAX_PANELS, moveLayer, noteDefaults,
  panelLayers, removeLayer, specFromDraft, storedDraft, withChart, withCount, type FigureDraft,
} from "./figure-draft";
import { figureSpecSchema, type FigureSpec } from "./figure-spec";
import type { FigureBoardSpec } from "./research-figure-board";

const MOSAIC = ["AAB", "CDE", "CDF"];
const six = (): FigureDraft => {
  const draft = withCount(emptyDraft(), 6);
  return { ...draft, layout: ARRANGEMENTS[6]!.find((layout) => layout.kind === "mosaic")!, panels: draft.panels.map((panel, i) => ({ ...panel, chartId: `chart${i}` })) };
};
const axis = (scale: "linear" | "category", ticks: number[]) => ({
  scale, domain: [ticks[0]! - 0.5, ticks.at(-1)! + 0.5] as [number, number], title: "", niceRule: null, grid: true,
  ticks: ticks.map((value) => ({ value, label: String(value) })),
});

describe("figure draft", () => {
  it.each(matrix.roundTrip)("opens and saves v1 board %# unchanged", (pair) => {
    expect(specFromDraft(draftFromSpec(pair.v1 as FigureBoardSpec))).toEqual(pair.v1);
  });

  it("saves a six-panel mosaic with a log axis and a bracket as v2, and reopens it", () => {
    const draft = six();
    draft.panels[0] = { ...draft.panels[0]!, axes: { x: draft.panels[0]!.axes.x, y: { scale: "log10", domain: [0.1, 100], ticks: null } },
      notes: [{ kind: "bracket", x: [0, 1], y: 50, label: "p < 0.01" }] };
    const spec = specFromDraft(draft) as FigureSpec;
    expect(spec.schemaVersion).toBe(2);
    expect(figureSpecSchema.safeParse(spec).success).toBe(true);
    expect(spec.layout).toMatchObject({ kind: "mosaic", rows: 3, cols: 3, mosaic: MOSAIC });
    expect(spec.panels.map((panel) => panel.label).join("")).toBe("ABCDEF");
    expect(spec.panels[0]!.axes.y.scale).toBe("log10");
    expect(spec.panels[0]!.annotations).toEqual([{ kind: "bracket", x: [0, 1], y: 50, label: "p < 0.01" }]);
    expect(specFromDraft(draftFromSpec(spec))).toEqual(spec);
  });

  it("saves three panels, a label style or a note as v2 only", () => {
    const three = withCount({ ...emptyDraft(), panels: emptyDraft().panels.map((p) => ({ ...p, chartId: "c" })) }, 3);
    expect(specFromDraft(three).schemaVersion).toBe(2);
    const two = withCount({ ...emptyDraft(), panels: three.panels }, 2);
    expect(specFromDraft(two).schemaVersion).toBe(1);
    expect(specFromDraft({ ...two, labelStyle: "lower" }).schemaVersion).toBe(2);
    expect(specFromDraft({ ...two, panels: two.panels.map((p, i) => (i ? p : { ...p, notes: [{ kind: "hline", value: 1, label: "" }] })) }).schemaVersion).toBe(2);
  });

  it("does not send a panel's control on a shared axis, and keeps it in the draft", () => {
    const draft = six();
    draft.panels[1] = { ...draft.panels[1]!, axes: { x: draft.panels[1]!.axes.x, y: { scale: "symlog", domain: null, ticks: null } } };
    const spec = specFromDraft({ ...draft, axisMode: "shared_y" }) as FigureSpec;
    expect(spec.panels[1]!.axes.y.scale).toBe("linear");
    expect((specFromDraft({ ...draft, axisMode: "shared_x" }) as FigureSpec).panels[1]!.axes.y.scale).toBe("symlog");
  });

  it.each(Object.entries(ARRANGEMENTS))("offers only arrangements the validator accepts for %s panels", (count, layouts) => {
    for (const layout of layouts) {
      const draft = { ...withCount(emptyDraft(), Number(count)), layout };
      const spec = specFromDraft({ ...draft, panels: draft.panels.map((p) => ({ ...p, chartId: "c" })) });
      const result = spec.schemaVersion === 2 ? figureSpecSchema.safeParse(spec) : { success: true };
      expect(result.success, JSON.stringify(layout)).toBe(true);
      expect(layoutCells(layout, Number(count))).toHaveLength(Number(count));
    }
  });

  it("lays out a mosaic the way the API does", () => {
    expect(layoutCells({ kind: "mosaic", rows: 3, cols: 3, mosaic: MOSAIC }, 6)).toEqual([
      [0, 0, 1, 2], [0, 2, 1, 1], [1, 0, 2, 1], [1, 1, 2, 1], [1, 2, 1, 1], [2, 2, 1, 1],
    ]);
  });

  it("keeps an arrangement that still fits a new count and replaces one that does not", () => {
    const grid23 = withCount(withCount(emptyDraft(), 6), 5);
    expect(grid23.layout).toEqual({ kind: "grid", rows: 2, cols: 3, mosaic: null });
    expect(withCount(six(), 4).layout).toEqual(ARRANGEMENTS[4]![0]);
  });

  it("duplicates into the next slot until six panels are full", () => {
    let draft: FigureDraft = { ...emptyDraft(), panels: emptyDraft().panels.map((p, i) => (i ? p : { ...p, chartId: "a", title: "T" })) };
    for (let count = 2; count <= MAX_PANELS; count++) {
      const result = duplicatePanel(draft, 0)!;
      expect(result.target).toBe(count - 1);
      draft = result.draft;
      expect(draft.count).toBe(count);
      expect(draft.panels[count - 1]).toMatchObject({ chartId: "a", title: "T" });
    }
    expect(duplicatePanel(draft, 0)).toBeNull();
  });

  it("reads a desktop draft stored before six panels existed", () => {
    const draft = storedDraft({ title: "Old", count: 2, slots: ["a", "b", "", ""], panelTitles: ["x", "", "", ""], axisMode: "shared_x" });
    expect(draft.panels).toHaveLength(MAX_PANELS);
    expect(draft.panels.slice(0, 2).map((p) => [p.chartId, p.title])).toEqual([["a", "x"], ["b", ""]]);
    expect(draft.layout).toEqual(ARRANGEMENTS[2]![0]);
    expect(specFromDraft(draft)).toMatchObject({ schemaVersion: 1, panelChartIds: ["a", "b"], axisMode: "shared_x" });
  });

  it("names what keeps a draft from saving, and skips a shared axis", () => {
    const draft = six();
    const axes = (y: object) => ({ x: draft.panels[0]!.axes.x, y: { scale: "linear" as const, domain: null, ticks: null, ...y } });
    draft.panels[0] = { ...draft.panels[0]!, axes: axes({ domain: [5, null] }) };
    draft.panels[1] = { ...draft.panels[1]!, axes: axes({ domain: [5, 2] }) };
    draft.panels[2] = { ...draft.panels[2]!, axes: axes({ scale: "log10", domain: [0, 2] }) };
    draft.panels[3] = { ...draft.panels[3]!, notes: [{ kind: "text", x: 0, y: 0, text: " " }] };
    expect(draftProblems(draft)).toEqual([
      { panel: 0, kind: "limits-half" }, { panel: 1, kind: "limits-order" },
      { panel: 2, kind: "log-limit" }, { panel: 3, kind: "text-empty" },
    ]);
    expect(draftProblems({ ...draft, axisMode: "shared_y" })).toEqual([{ panel: 3, kind: "text-empty" }]);
  });

  it("keeps settings the editor does not show, and renames other letters to A–F", () => {
    const spec = specFromDraft(six()) as FigureSpec;
    const custom = {
      ...spec,
      canvas: { ...spec.canvas, sizePreset: "custom" as const, width: 180, height: 120, dpi: 600 },
      layout: { ...spec.layout, gutter: 0.2, mosaic: ["PPQ", "RST", "RSU"] },
      palette: { ...spec.palette, sequential: "magma" },
      panels: spec.panels.map((panel, i) => ({ ...panel, label: "PQRSTU"[i]! })),
    };
    const again = specFromDraft(draftFromSpec(custom)) as FigureSpec;
    expect(again.canvas).toEqual(custom.canvas);
    expect(again.layout.gutter).toBe(0.2);
    expect(again.palette.sequential).toBe("magma");
    expect(again.layout.mosaic).toEqual(MOSAIC);
    expect(again.panels.map((panel) => panel.label).join("")).toBe("ABCDEF");
    expect(figureSpecSchema.safeParse(again).success).toBe(true);
  });

  it("saves a layered panel with its own chart first and the drawing order in zIndex, and reopens it", () => {
    const draft = withCount(emptyDraft(), 2);
    const points = { ...draft.panels[0]!, chartId: "points" };
    // A violin added on top, then moved under the raw points.
    const layered = moveLayer(addLayer(points, "violin"), "violin", -1);
    expect(panelLayers(layered)).toEqual(["violin", "points"]);
    draft.panels = draft.panels.map((panel, i) => (i === 0 ? layered : i === 1 ? { ...panel, chartId: "other" } : panel));
    const spec = specFromDraft(draft) as FigureSpec;
    expect(spec.schemaVersion).toBe(2);
    expect(figureSpecSchema.safeParse(spec).success).toBe(true);
    expect(spec.panels[0]!.layers.map((layer) => [layer.chartId, layer.style.zIndex])).toEqual([["points", 1], ["violin", 0]]);
    expect(spec.panels[1]!.layers.map((layer) => [layer.chartId, layer.style.zIndex])).toEqual([["other", 0]]);
    const reopened = draftFromSpec(spec);
    expect(reopened.panels[0]).toMatchObject({ chartId: "points", layers: ["violin", "points"] });
    expect(reopened.panels[1]!.layers).toEqual([]);
    expect(specFromDraft(reopened)).toEqual(spec);
    // Back to one layer, the figure is a v1 board again.
    expect(specFromDraft({ ...draft, panels: draft.panels.map((p, i) => (i ? p : removeLayer(p, "violin"))) }).schemaVersion).toBe(1);
  });

  it("edits a panel's layers within the cap and keeps its own chart", () => {
    let panel = { ...emptyDraft().panels[0]!, chartId: "own" };
    expect(panelLayers(panel)).toEqual(["own"]);
    expect(panelLayers({ ...panel, chartId: "" })).toEqual([]);
    expect(addLayer(addLayer(panel, "a"), "a").layers).toEqual(["own", "a"]);
    for (let i = 1; i < MAX_LAYERS + 2; i += 1) panel = addLayer(panel, `l${i}`);
    expect(panelLayers(panel)).toHaveLength(MAX_LAYERS);
    expect(removeLayer(panel, "own")).toBe(panel);
    expect(moveLayer(panel, "own", -1)).toBe(panel);
    expect(moveLayer(panel, "own", 1).layers.slice(0, 2)).toEqual(["l1", "own"]);
    // A new own chart takes the old one's place; added layers stay only if they fit it.
    const two = addLayer({ ...emptyDraft().panels[0]!, chartId: "own" }, "a");
    expect(withChart(two, "new", () => true).layers).toEqual(["new", "a"]);
    expect(withChart(two, "new", () => false)).toMatchObject({ chartId: "new", layers: [] });
    expect(withChart(two, "a", () => true)).toMatchObject({ chartId: "a", layers: [] });
    expect(withChart(two, "", () => true)).toMatchObject({ chartId: "", layers: [] });
  });

  it("gives panels stored before layers existed no layers", () => {
    const old: Record<string, unknown> = { ...emptyDraft().panels[0]!, chartId: "a" };
    delete old.layers;
    const draft = storedDraft({ ...emptyDraft(), panels: [old] });
    expect(draft.panels[0]).toMatchObject({ chartId: "a", layers: [] });
    expect(specFromDraft(draft)).toMatchObject({ schemaVersion: 1, panelChartIds: ["a"] });
  });

  it("takes only draft fields from an old desktop session", () => {
    const draft = storedDraft({ title: "Old", count: 1, slots: ["a"], undoDrafts: [{}], editOpen: true });
    expect(Object.keys(draft).sort()).toEqual(Object.keys(emptyDraft()).sort());
  });

  it("starts notes on round values of the source chart's axes", () => {
    const axes = { x: axis("category", [0, 1, 2]), y: axis("linear", [0, 5, 10, 15]) };
    const linear = { scale: "linear" as const, domain: null, ticks: null };
    expect(noteDefaults("bracket", axes, { x: linear, y: linear })).toEqual({ kind: "bracket", x: [0, 1], y: 10, label: "" });
    expect(noteDefaults("hline", axes, { x: linear, y: { ...linear, scale: "log10" } })).toEqual({ kind: "hline", value: 10, label: "" });
    expect(noteDefaults("rect", null, { x: linear, y: linear })).toEqual({ kind: "rect", x: [0, 1], y: [0, 1], label: "" });
  });
});
