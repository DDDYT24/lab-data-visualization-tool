import { describe, expect, it } from "vitest";

import { renderPlanSchema, type RenderPlan } from "./render-plan";

const digest = "a".repeat(64);
const axis = (scale: "linear" | "category") => ({
  scale, domain: [0, 1], ticks: [{ value: 0, label: "0" }], title: "", niceRule: scale === "linear" ? "nice-1-2-5-v1" : null, grid: scale === "linear",
});
const plan: RenderPlan = {
  schemaVersion: 1, contract: "render-plan-v1", planHash: digest,
  source: { kind: "violin", resultSha256: digest, algorithmVersion: "scientific-charts-v1" },
  geometry: { contract: "chart-render-v2", axis: "nice-1-2-5-v1", jitter: "sha256-uniform-v1", density: null, histogram: null, quantile: null, decimation: null },
  rendering: { pointRendering: "auto", rasterizePoints: false },
  panels: [{
    id: "A", title: "Violin distribution", legend: [{ label: "A", color: "#1768ac", symbol: "line" }],
    axes: { x: axis("category") as RenderPlan["panels"][0]["axes"]["x"], y: axis("linear") as RenderPlan["panels"][0]["axes"]["y"], y2: null },
    marks: [
      { kind: "violin", id: "density:0", group: "A", color: "#1768ac", opacity: 0.3, position: 0, grid: [0, 1], halfWidths: [0.1, 0.2], bandwidth: 0.5, bandwidthRule: "scott-sample-sd", scale: "equal_width", n: 3, sourceRefs: ["o1"] },
      { kind: "segments", id: "pairs:0", group: "A", color: "#1768ac", width: 1.2, opacity: 0.65, x0: [0], y0: [1], x1: [1], y1: [2], sourceRefs: [["o1", "o2"]] },
      { kind: "points", id: "points:0", group: "A", color: "#1768ac", marker: "o", size: 4, opacity: 0.7, axis: "y", x: [0.1], y: [1], sourceRefs: ["o1"], decimation: null },
      { kind: "polyline", id: "line:0", group: "A", color: "#1768ac", width: 1.5, lineStyle: "dashed", markerSize: 3, axis: "y2", x: [0], y: [1], decimation: { rule: "min-max-envelope-v1", input: 9, drawn: 1 } },
      { kind: "band", id: "band:0", group: "A", color: "#1768ac", opacity: 0.15, axis: "y", x: [0], lower: [0], upper: [1] },
      { kind: "errorbar", id: "error:0", group: "A", color: "#1768ac", width: 1, capSize: 2, opacity: 0.9, axis: "y", x: [0], y: [1], error: [0.5] },
      { kind: "summary", id: "summary:0", group: "A", color: "#192d40", width: 2.4, x0: -0.25, x1: 0.25, y: 1, statistic: "mean", summaryUnit: "experimental_unit", n: 3 },
      { kind: "rects", id: "bar:0", group: "A", color: "#1768ac", opacity: 0.55, width: 1, shape: "stairs", axis: "y", x0: [0], x1: [1], y0: [0], y1: [3], decimation: null },
      { kind: "box", id: "box:0", group: "A", color: "#000000", medianColor: "#ff7f0e", width: 1, position: 1, halfWidth: 0.075, low: 1, q1: 2, median: 3, q3: 4, high: 5, outliers: [9] },
    ],
  }],
  disclosures: ["Observations=1; independent units=1; excluded=0"],
};

describe("render plan contract", () => {
  it("accepts every mark kind the Python model defines", () => {
    expect(renderPlanSchema.parse(plan)).toEqual(plan);
  });

  it("rejects marks and colours the contract does not define", () => {
    const unknownMark = structuredClone(plan) as unknown as { panels: { marks: { kind: string }[] }[] };
    unknownMark.panels[0].marks[0].kind = "bar";
    expect(renderPlanSchema.safeParse(unknownMark).success).toBe(false);
    const namedColor = structuredClone(plan);
    namedColor.panels[0].marks[2].color = "red";
    expect(renderPlanSchema.safeParse(namedColor).success).toBe(false);
  });
});
