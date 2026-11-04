import { z } from "zod";
import generatedPlan from "../../../contracts/render-plan-v1.schema.json";

type Tick = { value: number; label: string };
type Axis = {
  scale: "linear" | "category"; domain: [number, number]; ticks: Tick[]; title: string;
  niceRule: "nice-1-2-5-v1" | null; grid: boolean;
};
type MarkBase = { id: string; group: string; color: string };
export type RenderPlanMark =
  | (MarkBase & { kind: "points"; marker: "o" | "s" | "^" | "D" | "v"; size: number; opacity: number; x: number[]; y: number[]; sourceRefs: string[] })
  | (MarkBase & { kind: "segments"; width: number; opacity: number; x0: number[]; y0: number[]; x1: number[]; y1: number[]; sourceRefs: [string, string][] })
  | (MarkBase & { kind: "violin"; opacity: number; position: number; grid: number[]; halfWidths: number[]; bandwidth: number; bandwidthRule: string; scale: "equal_width" | "common_density"; n: number; sourceRefs: string[] })
  | (MarkBase & { kind: "summary"; width: number; x0: number; x1: number; y: number; statistic: "mean" | "median"; summaryUnit: "observation" | "sample" | "experimental_unit"; n: number })
  | (MarkBase & { kind: "rects"; opacity: number; width: number; shape: "stairs" | "bars"; x0: number[]; x1: number[]; y0: number[]; y1: number[] })
  | (MarkBase & { kind: "box"; medianColor: string; width: number; position: number; halfWidth: number; low: number; q1: number; median: number; q3: number; high: number; outliers: number[] });
export type RenderPlanPanel = {
  id: string; title: string; axes: { x: Axis; y: Axis }; marks: RenderPlanMark[];
  legend: { label: string; color: string; symbol: "line" | "marker" }[];
};
// The Python RenderPlan model owns this contract; the frontend only draws what it states.
export type RenderPlan = {
  schemaVersion: 1; contract: "render-plan-v1"; planHash: string;
  source: { kind: "grouped_strip" | "paired" | "violin" | "histogram" | "box" | "bar"; resultSha256: string; algorithmVersion: string };
  geometry: { contract: "chart-render-v2"; axis: string; jitter: string | null; density: string | null; histogram: string | null; quantile: string | null };
  rendering: { pointRendering: "auto" | "vector" | "raster"; rasterizePoints: boolean };
  panels: RenderPlanPanel[]; disclosures: string[];
};

export const renderPlanSchema = z.fromJSONSchema(
  generatedPlan as Parameters<typeof z.fromJSONSchema>[0],
) as unknown as z.ZodType<RenderPlan>;
