import { z } from "zod";
import generatedPlan from "../../../contracts/render-plan-v1.schema.json";
import geometry from "../../../contracts/chart-render-v2.json";

export type AxisScale = "linear" | "category" | "log10" | "symlog";
const SYMLOG_THRESHOLD = geometry.axis.symlogThreshold;
/** Where a value sits along an axis, in the units the axis is linear in; the API uses the same rule. */
export function axisPosition(scale: AxisScale, value: number): number {
  if (scale === "log10") return Math.log10(value);
  if (scale === "symlog") return Math.sign(value) * Math.log10(1 + Math.abs(value) / SYMLOG_THRESHOLD);
  return value;
}
export function axisValue(scale: AxisScale, position: number): number {
  if (scale === "log10") return 10 ** position;
  if (scale === "symlog") return Math.sign(position) * (10 ** Math.abs(position) - 1) * SYMLOG_THRESHOLD;
  return position;
}

type Tick = { value: number; label: string };
type Axis = {
  scale: AxisScale; domain: [number, number]; ticks: Tick[]; title: string;
  niceRule: "nice-1-2-5-v1" | null; grid: boolean;
};
// `layer` is set only on marks of a panel's second and later layers (S4e).
type MarkBase = { id: string; group: string; color: string; layer?: number };
type YAxisName = "y" | "y2";
type Decimation = { rule: string; input: number; drawn: number } | null;
export type RenderPlanMark =
  | (MarkBase & { kind: "points"; marker: "o" | "s" | "^" | "D" | "v"; size: number; opacity: number; axis: YAxisName; x: number[]; y: number[]; sourceRefs: string[]; decimation: Decimation })
  | (MarkBase & { kind: "polyline"; width: number; lineStyle: "solid" | "dashed" | "dotted" | "dashdot"; markerSize: number; axis: YAxisName; x: number[]; y: number[]; decimation: Decimation })
  | (MarkBase & { kind: "band"; opacity: number; axis: YAxisName; x: number[]; lower: number[]; upper: number[] })
  | (MarkBase & { kind: "errorbar"; width: number; capSize: number; opacity: number; axis: YAxisName; x: number[]; y: number[]; error: number[] })
  | (MarkBase & { kind: "segments"; width: number; opacity: number; x0: number[]; y0: number[]; x1: number[]; y1: number[]; sourceRefs: [string, string][] })
  | (MarkBase & { kind: "violin"; opacity: number; position: number; grid: number[]; halfWidths: number[]; bandwidth: number; bandwidthRule: string; scale: "equal_width" | "common_density"; n: number; sourceRefs: string[] })
  | (MarkBase & { kind: "summary"; width: number; x0: number; x1: number; y: number; statistic: "mean" | "median"; summaryUnit: "observation" | "sample" | "experimental_unit"; n: number })
  | (MarkBase & { kind: "rects"; opacity: number; width: number; shape: "stairs" | "bars"; axis: YAxisName; x0: number[]; x1: number[]; y0: number[]; y1: number[]; decimation: Decimation })
  | (MarkBase & { kind: "box"; medianColor: string; width: number; position: number; halfWidth: number; low: number; q1: number; median: number; q3: number; high: number; outliers: number[] })
  | { kind: "mesh"; id: string; group: string; layer?: number; shape: "cells" | "surface"; x: number[]; y: number[]; values: (number | null)[][]; colorDomain: [number, number]; palette: string[]; colorTitle: string; decimation: Decimation };
export type RenderPlanNote =
  | { kind: "text"; x: number; y: number; text: string }
  | { kind: "hline" | "vline"; value: number; label: string }
  | { kind: "rect"; x: [number, number]; y: [number, number]; label: string; labelAt: [number, number] }
  | { kind: "bracket"; x: [number, number]; y: number; foot: number; label: string; labelAt: [number, number] };
export type RenderPlanPanel = {
  // A y domain that runs from high to low is drawn reversed, as heatmaps are.
  id: string; title: string; axes: { x: Axis; y: Axis; y2: Axis | null; z: Axis | null }; marks: RenderPlanMark[];
  legend: { label: string; color: string; symbol: "line" | "marker" | "dashed" }[];
  // Absent on a panel without annotations.
  notes?: RenderPlanNote[];
};
// The Python RenderPlan model owns this contract; the frontend only draws what it states.
export type RenderPlan = {
  schemaVersion: 1; contract: "render-plan-v1"; planHash: string;
  source: { kind: "grouped_strip" | "paired" | "violin" | "histogram" | "box" | "bar" | "line" | "scatter" | "heatmap" | "surface3d"; resultSha256: string; algorithmVersion: string };
  geometry: { contract: "chart-render-v2"; axis: string; jitter: string | null; density: string | null; histogram: string | null; quantile: string | null; decimation: string | null };
  rendering: { pointRendering: "auto" | "vector" | "raster"; rasterizePoints: boolean };
  panels: RenderPlanPanel[]; disclosures: string[];
};

export const renderPlanSchema = z.fromJSONSchema(
  generatedPlan as Parameters<typeof z.fromJSONSchema>[0],
) as unknown as z.ZodType<RenderPlan>;
