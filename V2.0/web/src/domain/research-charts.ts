import { z } from "zod";
import specJson from "../../../contracts/scientific-chart-v1.schema.json";
import { researchDataSchema } from "./research-preparation";

export const scientificChartSpecSchema = z.fromJSONSchema(specJson as Parameters<typeof z.fromJSONSchema>[0]);
export type ScientificChartSpec = {
  schemaVersion: 1; kind: "grouped_strip" | "paired" | "violin"; title: string; locale: "en" | "zh";
  jitterSeed: string; jitterWidth: number; showPoints: boolean; summary: "none" | "mean" | "median";
  summaryUnit: "observation" | "sample" | "experimental_unit"; pairTimes: string[];
  categoryStyles?: { group: string | null; color: string; marker: "o" | "s" | "^" | "D" | "v" }[];
  pointRendering?: "auto" | "vector" | "raster";
  bandwidth: number | null; densityScale: "equal_width" | "common_density";
};
const chartHeader = z.object({
  id: z.string(), revisionId: z.string(), createdAt: z.string(),
  spec: scientificChartSpecSchema,
});
export const chartHistorySchema = z.object({ charts: z.array(chartHeader) });
export const scientificChartSchema = chartHeader.extend({
  resultSha256: z.string(), observations: z.number(), excluded: z.number(), totalRows: z.number(),
  offset: z.number(), pairCount: z.number(),
  sources: researchDataSchema.shape.sources,
  rows: researchDataSchema.shape.observations,
  groups: z.array(z.object({
    label: z.string().nullable(), observations: z.number(), knownSamples: z.number(),
    unknownSamples: z.number(), independentUnits: z.number().nullable(),
    summary: z.number().nullable(), summaryN: z.number().nullable(),
  })),
  pairs: z.array(z.object({
    sampleId: z.string(), sampleLabel: z.string().nullable(), pairKey: z.string(),
    group: z.string().nullable(), firstId: z.string(), secondId: z.string(),
    first: z.number(), second: z.number(), delta: z.number(),
  })),
  densityMethods: z.array(z.object({
    group: z.string().nullable(), bandwidth: z.number(), bandwidthRule: z.string(),
    nObservations: z.number(), scale: z.string(), support: z.array(z.number()),
    gridSize: z.number(), inputUnit: z.string(), boundaryCorrection: z.string(),
  })),
});
export type ScientificChart = z.infer<typeof scientificChartSchema>;
export type ChartHistory = z.infer<typeof chartHistorySchema>;
