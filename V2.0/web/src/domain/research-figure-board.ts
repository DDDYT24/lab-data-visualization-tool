import { z } from "zod";
import generatedSpec from "../../../contracts/figure-board-v1.schema.json";

export type FigureBoardSpec = {
  schemaVersion: 1; title: string; axisMode: "independent" | "shared_x" | "shared_y" | "shared_xy"; palette?: "blue_teal" | "grayscale"; fontSize?: number; gridVisible?: boolean; pointRendering?: "auto" | "vector" | "raster"; panelTitles?: string[]; panelChartIds: string[];
};
export const figureBoardSpecSchema = z.fromJSONSchema(
  generatedSpec as Parameters<typeof z.fromJSONSchema>[0],
).refine((value) => [1, 2, 4].includes((value as FigureBoardSpec).panelChartIds.length),
  "Choose one, two or four panels");
export const figureSourceSchema = z.object({ charts: z.array(z.object({
  id: z.string(), revisionId: z.string(), createdAt: z.string(), legacy: z.boolean(),
  spec: z.object({ kind: z.string(), title: z.string() }),
})) });
export type FigureSources = z.infer<typeof figureSourceSchema>;
const boardHeader = z.object({ id: z.string(), createdAt: z.string(), spec: figureBoardSpecSchema });
export const figureBoardHistorySchema = z.object({ boards: z.array(boardHeader) });
export const figureBoardSchema = boardHeader.extend({
  composed: z.boolean().default(false),
  artifactState: z.enum(["queued", "running", "ready", "failed", "unavailable"]).optional(),
  artifactError: z.string().nullable().optional(),
  panels: z.array(z.object({
    label: z.enum(["A", "B", "C", "D"]), chartId: z.string(), revisionId: z.string(),
    kind: z.enum(["grouped_strip", "paired", "violin", "line", "scatter", "box", "heatmap"]), title: z.string(), unit: z.string(),
    observations: z.number(), independentUnits: z.number().nullable(), resultSha256: z.string(),
  })),
});
export type FigureBoard = z.infer<typeof figureBoardSchema>;
export type FigureBoardHistory = z.infer<typeof figureBoardHistorySchema>;

export const figureInteractionSchema = z.object({
  marks: z.array(z.object({ id: z.string(), panel: z.string(), kind: z.string(), label: z.string(),
    cx: z.number(), cy: z.number(), inputCount: z.number(), highlighted: z.boolean() })),
  rows: z.array(z.object({ id: z.string(), panel: z.string(), sampleId: z.string().nullable(),
    sampleLabel: z.string().nullable(), value: z.number().nullable(), unit: z.string(),
    values: z.record(z.string(), z.unknown()).optional(),
    lineage: z.array(z.object({ sourceId: z.string(), row: z.number(), column: z.string(), sheet: z.string().nullable() })),
  })),
  totalRows: z.number(), offset: z.number(), sampled: z.boolean(), selectedKind: z.string().nullable(), selectedInputs: z.number(),
});
export type FigureInteraction = z.infer<typeof figureInteractionSchema>;
