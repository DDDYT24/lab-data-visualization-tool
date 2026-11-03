import { z } from "zod";
import generatedPlan from "../../../contracts/preparation-plan-v1.schema.json";

export const preparationPlanSchema = z.fromJSONSchema(
  generatedPlan as Parameters<typeof z.fromJSONSchema>[0],
);
export type PreparationPlan = {
  schemaVersion?: 1;
  mapping: Record<string, string | null>;
  steps: Record<string, unknown>[];
  exclusions: { observationId: string; reason: string }[];
  analysisUnit: "experimental_unit" | "sample" | "observation";
  validatePairs: boolean;
  pairTimes: string[];
};
const source = z.object({ id: z.string(), name: z.string(), sha256: z.string(), bytes: z.number() });
const summary = z.object({
  independentUnits: z.number().nullable(), knownIndependentUnits: z.number(),
  samples: z.number(), observations: z.number(), states: z.record(z.string(), z.number()),
  excluded: z.number(), includedObserved: z.number(), unknownIdentity: z.number(),
  analysisUnit: z.string(), withheldReasons: z.array(z.string()),
  mean: z.number().nullable(), analysisN: z.number().nullable(),
  pairDeltas: z.array(z.number()), pairDirection: z.array(z.string()),
});
const observation = z.object({
  id: z.string(), sampleLabel: z.string().nullable(), experimentalUnit: z.string().nullable(),
  group: z.string().nullable(), time: z.string().nullable(), batch: z.string().nullable(),
  technicalRepeat: z.number().nullable(), unit: z.string(), state: z.string(),
  value: z.number().nullable(), limit: z.number().nullable(), reportedText: z.string(), reportedUnit: z.string(),
  role: z.string(), notes: z.string().nullable(), recordLink: z.string().nullable(),
  exclusionReason: z.string().nullable(),
  lineage: z.array(z.object({ sourceId: z.string(), row: z.number(), column: z.string() })),
});
export const researchDataSchema = z.object({
  schemaVersion: z.literal(1), revisionId: z.string().nullable(), supported: z.literal(true),
  sources: z.array(source), columns: z.array(z.string()), summary,
  plan: preparationPlanSchema,
  samples: z.array(z.object({ id: z.string(), label: z.string(),
    experimentalUnit: z.string().nullable(), parentSample: z.string().nullable(), group: z.string().nullable() })),
  observations: z.array(observation),
  history: z.array(z.object({ inputRows: z.number(), outputRows: z.number(),
    algorithmVersion: z.string(), step: z.record(z.string(), z.unknown()) })),
  displayedObservations: z.number(), displayedSamples: z.number(), previewSampled: z.boolean(),
});
export const researchViewSchema = z.object({
  schemaVersion: z.literal(1), supported: z.boolean(), revisionId: z.string().nullable(),
  canUndo: z.boolean(), canRedo: z.boolean(), data: researchDataSchema.nullable(),
});
export const researchPreviewSchema = z.object({
  token: z.string(), expectedRevision: z.string().nullable(), data: researchDataSchema,
});
export const researchSourcesSchema = z.object({
  columns: z.array(z.string()), sources: z.array(source), rows: z.number(),
});
export type ResearchView = z.infer<typeof researchViewSchema>;
export type ResearchPreview = z.infer<typeof researchPreviewSchema>;


export const researchPageSchema = z.object({
  kind: z.enum(["observations", "samples"]), offset: z.number(), total: z.number(),
  observations: researchDataSchema.shape.observations,
  samples: researchDataSchema.shape.samples,
});
export type ResearchPage = z.infer<typeof researchPageSchema>;
