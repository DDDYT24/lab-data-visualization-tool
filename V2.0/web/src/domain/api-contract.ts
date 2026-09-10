import { z } from "zod";

import { chartSpecSchema } from "./chart-spec";

export const apiVersionSchema = z.literal("v1");
export const PROJECT_DESCRIPTION_MAX_UTF8_BYTES = 4_000;
export const projectDescriptionTextSchema = z.string().superRefine((value, ctx) => {
  if (value.includes("\0")) {
    ctx.addIssue({ code: "custom", message: "Description must not contain NUL." });
  }
  if (new TextEncoder().encode(value).byteLength > PROJECT_DESCRIPTION_MAX_UTF8_BYTES) {
    ctx.addIssue({
      code: "custom",
      message: `Description must not exceed ${PROJECT_DESCRIPTION_MAX_UTF8_BYTES} UTF-8 bytes.`,
    });
  }
});
const downloadLocationSchema = z.union([
  z.string().url(),
  z.string().startsWith("/api/v1/"),
]);

export const sourceFileSchema = z.object({
  name: z.string().min(1),
  size: z.number().int().nonnegative(),
  mediaType: z.string(),
  sheetName: z.string().nullable(),
  availableSheets: z.array(z.string()).default([]),
  headerRow: z.number().int().positive().nullable(),
});

export const processingJobSchema = z.object({
  apiVersion: apiVersionSchema,
  id: z.string().min(1),
  projectId: z.string().min(1),
  stage: z.enum([
    "queued",
    "uploading",
    "parsing",
    "profiling",
    "ready",
    "failed",
  ]),
  progress: z.number().min(0).max(100),
  message: z.string(),
  errorCode: z.string().nullable(),
});

export const experimentContextSchema = z.object({
  experimentId: z.string().min(1),
  experimentRunId: z.string().min(1),
  title: z.string().min(1).max(200),
  runLabel: z.string().min(1).max(200),
  replicateId: z.string().min(1).max(100).nullable(),
  batchId: z.string().min(1).max(100).nullable(),
});

export const projectSessionSchema = z.object({
  apiVersion: apiVersionSchema,
  projectId: z.string().min(1),
  storageMode: z.enum(["temporary-cloud", "saved-cloud", "local"]),
  description: projectDescriptionTextSchema,
  currentRevisionId: z.string().uuid().nullable(),
  source: sourceFileSchema,
  job: processingJobSchema.nullable(),
  expiresAt: z.string().datetime().nullable(),
  experiment: experimentContextSchema.nullable().default(null),
});

export const previewColumnSchema = z.object({
  field: z.string().min(1),
  label: z.string().min(1),
  kind: z.enum(["number", "text", "datetime", "boolean"]),
  unit: z.string().nullable(),
  nullable: z.boolean(),
});

export const previewValueSchema = z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.null(),
]);

export const previewRowSchema = z
  .object({
    rowId: z.union([z.string(), z.number()]),
  })
  .catchall(previewValueSchema);

export const dataPreviewSchema = z.object({
  apiVersion: apiVersionSchema,
  projectId: z.string().min(1),
  columns: z.array(previewColumnSchema).min(1),
  rows: z.array(previewRowSchema),
  totalRows: z.number().int().nonnegative(),
  sampled: z.boolean(),
  sampleStrategy: z.enum(["none", "evenly-distributed"]),
});

const messageParamSchema = z.union([z.string(), z.number(), z.boolean(), z.null()]);
const messageParamsSchema = z.record(z.string(), messageParamSchema);

export const qualityFindingSchema = z.object({
  id: z.string().min(1),
  kind: z.enum([
    "missing",
    "duplicate",
    "type-conflict",
    "extreme-value",
    "sudden-change",
    "outside-range",
    "trend-inconsistent",
  ]),
  severity: z.enum(["info", "warning", "error"]),
  column: z.string().nullable(),
  rowIds: z.array(z.union([z.string(), z.number()])),
  affectedCount: z.number().int().nonnegative().default(0),
  rowIdsTruncated: z.boolean().default(false),
  summary: z.string().min(1),
  reason: z.string().min(1),
  summaryCode: z.string().min(1).nullable().optional(),
  summaryParams: messageParamsSchema.nullable().optional(),
  reasonCode: z.string().min(1).nullable().optional(),
  reasonParams: messageParamsSchema.nullable().optional(),
});

export const qualityReportSchema = z.object({
  apiVersion: apiVersionSchema,
  projectId: z.string().min(1),
  totalRows: z.number().int().nonnegative(),
  validRows: z.number().int().nonnegative(),
  missingValues: z.number().int().nonnegative(),
  duplicateRows: z.number().int().nonnegative(),
  suspiciousPoints: z.number().int().nonnegative(),
  findings: z.array(qualityFindingSchema),
});

export const cleaningDecisionSchema = z.object({
  findingId: z.string().min(1),
  action: z.enum(["ignore", "exclude", "remove"]),
});

export const cleaningDecisionsResponseSchema = z.object({
  apiVersion: apiVersionSchema,
  projectId: z.string().min(1),
  decisions: z.array(cleaningDecisionSchema),
  updatedAt: z.string().datetime(),
});

export const savedChartResponseSchema = z.object({
  apiVersion: apiVersionSchema,
  projectId: z.string().min(1),
  chart: chartSpecSchema,
  updatedAt: z.string().datetime(),
});

export const projectDescriptionResponseSchema = z.object({
  apiVersion: apiVersionSchema,
  projectId: z.string().min(1),
  description: projectDescriptionTextSchema,
  revisionId: z.string().uuid(),
  revisionNumber: z.number().int().positive(),
  updatedAt: z.string().datetime(),
});

export const projectSummarySchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1),
  sourceName: z.string().min(1),
  chartType: chartSpecSchema.shape.type,
  updatedAt: z.string().datetime(),
  storageMode: z.enum(["saved-cloud", "local"]),
  thumbnailUrl: z.string().url().nullable(),
  experiment: experimentContextSchema.nullable().default(null),
});

export const projectListSchema = z.object({
  apiVersion: apiVersionSchema,
  projects: z.array(projectSummarySchema),
});

export const sharedChartSchema = z.object({
  apiVersion: apiVersionSchema,
  token: z.string().min(1),
  title: z.string().min(1),
  description: z.string(),
  updatedAt: z.string().datetime(),
  chart: chartSpecSchema,
  preview: dataPreviewSchema,
  analysis: z.lazy(() => chartAnalysisSchema).optional(),
  downloads: z.object({
    png: downloadLocationSchema.nullable(),
    svg: downloadLocationSchema.nullable(),
    pdf: downloadLocationSchema.nullable(),
  }),
});

export const requestEmailCodeResponseSchema = z.object({
  apiVersion: apiVersionSchema,
  challengeId: z.string().min(1),
  expiresInSeconds: z.number().int().positive(),
  resendAfterSeconds: z.number().int().nonnegative(),
  deliveryMode: z.enum(["console", "email"]),
});

export const authenticatedUserSchema = z.object({
  id: z.string().min(1),
  email: z.string().email(),
});

export const verifyEmailCodeResponseSchema = z.object({
  apiVersion: apiVersionSchema,
  authenticated: z.literal(true),
  user: authenticatedUserSchema,
});

export const authStateSchema = z.object({
  apiVersion: apiVersionSchema,
  authenticated: z.boolean(),
  user: authenticatedUserSchema.nullable(),
});

export const exportJobSchema = z.object({
  apiVersion: apiVersionSchema,
  id: z.string().min(1),
  projectId: z.string().min(1),
  status: z.enum(["queued", "rendering", "ready", "failed"]),
  downloadUrl: downloadLocationSchema.nullable(),
  expiresAt: z.string().datetime().nullable(),
  message: z.string(),
  experiment: experimentContextSchema.nullable().default(null),
});

export const shareLinkSchema = z.object({
  apiVersion: apiVersionSchema,
  token: z.string().min(1),
  url: z.string().url(),
  downloadsEnabled: z.boolean(),
  createdAt: z.string().datetime(),
});

export const shareSummarySchema = shareLinkSchema.omit({ apiVersion: true });

export const projectWorkspaceSchema = z.object({
  apiVersion: apiVersionSchema,
  session: projectSessionSchema,
  preview: dataPreviewSchema,
  quality: qualityReportSchema,
  decisions: z.array(cleaningDecisionSchema),
  chart: chartSpecSchema,
  shares: z.array(shareSummarySchema),
});

const fitPointSchema = z.object({
  x: z.number(),
  y: z.number(),
  lower: z.number().nullable(),
  upper: z.number().nullable(),
});

const fitAnalysisSchema = z.object({
  model: z.enum(["linear", "polynomial", "exponential", "logarithmic", "power"]),
  equation: z.string(),
  rSquared: z.number(),
  sampleSize: z.number().int().nonnegative().optional(),
  excludedCount: z.number().int().nonnegative().optional(),
  fitMethod: z.enum(["ordinary-least-squares", "weighted-least-squares"]).optional(),
  confidenceMethod: z.enum(["none", "student-t", "bootstrap"]).optional(),
  intervalKind: z.enum(["none", "pointwise-mean"]).optional(),
  points: z.array(fitPointSchema),
});

const uncertaintyAnalysisSchema = z.object({
  mode: z.enum([
    "standard-deviation",
    "standard-error",
    "confidence-interval",
    "column",
  ]),
  points: z.array(
    z.object({ x: z.number(), y: z.number(), error: z.number().nonnegative() }),
  ),
});

const residualDiagnosticSchema = z.object({
  status: z.enum(["supported", "insufficient-data"]),
  sampleSize: z.number().int().nonnegative(),
  meanResidual: z.number().nullable(),
  rmse: z.number().nullable(),
  mae: z.number().nullable(),
  maxAbsResidual: z.number().nullable(),
  residualTrend: z.enum(["none", "possible-trend"]).nullable(),
  assumptions: z.array(z.string()),
  limitations: z.array(z.string()),
});

const analysisDisclosureSchema = z.object({
  method: z.enum([
    "fit",
    "confidence-band",
    "residual-diagnostic",
    "prediction-band",
    "simultaneous-band",
    "robust-fitting",
    "weighted-fitting",
    "multiple-comparison",
  ]),
  status: z.enum(["supported", "not-requested", "insufficient-data", "deferred"]),
  sampleSize: z.number().int().nonnegative(),
  excludedCount: z.number().int().nonnegative(),
  assumptions: z.array(z.string()).optional(),
  limitations: z.array(z.string()).optional(),
});

export const chartAnalysisSchema = z.object({
  apiVersion: apiVersionSchema,
  projectId: z.string().min(1),
  series: z.array(
    z.object({
      field: z.string().min(1),
      label: z.string().min(1),
      panel: z.number().int().min(1).max(4),
      group: previewValueSchema.default(null),
      points: z.array(
        z.object({
          x: previewValueSchema,
          y: z.number(),
        }),
      ),
      fit: fitAnalysisSchema.nullable(),
      uncertainty: uncertaintyAnalysisSchema.nullable(),
      residualDiagnostic: residualDiagnosticSchema.nullable().optional(),
      disclosures: z.array(analysisDisclosureSchema).optional(),
      warnings: z.array(z.string()),
    }),
  ),
  preview: z.object({
    histograms: z.array(
      z.object({
        field: z.string(),
        label: z.string(),
        bins: z.array(
          z.object({ start: z.number(), end: z.number(), count: z.number().int().nonnegative() }),
        ),
      }),
    ),
    boxes: z.array(
      z.object({
        field: z.string(),
        label: z.string(),
        minimum: z.number(),
        q1: z.number(),
        median: z.number(),
        q3: z.number(),
        maximum: z.number(),
        outliers: z.array(z.number()),
      }),
    ),
    heatmaps: z.array(
      z.object({
        panel: z.number().int().min(1).max(4),
        labels: z.array(z.string()),
        // Constant or entirely-missing columns have an undefined correlation.
        // Preserve that scientific meaning instead of coercing it to zero.
        matrix: z.array(z.array(z.number().nullable())),
      }),
    ),
    surfacePoints: z.array(
      z.object({
        panel: z.number().int().min(1).max(4),
        x: z.number(),
        y: z.number(),
        z: z.number(),
      }),
    ),
    surfaceDiagnostics: z.array(
      z.object({
        panel: z.number().int().min(1).max(4),
        xField: z.string().min(1),
        yField: z.string().min(1),
        zField: z.string().min(1),
        xCount: z.number().int().nonnegative(),
        yCount: z.number().int().nonnegative(),
        expectedPoints: z.number().int().nonnegative(),
        usablePoints: z.number().int().nonnegative(),
        duplicateCoordinatePairs: z.number().int().nonnegative(),
        duplicateCoordinateRows: z.number().int().nonnegative(),
        missingGridCells: z.number().int().nonnegative(),
        nonFinitePoints: z.number().int().nonnegative(),
        uniformXSpacing: z.boolean(),
        uniformYSpacing: z.boolean(),
        collinear: z.boolean(),
        status: z.enum([
          "valid",
          "invalid-values",
          "insufficient-points",
          "collinear",
          "duplicate-coordinates",
          "missing-grid",
          "irregular-grid",
        ]),
      }),
    ).default([]),
  }),
  recommendations: z.array(
    z.object({
      chartType: z.enum(["surface3d", "heatmap", "scatter"]),
      source: z.enum(["regular-grid", "grid-like"]),
      xField: z.string().min(1),
      yField: z.string().min(1),
      zField: z.string().min(1),
      reason: z.string().min(1),
      reasonCode: z.string().min(1),
      reasonParams: messageParamsSchema.default({}),
    }),
  ).default([]),
});

export type CleaningDecision = z.infer<typeof cleaningDecisionSchema>;
export type AuthState = z.infer<typeof authStateSchema>;
export type ChartAnalysis = z.infer<typeof chartAnalysisSchema>;
export type DataPreview = z.infer<typeof dataPreviewSchema>;
export type ExperimentContext = z.infer<typeof experimentContextSchema>;
export type PreviewColumn = z.infer<typeof previewColumnSchema>;
export type PreviewRow = z.infer<typeof previewRowSchema>;
export type ProcessingJob = z.infer<typeof processingJobSchema>;
export type ProjectList = z.infer<typeof projectListSchema>;
export type ProjectDescription = z.infer<typeof projectDescriptionResponseSchema>;
export type ProjectSession = z.infer<typeof projectSessionSchema>;
export type ProjectWorkspace = z.infer<typeof projectWorkspaceSchema>;
export type ProjectSummary = z.infer<typeof projectSummarySchema>;
export type QualityFinding = z.infer<typeof qualityFindingSchema>;
export type QualityReport = z.infer<typeof qualityReportSchema>;
export type SharedChart = z.infer<typeof sharedChartSchema>;
export type ShareLink = z.infer<typeof shareLinkSchema>;
export type ShareSummary = z.infer<typeof shareSummarySchema>;
export type ExportJob = z.infer<typeof exportJobSchema>;
