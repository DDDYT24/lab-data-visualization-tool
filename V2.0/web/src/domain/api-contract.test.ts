import { describe, expect, it } from "vitest";

import {
  chartAnalysisSchema,
  dataPreviewSchema,
  exportJobSchema,
  experimentContextSchema,
  projectDescriptionTextSchema,
  processingJobSchema,
  qualityReportSchema,
} from "./api-contract";

describe("LabViz API v1 contract", () => {
  it("accepts a bounded preview with typed columns", () => {
    const result = dataPreviewSchema.safeParse({
      apiVersion: "v1",
      projectId: "project-1",
      columns: [
        {
          field: "time",
          label: "Time",
          kind: "number",
          unit: "min",
          nullable: false,
        },
      ],
      rows: [{ rowId: 1, time: 0 }],
      totalRows: 2_400,
      sampled: true,
      sampleStrategy: "evenly-distributed",
    });

    expect(result.success).toBe(true);
  });

  it("rejects progress outside the 0 to 100 range", () => {
    const result = processingJobSchema.safeParse({
      apiVersion: "v1",
      id: "job-1",
      projectId: "project-1",
      stage: "parsing",
      progress: 120,
      message: "Reading workbook",
      errorCode: null,
    });

    expect(result.success).toBe(false);
  });

  it("requires findings to explain why a value was flagged", () => {
    const result = qualityReportSchema.safeParse({
      apiVersion: "v1",
      projectId: "project-1",
      totalRows: 100,
      validRows: 99,
      missingValues: 0,
      duplicateRows: 0,
      suspiciousPoints: 1,
      findings: [
        {
          id: "finding-1",
          kind: "extreme-value",
          severity: "warning",
          column: "response",
          rowIds: [38],
          summary: "Review this point",
          reason: "",
        },
      ],
    });

    expect(result.success).toBe(false);
  });

  it("accepts locale-independent finding message codes and parameters", () => {
    const result = qualityReportSchema.safeParse({
      apiVersion: "v1",
      projectId: "project-1",
      totalRows: 10,
      validRows: 9,
      missingValues: 1,
      duplicateRows: 0,
      suspiciousPoints: 0,
      findings: [
        {
          id: "missing:response",
          kind: "missing",
          severity: "warning",
          column: "response",
          rowIds: [4],
          affectedCount: 1,
          rowIdsTruncated: false,
          summary: "1 missing value",
          reason: "These cells are empty.",
          summaryCode: "quality.missing.summary",
          summaryParams: { count: 1 },
          reasonCode: "quality.missing.reason",
          reasonParams: {},
        },
      ],
    });

    expect(result.success).toBe(true);
  });

  it("preserves undefined correlations for constant columns", () => {
    const result = chartAnalysisSchema.safeParse({
      apiVersion: "v1",
      projectId: "project-1",
      series: [],
      preview: {
        histograms: [],
        boxes: [],
        heatmaps: [
          {
            panel: 1,
            labels: ["Constant", "Response"],
            matrix: [[null, null], [null, 1]],
          },
        ],
        surfaceDiagnostics: [],
        surfacePoints: [],
      },
      recommendations: [
        {
          chartType: "surface3d",
          source: "regular-grid",
          xField: "x",
          yField: "y",
          zField: "z",
          reason: "Detected a complete grid.",
          reasonCode: "chart.recommendation.surface-grid-line",
          reasonParams: { xCount: 21, yCount: 21, pointCount: 441 },
        },
      ],
    });

    expect(result.success).toBe(true);
  });

  it("accepts a same-origin export download path", () => {
    const result = exportJobSchema.safeParse({
      apiVersion: "v1",
      id: "export-1",
      projectId: "project-1",
      status: "ready",
      downloadUrl: "/api/v1/exports/export-1/download",
      expiresAt: "2026-07-29T12:00:00.000Z",
      message: "Export ready.",
    });

    expect(result.success).toBe(true);
  });

  it("validates experiment and physical-run provenance independently", () => {
    const context = {
      experimentId: "experiment-1",
      experimentRunId: "physical-run-1",
      title: "Dose response study",
      runLabel: "Acquisition 1",
      replicateId: "R1",
      batchId: "B1",
    };

    expect(experimentContextSchema.safeParse(context).success).toBe(true);
    expect(
      experimentContextSchema.safeParse({ ...context, runLabel: "" }).success,
    ).toBe(false);
  });

  it("enforces the project-description UTF-8 byte and NUL contract", () => {
    expect(
      projectDescriptionTextSchema.safeParse(`${"a".repeat(3_997)}研`).success,
    ).toBe(true);
    expect(projectDescriptionTextSchema.safeParse("研".repeat(1_334)).success).toBe(
      false,
    );
    expect(projectDescriptionTextSchema.safeParse("invalid\0text").success).toBe(
      false,
    );
  });
});
