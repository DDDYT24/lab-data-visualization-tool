import { describe, expect, it } from "vitest";

import {
  chartAnalysisSchema,
  dataPreviewSchema,
  exportJobSchema,
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
        surfacePoints: [],
      },
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
});
