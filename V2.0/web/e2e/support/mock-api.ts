import type { Page, Route } from "@playwright/test";

const UPDATED_AT = "2030-01-02T03:04:05Z";
const PROJECT_ID = "project-e2e";
const JOB_ID = "job-e2e";
const REVISION_1 = "00000000-0000-4000-8000-000000000001";
const REVISION_2 = "00000000-0000-4000-8000-000000000002";
const PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+XfW2AAAAAElFTkSuQmCC";
const SVG_DATA_URL =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'/%3E";

const chart = {
  schemaVersion: 1,
  type: "line",
  title: "Thermal response",
  xAxis: { field: "time", title: "Time", unit: "min" },
  yAxis: { field: "response", title: "Response", unit: "mV" },
  series: [
    {
      field: "response",
      label: "Sample A",
      color: "#2563EB",
    },
  ],
  panelCount: 1,
  export: {
    format: "png",
    dpi: 300,
    sizePreset: "double-column",
    grayscalePreview: false,
  },
};

const preview = {
  apiVersion: "v1",
  projectId: PROJECT_ID,
  columns: [
    {
      field: "time",
      label: "Time",
      kind: "number",
      unit: "min",
      nullable: false,
    },
    {
      field: "response",
      label: "Response",
      kind: "number",
      unit: "mV",
      nullable: true,
    },
  ],
  rows: [
    { rowId: 1, time: 0, response: 1.2 },
    { rowId: 2, time: 1, response: 2.1 },
    { rowId: 3, time: 2, response: 5.8 },
    { rowId: 4, time: 3, response: 4.4 },
  ],
  totalRows: 100_000,
  sampled: true,
  sampleStrategy: "evenly-distributed",
};

const quality = {
  apiVersion: "v1",
  projectId: PROJECT_ID,
  totalRows: 100_000,
  validRows: 99_998,
  missingValues: 1,
  duplicateRows: 0,
  suspiciousPoints: 1,
  findings: [
    {
      id: "finding-e2e",
      kind: "extreme-value",
      severity: "warning",
      column: "response",
      rowIds: [3],
      summary: "Possible extreme value",
      reason: "The value is outside the interquartile-range threshold.",
      affectedCount: 1,
      rowIdsTruncated: false,
    },
    {
      id: "finding-missing-e2e",
      kind: "missing",
      severity: "warning",
      column: "response",
      rowIds: [4],
      summary: "Missing measurement",
      reason: "This row has no response value.",
      affectedCount: 1,
      rowIdsTruncated: false,
    },
  ],
};

export type MockApiObservations = {
  analysisModels: string[];
  cleaningActions: string[];
  deletedProjects: string[];
  descriptionUpdates: string[];
  duplicatedProjects: string[];
  exportRequests: number;
  exportFormats: string[];
  requests: string[];
  requestedEmail: string | null;
  uploadedBytes: number[];
  verifiedCode: string | null;
};

type MockApiOptions = {
  dataDelayMs?: number;
  deliveryMode?: "console" | "email";
  history?: "empty" | "saved";
  shareExpired?: boolean;
  authenticated?: boolean;
  descriptionFailure?: "conflict" | "unauthorized" | "deleted" | "server";
  descriptionLoading?: boolean;
};

function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    body: JSON.stringify(body),
    contentType: "application/json",
    status,
  });
}

function processingJob(stage: "queued" | "ready") {
  return {
    apiVersion: "v1",
    id: JOB_ID,
    projectId: PROJECT_ID,
    stage,
    progress: stage === "ready" ? 100 : 5,
    message: stage === "ready" ? "Your data is ready to inspect." : "Upload accepted.",
    errorCode: null,
  };
}

export async function installMockApi(
  page: Page,
  options: MockApiOptions = {},
): Promise<MockApiObservations> {
  const observations: MockApiObservations = {
    analysisModels: [],
    cleaningActions: [],
    deletedProjects: [],
    descriptionUpdates: [],
    duplicatedProjects: [],
    exportRequests: 0,
    exportFormats: [],
    requests: [],
    requestedEmail: null,
    uploadedBytes: [],
    verifiedCode: null,
  };
  let historyProjectVisible = options.history !== "empty";
  let projectDescription = "";
  let currentRevisionId = REVISION_1;

  await page.context().route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace(/^\/api\/v1/, "");
    const method = request.method();
    observations.requests.push(`${method} ${path}`);

    if (method === "POST" && path === "/projects") {
      observations.uploadedBytes.push(request.postDataBuffer()?.byteLength ?? 0);
      return json(route, {
        apiVersion: "v1",
        projectId: PROJECT_ID,
        storageMode: "temporary-cloud",
        description: "",
        currentRevisionId: null,
        source: {
          name: "experiment-1.87mb.csv",
          size: 1_960_000,
          mediaType: "text/csv",
          sheetName: null,
          headerRow: 1,
        },
        job: processingJob("queued"),
        expiresAt: "2030-01-02T05:04:05Z",
      });
    }

    if (method === "POST" && path === "/samples/thermal-response/projects") {
      return json(route, {
        apiVersion: "v1",
        projectId: PROJECT_ID,
        storageMode: "temporary-cloud",
        description: "",
        currentRevisionId: null,
        source: {
          name: "labviz-sample.csv",
          size: 18_432,
          mediaType: "text/csv",
          sheetName: null,
          headerRow: 1,
        },
        job: processingJob("queued"),
        expiresAt: "2030-01-02T05:04:05Z",
      });
    }

    if (method === "GET" && path === `/jobs/${JOB_ID}`) {
      return json(route, processingJob("ready"));
    }

    if (method === "GET" && path === `/projects/${PROJECT_ID}/workspace`) {
      return json(route, {
        apiVersion: "v1",
        session: {
          apiVersion: "v1",
          projectId: PROJECT_ID,
          storageMode: "saved-cloud",
          description: projectDescription,
          currentRevisionId: options.descriptionLoading ? null : currentRevisionId,
          source: {
            name: "experiment-1.87mb.csv",
            size: 1_960_000,
            mediaType: "text/csv",
            sheetName: null,
            availableSheets: [],
            headerRow: 1,
          },
          job: processingJob("ready"),
          expiresAt: null,
        },
        preview,
        quality,
        decisions: [
          { findingId: "finding-e2e", action: "ignore" },
          { findingId: "finding-missing-e2e", action: "exclude" },
        ],
        chart,
        shares: [],
      });
    }

    if (method === "GET" && path === `/projects/${PROJECT_ID}/preview`) {
      if (options.dataDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.dataDelayMs));
      }
      return json(route, preview);
    }

    if (method === "GET" && path === `/projects/${PROJECT_ID}/quality`) {
      if (options.dataDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.dataDelayMs));
      }
      return json(route, quality);
    }

    if (method === "GET" && path === `/projects/${PROJECT_ID}`) {
      return json(route, {
        apiVersion: "v1",
        projectId: PROJECT_ID,
        storageMode: "saved-cloud",
        description: projectDescription,
        currentRevisionId: options.descriptionLoading ? null : currentRevisionId,
        source: {
          name: "experiment-1.87mb.csv",
          size: 1_960_000,
          mediaType: "text/csv",
          sheetName: null,
          availableSheets: [],
          headerRow: 1,
        },
        job: processingJob("ready"),
        expiresAt: null,
      });
    }

    if (method === "PATCH" && path === `/projects/${PROJECT_ID}/description`) {
      if (options.descriptionFailure === "conflict") {
        return json(
          route,
          {
            code: "project-revision-conflict",
            message: "The project description changed in another session.",
          },
          409,
        );
      }
      if (options.descriptionFailure === "unauthorized") {
        return json(
          route,
          { code: "project-access-denied", message: "Access denied." },
          403,
        );
      }
      if (options.descriptionFailure === "deleted") {
        return json(
          route,
          { code: "project-not-found", message: "Project not found." },
          404,
        );
      }
      if (options.descriptionFailure === "server") {
        return json(
          route,
          { code: "description-unavailable", message: "Temporary failure." },
          500,
        );
      }
      const body = request.postDataJSON() as {
        description: string;
        expectedRevisionId: string;
      };
      if (body.expectedRevisionId !== currentRevisionId) {
        return json(
          route,
          {
            code: "project-revision-conflict",
            message: "The project description changed in another session.",
          },
          409,
        );
      }
      projectDescription = body.description;
      currentRevisionId = REVISION_2;
      observations.descriptionUpdates.push(body.description);
      return json(route, {
        apiVersion: "v1",
        projectId: PROJECT_ID,
        description: projectDescription,
        revisionId: currentRevisionId,
        revisionNumber: 2,
        updatedAt: UPDATED_AT,
      });
    }

    if (
      method === "PATCH" &&
      path === `/projects/${PROJECT_ID}/cleaning-decisions`
    ) {
      const body = request.postDataJSON() as {
        decisions: Array<{ action: string; findingId: string }>;
      };
      observations.cleaningActions.splice(
        0,
        observations.cleaningActions.length,
        ...body.decisions.map((decision) => decision.action),
      );
      return json(route, {
        apiVersion: "v1",
        projectId: PROJECT_ID,
        decisions: body.decisions,
        updatedAt: UPDATED_AT,
      });
    }

    if (method === "PUT" && path === `/projects/${PROJECT_ID}/chart`) {
      const body = request.postDataJSON() as { chart: unknown };
      return json(route, {
        apiVersion: "v1",
        projectId: PROJECT_ID,
        chart: body.chart,
        updatedAt: UPDATED_AT,
      });
    }

    if (
      method === "POST" &&
      path === `/projects/${PROJECT_ID}/chart-analysis`
    ) {
      const body = request.postDataJSON() as {
        chart: { fitting?: { model?: string } };
      };
      const model = body.chart.fitting?.model ?? "none";
      observations.analysisModels.push(model);
      return json(route, {
        apiVersion: "v1",
        projectId: PROJECT_ID,
        series: [
          {
            field: "response",
            label: "Sample A",
            panel: 1,
            points: [
              { x: 0, y: 1 },
              { x: 1, y: 2.2 },
              { x: 2, y: 3.1 },
              { x: 3, y: 4.3 },
            ],
            fit:
              model === "none"
                ? null
                : {
                    model,
                    equation: "y = 1.1x + 1.0",
                    rSquared: 0.98,
                    points: [
                      { x: 0, y: 1, lower: 0.8, upper: 1.2 },
                      { x: 3, y: 4.3, lower: 4.1, upper: 4.5 },
                    ],
                  },
            uncertainty: null,
            warnings: [],
          },
        ],
        preview: {
          histograms: [],
          boxes: [],
          heatmaps: [],
          surfacePoints: [],
        },
      });
    }

    if (method === "POST" && path === `/projects/${PROJECT_ID}/exports`) {
      const body = request.postDataJSON() as {
        chart: { export: { format: "png" | "svg" | "pdf" } };
      };
      const format = body.chart.export.format;
      observations.exportRequests += 1;
      observations.exportFormats.push(format);
      return json(route, {
        apiVersion: "v1",
        id: "export-e2e",
        projectId: PROJECT_ID,
        status: "ready",
        downloadUrl: format === "svg" ? SVG_DATA_URL : PNG_DATA_URL,
        expiresAt: "2030-01-02T05:04:05Z",
        message: `${format.toUpperCase()} export is ready.`,
      });
    }

    if (method === "GET" && path === "/exports/export-e2e/download") {
      return route.fulfill({
        body: Buffer.from("deterministic-e2e-export"),
        contentType: "image/png",
        headers: {
          "Content-Disposition": 'attachment; filename="thermal-response.png"',
        },
      });
    }

    if (method === "POST" && path === "/auth/email-code") {
      const body = request.postDataJSON() as { email: string };
      observations.requestedEmail = body.email;
      return json(route, {
        apiVersion: "v1",
        challengeId: "challenge-e2e",
        expiresInSeconds: 600,
        resendAfterSeconds: 0,
        deliveryMode: options.deliveryMode ?? "console",
      });
    }

    if (method === "POST" && path === "/auth/email-code/verify") {
      const body = request.postDataJSON() as { code: string };
      observations.verifiedCode = body.code;
      if (body.code !== "123456") {
        return json(
          route,
          { code: "invalid-code", message: "The verification code is invalid." },
          400,
        );
      }
      return json(route, {
        apiVersion: "v1",
        authenticated: true,
        user: { id: "user-e2e", email: "researcher@example.com" },
      });
    }

    if (method === "GET" && path === "/auth/me") {
      return json(route, {
        apiVersion: "v1",
        authenticated: options.authenticated ?? false,
        user: options.authenticated
          ? { id: "user-e2e", email: "researcher@example.com" }
          : null,
      });
    }

    if (method === "GET" && path === "/projects") {
      const projects =
        !historyProjectVisible
          ? []
          : [
              {
                id: PROJECT_ID,
                title: "Thermal response",
                sourceName: "experiment.csv",
                chartType: "line",
                updatedAt: UPDATED_AT,
                storageMode: "saved-cloud",
                thumbnailUrl: null,
              },
            ];
      return json(route, { apiVersion: "v1", projects });
    }

    if (method === "POST" && path === `/projects/${PROJECT_ID}/duplicate`) {
      observations.duplicatedProjects.push(PROJECT_ID);
      return json(route, {
        apiVersion: "v1",
        projectId: "project-copy-e2e",
        storageMode: "saved-cloud",
        description: projectDescription,
        currentRevisionId,
        source: {
          name: "experiment.csv",
          size: 18_432,
          mediaType: "text/csv",
          sheetName: null,
          headerRow: 1,
        },
        job: processingJob("ready"),
        expiresAt: null,
      });
    }

    if (method === "DELETE" && path === `/projects/${PROJECT_ID}`) {
      observations.deletedProjects.push(PROJECT_ID);
      historyProjectVisible = false;
      return route.fulfill({ status: 204 });
    }

    if (method === "GET" && path === "/shares/share-e2e") {
      if (options.shareExpired) {
        return json(
          route,
          { code: "share-expired", message: "This shared link has expired." },
          410,
        );
      }
      return json(route, {
        apiVersion: "v1",
        token: "share-e2e",
        title: "Thermal response",
        description: "A deterministic read-only E2E figure.",
        updatedAt: UPDATED_AT,
        chart,
        preview,
        downloads: {
          png: PNG_DATA_URL,
          svg: null,
          pdf: null,
        },
      });
    }

    return json(
      route,
      { code: "unmocked-endpoint", message: `${method} ${path} was not mocked.` },
      501,
    );
  });

  return observations;
}

export function createCsvNearSize(targetBytes = 1_960_000) {
  const rows = ["time,response,error\n"];
  let bytes = Buffer.byteLength(rows[0]);
  let index = 0;

  while (bytes < targetBytes) {
    const row = `${index},${(index / 10).toFixed(1)},0.2\n`;
    rows.push(row);
    bytes += Buffer.byteLength(row);
    index += 1;
  }

  return Buffer.from(rows.join(""));
}
