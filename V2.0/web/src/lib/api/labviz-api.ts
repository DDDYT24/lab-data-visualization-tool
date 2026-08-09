import { z, type ZodType } from "zod";

import {
  authStateSchema,
  chartAnalysisSchema,
  cleaningDecisionsResponseSchema,
  dataPreviewSchema,
  exportJobSchema,
  processingJobSchema,
  projectListSchema,
  projectSessionSchema,
  projectWorkspaceSchema,
  qualityReportSchema,
  requestEmailCodeResponseSchema,
  savedChartResponseSchema,
  shareLinkSchema,
  sharedChartSchema,
  verifyEmailCodeResponseSchema,
  type ExportJob,
  type CleaningDecision,
  type ProcessingJob,
  type ProjectSession,
  type SharedChart,
} from "@/domain/api-contract";
import type { ChartSpec } from "@/domain/chart-spec";

const API_BASE = (
  process.env.NEXT_PUBLIC_LABVIZ_API_URL ?? "/api/v1"
).replace(/\/$/, "");
const emptyResponseSchema = z.null();

export class LabVizApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code = "api-error",
  ) {
    super(message);
    this.name = "LabVizApiError";
  }
}

async function request<T>(
  path: string,
  schema: ZodType<T>,
  init?: RequestInit,
): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      credentials: "include",
      headers: {
        Accept: "application/json",
        ...init?.headers,
      },
    });
  } catch {
    throw new LabVizApiError(
      "The LabViz processing API could not be reached. Your source file was not changed.",
      0,
      "network-unavailable",
    );
  }

  const contentType = response.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : null;

  if (!response.ok) {
    const message =
      typeof body?.message === "string"
        ? body.message
        : "The LabViz API could not complete this request.";
    const code = typeof body?.code === "string" ? body.code : "api-error";
    throw new LabVizApiError(message, response.status, code);
  }

  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    throw new LabVizApiError(
      "The API returned data that does not match the LabViz v1 contract.",
      response.status,
      "invalid-contract-response",
    );
  }

  return parsed.data;
}

export const labvizApi = {
  cleanedDataUrl(projectId: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectId)}/exports/cleaned-data.csv`;
  },
  createProject(
    file: File,
    options?: {
      sheetName?: string | null;
      headerRow?: number | null;
      idempotencyKey?: string;
    },
    signal?: AbortSignal,
  ): Promise<ProjectSession> {
    const body = new FormData();
    body.append("file", file);
    if (options?.sheetName) body.append("sheetName", options.sheetName);
    if (options?.headerRow) body.append("headerRow", String(options.headerRow));
    return request("/projects", projectSessionSchema, {
      method: "POST",
      body,
      headers: options?.idempotencyKey
        ? { "Idempotency-Key": options.idempotencyKey }
        : undefined,
      signal,
    });
  },

  createSampleProject(signal?: AbortSignal): Promise<ProjectSession> {
    return request("/samples/thermal-response/projects", projectSessionSchema, {
      method: "POST",
      signal,
    });
  },

  getProject(projectId: string, signal?: AbortSignal): Promise<ProjectSession> {
    return request(`/projects/${encodeURIComponent(projectId)}`, projectSessionSchema, {
      signal,
    });
  },

  getProjectWorkspace(projectId: string, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/workspace`,
      projectWorkspaceSchema,
      { signal },
    );
  },

  getJob(jobId: string, signal?: AbortSignal): Promise<ProcessingJob> {
    return request(`/jobs/${encodeURIComponent(jobId)}`, processingJobSchema, {
      signal,
    });
  },

  getPreview(projectId: string, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/preview`,
      dataPreviewSchema,
      { signal },
    );
  },

  getQuality(projectId: string, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/quality`,
      qualityReportSchema,
      { signal },
    );
  },

  applyQualityRules(
    projectId: string,
    ranges: Array<{ field: string; minimum: number | null; maximum: number | null }>,
    signal?: AbortSignal,
  ) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/quality-rules`,
      qualityReportSchema,
      {
        method: "PUT",
        body: JSON.stringify({ ranges }),
        headers: { "Content-Type": "application/json" },
        signal,
      },
    );
  },

  listProjects(signal?: AbortSignal) {
    return request("/projects", projectListSchema, { signal });
  },

  getSharedChart(token: string, signal?: AbortSignal): Promise<SharedChart> {
    return request(`/shares/${encodeURIComponent(token)}`, sharedChartSchema, {
      signal,
    });
  },

  requestEmailCode(email: string, signal?: AbortSignal) {
    return request("/auth/email-code", requestEmailCodeResponseSchema, {
      method: "POST",
      body: JSON.stringify({ email }),
      headers: { "Content-Type": "application/json" },
      signal,
    });
  },

  verifyEmailCode(
    challengeId: string,
    code: string,
    signal?: AbortSignal,
  ) {
    return request("/auth/email-code/verify", verifyEmailCodeResponseSchema, {
      method: "POST",
      body: JSON.stringify({ challengeId, code }),
      headers: { "Content-Type": "application/json" },
      signal,
    });
  },

  getAuthState(signal?: AbortSignal) {
    return request("/auth/me", authStateSchema, { signal });
  },

  logout(signal?: AbortSignal) {
    return request("/auth/logout", authStateSchema, {
      method: "POST",
      signal,
    });
  },

  saveProject(projectId: string, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/save`,
      projectSessionSchema,
      { method: "POST", signal },
    );
  },

  duplicateProject(projectId: string, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/duplicate`,
      projectSessionSchema,
      { method: "POST", signal },
    );
  },

  deleteProject(projectId: string, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}`,
      emptyResponseSchema,
      { method: "DELETE", signal },
    );
  },

  analyzeChart(projectId: string, chart: ChartSpec, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/chart-analysis`,
      chartAnalysisSchema,
      {
        method: "POST",
        body: JSON.stringify({ chart }),
        headers: { "Content-Type": "application/json" },
        signal,
      },
    );
  },

  requestExport(
    projectId: string,
    chart: ChartSpec,
    signal?: AbortSignal,
  ): Promise<ExportJob> {
    return request(
      `/projects/${encodeURIComponent(projectId)}/exports`,
      exportJobSchema,
      {
        method: "POST",
        body: JSON.stringify({ chart }),
        headers: { "Content-Type": "application/json" },
        signal,
      },
    );
  },

  saveCleaningDecisions(
    projectId: string,
    decisions: CleaningDecision[],
    signal?: AbortSignal,
  ) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/cleaning-decisions`,
      cleaningDecisionsResponseSchema,
      {
        method: "PATCH",
        body: JSON.stringify({ decisions }),
        headers: { "Content-Type": "application/json" },
        signal,
      },
    );
  },

  saveChart(projectId: string, chart: ChartSpec, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/chart`,
      savedChartResponseSchema,
      {
        method: "PUT",
        body: JSON.stringify({ chart }),
        headers: { "Content-Type": "application/json" },
        signal,
      },
    );
  },

  createShareLink(
    projectId: string,
    downloadsEnabled: boolean,
    signal?: AbortSignal,
  ) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/shares`,
      shareLinkSchema,
      {
        method: "POST",
        body: JSON.stringify({ downloadsEnabled }),
        headers: { "Content-Type": "application/json" },
        signal,
      },
    );
  },

  updateShareLink(
    projectId: string,
    token: string,
    downloadsEnabled: boolean,
    signal?: AbortSignal,
  ) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/shares/${encodeURIComponent(token)}`,
      shareLinkSchema,
      {
        method: "PATCH",
        body: JSON.stringify({ downloadsEnabled }),
        headers: { "Content-Type": "application/json" },
        signal,
      },
    );
  },

  revokeShareLink(projectId: string, token: string, signal?: AbortSignal) {
    return request(
      `/projects/${encodeURIComponent(projectId)}/shares/${encodeURIComponent(token)}`,
      emptyResponseSchema,
      { method: "DELETE", signal },
    );
  },
};
