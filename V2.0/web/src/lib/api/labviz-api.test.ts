import { afterEach, describe, expect, it, vi } from "vitest";

import { labvizApi } from "./labviz-api";

const readySession = {
  apiVersion: "v1",
  projectId: "project-regression",
  storageMode: "temporary-cloud",
  description: "",
  currentRevisionId: "00000000-0000-4000-8000-000000000001",
  source: {
    name: "experiment.csv",
    size: 1_960_000,
    mediaType: "text/csv",
    sheetName: null,
    headerRow: 1,
  },
  job: {
    apiVersion: "v1",
    id: "job-regression",
    projectId: "project-regression",
    stage: "ready",
    progress: 100,
    message: "Your data is ready to inspect.",
    errorCode: null,
  },
  expiresAt: "2030-01-02T05:04:05Z",
};

function apiResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
    status,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("LabViz API client contract", () => {
  it("parses a ready processing job and keeps cookie credentials enabled", async () => {
    const fetchMock = vi.fn().mockResolvedValue(apiResponse(readySession));
    vi.stubGlobal("fetch", fetchMock);

    const session = await labvizApi.getProject("project-regression");

    expect(session.job).toMatchObject({ stage: "ready", progress: 100 });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project-regression",
      expect.objectContaining({ credentials: "include" }),
    );
  });

  it("places a 1.87 MB file in multipart form data without truncation", async () => {
    const fetchMock = vi.fn().mockResolvedValue(apiResponse(readySession));
    vi.stubGlobal("fetch", fetchMock);
    const bytes = new Uint8Array(1_960_000);
    const file = new File([bytes], "experiment.csv", { type: "text/csv" });

    await labvizApi.createProject(file);

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const uploaded = (init.body as FormData).get("file");
    expect(uploaded).toBeInstanceOf(File);
    expect((uploaded as File).size).toBe(bytes.byteLength);
    expect(init.headers).not.toHaveProperty("Content-Type");
  });

  it("sends an Excel sheet and one-based header row when re-reading a workbook", async () => {
    const fetchMock = vi.fn().mockResolvedValue(apiResponse(readySession));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File([new Uint8Array(512)], "experiment.xlsx");

    await labvizApi.createProject(file, { sheetName: "Measurements", headerRow: 3 });

    const body = fetchMock.mock.calls[0]?.[1]?.body as FormData;
    expect(body.get("sheetName")).toBe("Measurements");
    expect(body.get("headerRow")).toBe("3");
  });

  it("sends the upload idempotency key without taking over multipart headers", async () => {
    const fetchMock = vi.fn().mockResolvedValue(apiResponse(readySession));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File([new Uint8Array(512)], "experiment.csv", {
      type: "text/csv",
    });

    await labvizApi.createProject(file, { idempotencyKey: "upload-key-1" });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.headers).toMatchObject({ "Idempotency-Key": "upload-key-1" });
    expect(init.headers).not.toHaveProperty("Content-Type");
  });

  it("updates a project description with optimistic revision concurrency", async () => {
    const response = {
      apiVersion: "v1",
      projectId: "project-regression",
      description: "A plain-text project description.",
      revisionId: "00000000-0000-4000-8000-000000000002",
      revisionNumber: 2,
      updatedAt: "2030-01-02T05:04:06Z",
    };
    const fetchMock = vi.fn().mockResolvedValue(apiResponse(response));
    vi.stubGlobal("fetch", fetchMock);

    const result = await labvizApi.updateProjectDescription(
      "project-regression",
      response.description,
      readySession.currentRevisionId,
    );

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project-regression/description",
      expect.objectContaining({
        body: JSON.stringify({
          description: response.description,
          expectedRevisionId: readySession.currentRevisionId,
        }),
        credentials: "include",
        method: "PATCH",
      }),
    );
  });

  it("reports malformed successful responses as a Contract error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(apiResponse({ projectId: "missing-fields" })),
    );

    await expect(labvizApi.getProject("missing-fields")).rejects.toMatchObject({
      code: "invalid-contract-response",
      status: 200,
    });
  });
});
