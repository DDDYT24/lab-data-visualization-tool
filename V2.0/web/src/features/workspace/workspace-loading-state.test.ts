import { describe, expect, it } from "vitest";

import { getWorkspaceLoadingCopy } from "./workspace-loading-state";

describe("getWorkspaceLoadingCopy", () => {
  it("does not describe a ready job as an upload in progress", () => {
    const copy = getWorkspaceLoadingCopy(true, {
      apiVersion: "v1",
      id: "job-1",
      projectId: "project-1",
      stage: "ready",
      progress: 100,
      message: "Your data is ready to inspect.",
      errorCode: null,
    });

    expect(copy.title).toBe("Loading processed data");
    expect(copy.progress).toBe(100);
  });

  it("lets a ready job override a request that the proxy still reports as pending", () => {
    const copy = getWorkspaceLoadingCopy(true, {
      apiVersion: "v1",
      id: "job-ready-behind-proxy",
      projectId: "project-ready-behind-proxy",
      stage: "ready",
      progress: 100,
      message: "Your data is ready to inspect.",
      errorCode: null,
    });

    expect(copy.title).toBe("Loading processed data");
    expect(copy.description).toMatch(/complete/i);
    expect(copy.progress).toBe(100);
  });
});
