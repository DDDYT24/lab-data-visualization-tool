import { describe, expect, it } from "vitest";
import type { FigureBoard } from "@/domain/research-figure-board";
import { figureFilesReady, figureWorkspaceState } from "./figure-workspace-state";

const ready: FigureBoard = { id: "saved-1", createdAt: "2026-10-15", composed: true, artifactState: "ready",
  spec: { schemaVersion: 1, title: "Saved", panelChartIds: ["chart-1"], axisMode: "independent" }, panels: [] };
const base = { saved: ready, preview: ready, loading: false, saving: false, saveFailed: false, dirty: false };

describe("saved revision and download boundaries", () => {
  it("keeps saved files available during draft edits and save failure", () => {
    const state = figureWorkspaceState({ ...base, dirty: true, saveFailed: true });
    expect(state.layout).toBe("saveFailed");
    expect(state.downloadsReady).toBe(true);
    expect(state.savedId).toBe("saved-1");
  });
  it.each(["queued", "running", "failed", "unavailable"] as const)("does not treat the previous preview as %s files", (artifactState) => {
    const state = figureWorkspaceState({ ...base, saved: { ...ready, id: "saved-2", composed: false, artifactState } });
    expect(state.layout).toBe("saved");
    expect(state.generation).toBe(artifactState);
    expect(state.previewId).toBe("saved-1");
    expect(state.downloadsReady).toBe(false);
  });
  it("keeps the committed revision while retrying generation", () => {
    const state = figureWorkspaceState({ ...base, saved: { ...ready, composed: false, artifactState: "failed" }, retrying: true });
    expect(state.layout).toBe("saved");
    expect(state.generation).toBe("retrying");
    expect(state.downloadsReady).toBe(false);
    expect(state.savedId).toBe(ready.id);
  });
  it("requires both complete composition and ready artifacts", () => {
    expect(figureFilesReady({ ...ready, composed: false })).toBe(false);
    expect(figureWorkspaceState({ ...base, saved: { ...ready, composed: false } }).generation).toBe("unavailable");
    expect(figureFilesReady({ ...ready, artifactState: "failed" })).toBe(false);
    expect(figureFilesReady({ ...ready, artifactState: undefined })).toBe(true);
  });
});
