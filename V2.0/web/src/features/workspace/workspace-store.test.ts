import { afterEach, describe, expect, it } from "vitest";

import { useWorkspaceStore } from "./workspace-store";

describe("workspace store", () => {
  afterEach(() => {
    useWorkspaceStore.getState().reset();
  });

  it("starts a new file at the import step", () => {
    useWorkspaceStore.getState().selectFile({
      name: "experiment.xlsx",
      size: 2048,
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      isSample: false,
    });

    const state = useWorkspaceStore.getState();
    expect(state.currentStep).toBe("import");
    expect(state.selectedFile?.name).toBe("experiment.xlsx");
  });

  it("keeps data-quality decisions explicit", () => {
    useWorkspaceStore.getState().setIssueAction("row-38", "exclude");
    expect(useWorkspaceStore.getState().issueActions).toEqual({
      "row-38": "exclude",
    });
  });

  it("supports undo, redo, and restoring the original decision set", () => {
    const store = useWorkspaceStore.getState();
    store.setIssueAction("finding-1", "exclude");
    store.setIssueAction("finding-2", "remove");

    useWorkspaceStore.getState().undoIssueActions();
    expect(useWorkspaceStore.getState().issueActions).toEqual({
      "finding-1": "exclude",
    });

    useWorkspaceStore.getState().redoIssueActions();
    expect(useWorkspaceStore.getState().issueActions).toEqual({
      "finding-1": "exclude",
      "finding-2": "remove",
    });

    useWorkspaceStore.getState().restoreIssueActions();
    expect(useWorkspaceStore.getState().issueActions).toEqual({});
  });

  it("updates export settings without changing the rest of the chart", () => {
    useWorkspaceStore.getState().updateExport({ dpi: 600 });
    const state = useWorkspaceStore.getState();
    expect(state.chartSpec.export.dpi).toBe(600);
    expect(state.chartSpec.type).toBe("line");
    expect(state.exportReady).toBe(false);
  });

  it("hydrates preview and chart fields from the API contract", () => {
    useWorkspaceStore.getState().setWorkspaceData(
      {
        apiVersion: "v1",
        projectId: "project-1",
        columns: [
          {
            field: "elapsed",
            label: "Elapsed time",
            kind: "number",
            unit: "s",
            nullable: false,
          },
          {
            field: "voltage",
            label: "Voltage",
            kind: "number",
            unit: "mV",
            nullable: false,
          },
        ],
        rows: [{ rowId: 1, elapsed: 0, voltage: 2.4 }],
        totalRows: 1,
        sampled: false,
        sampleStrategy: "none",
      },
      {
        apiVersion: "v1",
        projectId: "project-1",
        totalRows: 1,
        validRows: 1,
        missingValues: 0,
        duplicateRows: 0,
        suspiciousPoints: 0,
        findings: [],
      },
    );

    const state = useWorkspaceStore.getState();
    expect(state.loadStatus).toBe("ready");
    expect(state.chartSpec.xAxis.field).toBe("elapsed");
    expect(state.chartSpec.series[0].field).toBe("voltage");
  });
});
