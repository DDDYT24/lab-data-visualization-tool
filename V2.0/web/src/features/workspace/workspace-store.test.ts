import { afterEach, describe, expect, it } from "vitest";
import type { ProjectSession, DataPreview, QualityReport } from "@/domain/api-contract";

import { useWorkspaceStore } from "./workspace-store";

describe("workspace store", () => {
  afterEach(() => {
    useWorkspaceStore.getState().reset();
  });

  it("ignores a previous project's late session response after opening another project", () => {
    useWorkspaceStore.setState({projectId:"next-project",currentRevisionId:"next-revision",description:"Current description"});
    const previous = {projectId:"previous-project",currentRevisionId:"previous-revision",description:"Late old description",storageMode:"local",job:null} as ProjectSession;
    useWorkspaceStore.getState().markProjectSaved(previous);
    expect(useWorkspaceStore.getState().projectId).toBe("next-project");
    expect(useWorkspaceStore.getState().currentRevisionId).toBe("next-revision");
    expect(useWorkspaceStore.getState().description).toBe("Current description");
  });

  it("keeps the next project loading when a previous preview and quality response arrives", () => {
    useWorkspaceStore.getState().openProject("next-project");
    const preview = {apiVersion:"v1",projectId:"previous-project",columns:[],rows:[],totalRows:0,sampled:false,sampleStrategy:"none"} as DataPreview;
    const quality = {apiVersion:"v1",projectId:"previous-project",totalRows:0,validRows:0,missingValues:0,duplicateRows:0,suspiciousPoints:0,findings:[]} as QualityReport;
    useWorkspaceStore.getState().setWorkspaceData(preview, quality);
    expect(useWorkspaceStore.getState().preview).toBeNull();
    expect(useWorkspaceStore.getState().quality).toBeNull();
    expect(useWorkspaceStore.getState().loadStatus).toBe("processing");
  });

  it("still accepts a session update belonging to the currently selected project", () => {
    useWorkspaceStore.getState().openProject("current-project");
    const session = {projectId:"current-project",currentRevisionId:"updated-revision",description:"Saved description",storageMode:"local",job:null} as ProjectSession;
    useWorkspaceStore.getState().markProjectSaved(session);
    expect(useWorkspaceStore.getState().currentRevisionId).toBe("updated-revision");
    expect(useWorkspaceStore.getState().description).toBe("Saved description");
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
  it("opens a single numeric research column as a valid histogram, not X against itself", () => {
    useWorkspaceStore.getState().setWorkspaceData({
      apiVersion: "v1", projectId: "one-column",
      columns: [{ field: "value", label: "Value", kind: "number", unit: "mV", nullable: false }],
      rows: [{ rowId: 1, value: 2 }], totalRows: 1, sampled: false, sampleStrategy: "none",
    }, {
      apiVersion: "v1", projectId: "one-column", totalRows: 1, validRows: 1,
      missingValues: 0, duplicateRows: 0, suspiciousPoints: 0, findings: [],
    });
    expect(useWorkspaceStore.getState().chartSpec.type).toBe("histogram");
    expect(useWorkspaceStore.getState().chartSpec.series[0].field).toBe("value");
  });

});
