import { create } from "zustand";

import type {
  DataPreview,
  ProcessingJob,
  ProjectSession,
  ProjectWorkspace,
  QualityReport,
  ShareSummary,
} from "@/domain/api-contract";
import { defaultChartSpec, type ChartSpec } from "@/domain/chart-spec";
import { chartWithUserPreferences } from "@/features/settings/user-preferences";

export const workflowSteps = ["import", "inspect", "chart", "export"] as const;
export type WorkflowStep = (typeof workflowSteps)[number];
export type IssueAction = "ignore" | "exclude" | "remove";

export type SelectedFile = {
  name: string;
  size: number;
  type: string;
  isSample: boolean;
  sourceFile?: File;
  sheetName?: string | null;
  availableSheets?: string[];
  headerRow?: number | null;
  experimentTitle?: string | null;
  runLabel?: string | null;
  replicateId?: string | null;
  batchId?: string | null;
  experimentRunId?: string | null;
};

export type WorkspaceLoadStatus =
  | "idle"
  | "uploading"
  | "processing"
  | "ready"
  | "error";

type WorkspaceState = {
  selectedFile: SelectedFile | null;
  projectId: string | null;
  storageMode: ProjectSession["storageMode"] | null;
  description: string;
  currentRevisionId: string | null;
  job: ProcessingJob | null;
  preview: DataPreview | null;
  quality: QualityReport | null;
  loadStatus: WorkspaceLoadStatus;
  loadError: string | null;
  currentStep: WorkflowStep;
  issueActions: Record<string, IssueAction>;
  issueActionPast: Record<string, IssueAction>[];
  issueActionFuture: Record<string, IssueAction>[];
  shares: ShareSummary[];
  chartSpec: ChartSpec;
  exportReady: boolean;
  selectFile: (file: SelectedFile) => void;
  openProject: (projectId: string) => void;
  setSession: (session: ProjectSession) => void;
  hydrateWorkspace: (workspace: ProjectWorkspace) => void;
  markProjectSaved: (session: ProjectSession) => void;
  applyProjectDescription: (description: string, revisionId: string) => void;
  addShare: (share: ShareSummary) => void;
  removeShare: (token: string) => void;
  setJob: (job: ProcessingJob) => void;
  setWorkspaceData: (preview: DataPreview, quality: QualityReport) => void;
  replaceQuality: (quality: QualityReport) => void;
  setLoadError: (message: string) => void;
  goToStep: (step: WorkflowStep) => void;
  setIssueAction: (issueId: string, action: IssueAction) => void;
  setIssueActions: (actions: Record<string, IssueAction>) => void;
  hydrateIssueActions: (actions: Record<string, IssueAction>) => void;
  undoIssueActions: () => void;
  redoIssueActions: () => void;
  restoreIssueActions: () => void;
  updateChart: (patch: Partial<ChartSpec>) => void;
  updateExport: (patch: Partial<ChartSpec["export"]>) => void;
  prepareExport: () => void;
  reset: () => void;
};

const initialState = {
  selectedFile: null,
  projectId: null,
  storageMode: null,
  description: "",
  currentRevisionId: null,
  job: null,
  preview: null,
  quality: null,
  loadStatus: "idle" as const,
  loadError: null,
  currentStep: "import" as const,
  issueActions: {},
  issueActionPast: [],
  issueActionFuture: [],
  shares: [],
  chartSpec: defaultChartSpec,
  exportReady: false,
};

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  ...initialState,
  selectFile: (selectedFile) =>
    set({
      selectedFile,
      projectId: null,
      storageMode: null,
      description: "",
      currentRevisionId: null,
      job: null,
      preview: null,
      quality: null,
      loadStatus: "idle",
      loadError: null,
      currentStep: "import",
      issueActions: {},
      issueActionPast: [],
      issueActionFuture: [],
      shares: [],
      chartSpec: chartWithUserPreferences(defaultChartSpec),
      exportReady: false,
    }),
  openProject: (projectId) =>
    set({
      projectId,
      storageMode: null,
      description: "",
      currentRevisionId: null,
      job: null,
      preview: null,
      quality: null,
      loadStatus: "processing",
      loadError: null,
      currentStep: "import",
      issueActions: {},
      issueActionPast: [],
      issueActionFuture: [],
      shares: [],
      exportReady: false,
    }),
  setSession: (session) =>
    set((state) => {
      const failed = session.job?.stage === "failed";
      return {
        projectId: session.projectId,
        storageMode: session.storageMode,
        description: session.description,
        currentRevisionId: session.currentRevisionId,
        job: session.job,
        selectedFile: {
          ...state.selectedFile,
          name: session.source.name,
          size: session.source.size,
          type: session.source.mediaType,
          isSample: state.selectedFile?.isSample ?? false,
          sheetName: session.source.sheetName,
          availableSheets: session.source.availableSheets,
          headerRow: session.source.headerRow,
          experimentTitle: session.experiment?.title,
          runLabel: session.experiment?.runLabel,
          replicateId: session.experiment?.replicateId,
          batchId: session.experiment?.batchId,
          experimentRunId: session.experiment?.experimentRunId,
        },
        loadStatus: failed ? "error" : "processing",
        loadError: failed ? session.job?.message ?? "Processing failed." : null,
      };
    }),
  hydrateWorkspace: (workspace) =>
    set({
      projectId: workspace.session.projectId,
      storageMode: workspace.session.storageMode,
      description: workspace.session.description,
      currentRevisionId: workspace.session.currentRevisionId,
      job: workspace.session.job,
      selectedFile: {
        name: workspace.session.source.name,
        size: workspace.session.source.size,
        type: workspace.session.source.mediaType,
        isSample: false,
        sheetName: workspace.session.source.sheetName,
        availableSheets: workspace.session.source.availableSheets,
        headerRow: workspace.session.source.headerRow,
        experimentTitle: workspace.session.experiment?.title,
        runLabel: workspace.session.experiment?.runLabel,
        replicateId: workspace.session.experiment?.replicateId,
        batchId: workspace.session.experiment?.batchId,
        experimentRunId: workspace.session.experiment?.experimentRunId,
      },
      preview: workspace.preview,
      quality: workspace.quality,
      chartSpec: workspace.chart,
      issueActions: Object.fromEntries(
        workspace.decisions.map((decision) => [
          decision.findingId,
          decision.action,
        ]),
      ),
      issueActionPast: [],
      issueActionFuture: [],
      shares: workspace.shares,
      loadStatus: "ready",
      loadError: null,
      currentStep: "chart",
      exportReady: false,
    }),
  markProjectSaved: (session) =>
    set({
      storageMode: session.storageMode,
      description: session.description,
      currentRevisionId: session.currentRevisionId,
      job: session.job,
      loadError: null,
    }),
  applyProjectDescription: (description, currentRevisionId) =>
    set({ description, currentRevisionId }),
  addShare: (share) =>
    set((state) => ({
      shares: [share, ...state.shares.filter((item) => item.token !== share.token)],
    })),
  removeShare: (token) =>
    set((state) => ({
      shares: state.shares.filter((item) => item.token !== token),
    })),
  setJob: (job) =>
    set({
      job,
      loadStatus: job.stage === "failed" ? "error" : "processing",
      loadError: job.stage === "failed" ? job.message : null,
    }),
  setWorkspaceData: (preview, quality) =>
    set((state) => {
      const numericColumns = preview.columns.filter(
        (column) => column.kind === "number",
      );
      const xColumn = numericColumns[0];
      const yColumn = numericColumns[1] ?? numericColumns[0];
      const chartSpec =
        xColumn && yColumn
          ? {
              ...state.chartSpec,
              xAxis: {
                field: xColumn.field,
                title: xColumn.label,
                unit: xColumn.unit ?? "",
              },
              yAxis: {
                field: yColumn.field,
                title: yColumn.label,
                unit: yColumn.unit ?? "",
              },
              series: [
                {
                  ...state.chartSpec.series[0],
                  field: yColumn.field,
                  label: yColumn.label,
                },
              ],
            }
          : state.chartSpec;

      return {
        preview,
        quality,
        chartSpec,
        loadStatus: "ready",
        loadError: null,
      };
    }),
  replaceQuality: (quality) =>
    set({
      quality,
      issueActions: {},
      issueActionPast: [],
      issueActionFuture: [],
      exportReady: false,
    }),
  setLoadError: (loadError) =>
    set({ loadStatus: "error", loadError, job: null }),
  goToStep: (currentStep) => set({ currentStep, exportReady: false }),
  setIssueAction: (issueId, action) =>
    set((state) => {
      if (state.issueActions[issueId] === action) return state;
      return {
        issueActions: { ...state.issueActions, [issueId]: action },
        issueActionPast: [...state.issueActionPast, state.issueActions].slice(-50),
        issueActionFuture: [],
      };
    }),
  setIssueActions: (issueActions) =>
    set((state) => ({
      issueActions,
      issueActionPast: [...state.issueActionPast, state.issueActions].slice(-50),
      issueActionFuture: [],
    })),
  hydrateIssueActions: (issueActions) =>
    set({ issueActions, issueActionPast: [], issueActionFuture: [] }),
  undoIssueActions: () =>
    set((state) => {
      const previous = state.issueActionPast.at(-1);
      if (!previous) return state;
      return {
        issueActions: previous,
        issueActionPast: state.issueActionPast.slice(0, -1),
        issueActionFuture: [state.issueActions, ...state.issueActionFuture].slice(0, 50),
      };
    }),
  redoIssueActions: () =>
    set((state) => {
      const next = state.issueActionFuture[0];
      if (!next) return state;
      return {
        issueActions: next,
        issueActionPast: [...state.issueActionPast, state.issueActions].slice(-50),
        issueActionFuture: state.issueActionFuture.slice(1),
      };
    }),
  restoreIssueActions: () =>
    set((state) => {
      if (Object.keys(state.issueActions).length === 0) return state;
      return {
        issueActions: {},
        issueActionPast: [...state.issueActionPast, state.issueActions].slice(-50),
        issueActionFuture: [],
      };
    }),
  updateChart: (patch) =>
    set((state) => ({ chartSpec: { ...state.chartSpec, ...patch } })),
  updateExport: (patch) =>
    set((state) => ({
      chartSpec: {
        ...state.chartSpec,
        export: { ...state.chartSpec.export, ...patch },
      },
      exportReady: false,
    })),
  prepareExport: () => set({ exportReady: true }),
  reset: () => set(initialState),
}));
