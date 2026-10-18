import type { FigureBoard } from "@/domain/research-figure-board";

export type FigureWorkspaceState = {
  layout: "loading" | "saving" | "saveFailed" | "draft" | "saved";
  generation: "none" | "queued" | "running" | "ready" | "failed" | "unavailable" | "retrying";
  savedId: string | null;
  previewId: string | null;
  downloadsReady: boolean;
};

export function figureFilesReady(board: FigureBoard | null): boolean {
  return !!board?.composed && (board.artifactState === undefined || board.artifactState === "ready");
}

export function figureWorkspaceState({ saved, preview, loading, saving, saveFailed, dirty, retrying = false }: {
  saved: FigureBoard | null; preview: FigureBoard | null; loading: boolean;
  saving: boolean; saveFailed: boolean; dirty: boolean; retrying?: boolean;
}): FigureWorkspaceState {
  return {
    layout: loading ? "loading" : saving ? "saving" : saveFailed ? "saveFailed" : dirty || !saved ? "draft" : "saved",
    generation: retrying ? "retrying" : saved ? figureFilesReady(saved) ? "ready" : saved.artifactState && saved.artifactState !== "ready" ? saved.artifactState : "unavailable" : "none",
    savedId: saved?.id ?? null, previewId: preview?.id ?? null,
    downloadsReady: figureFilesReady(saved),
  };
}

export const figureLayoutLabels = {
  loading: ["Loading layout…", "正在加载布局…"], saving: ["Saving layout…", "正在保存布局…"],
  saveFailed: ["Save failed — draft retained", "保存失败：已保留草稿"],
  draft: ["Unsaved layout", "布局尚未保存"], saved: ["Saved locally", "已保存到本地"],
} as const;
export const figureGenerationLabels = {
  none: ["Not submitted", "尚未提交"], queued: ["Queued", "排队生成"],
  running: ["Generating", "生成中"], ready: ["Files ready", "文件已就绪"],
  retrying: ["Retrying", "重试中"],
  failed: ["Generation failed", "生成失败"], unavailable: ["Files unavailable", "文件不可用"],
} as const;
