"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useLocale } from "next-intl";
import {
  Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Drawer, FormControlLabel, MenuItem, useMediaQuery, Paper, Stack, Switch, Tab, Tabs, TextField, Typography,
} from "@mui/material";
import ExpandMoreRoundedIcon from "@mui/icons-material/ExpandMoreRounded";
import { FigureArrangementPicker } from "./figure-arrangement-picker";
import { FigureStatus } from "./figure-status";
import { figureFilesReady, figureWorkspaceState, type FigureWorkspaceState } from "./figure-workspace-state";
import { ResearchFigureInspector } from "./research-figure-inspector";
import { ResearchFigurePanelEditor } from "./research-figure-panel-editor";
import {
  chartIdsOf, draftFromSpec, draftProblems, duplicatePanel, emptyDraft, figureOf, MAX_PANELS, movePanel, PANEL_LABELS, panelLayers, shares, specFromDraft, storedDraft, withChart, withCount,
  type DraftProblem, type FigureDraft, type PanelDraft,
} from "@/domain/figure-draft";
import { layerKinds, layers } from "@/domain/capabilities";
import { FIGURE_FONTS, type FigureFont } from "@/domain/figure-spec";
import type { RenderPlan } from "@/domain/render-plan";
import type { FigureBoard, FigureBoardHistory, FigureSources } from "@/domain/research-figure-board";
import { researchChartApi } from "@/lib/api/research-chart-api";
import { researchFigureApi } from "@/lib/api/research-figure-api";
import { useInterfaceMode } from "@/components/layout/interface-mode";
import { useDesktopDraft, desktopDraftBase } from "./desktop-session";
import { useRefusalText } from "./use-refusal-text";

export function ResearchFigureWorkspace({ projectId, active = true, currentRevision, refreshKey, onSaved, onStateChange, onSelectAnalysis }: {
  active?: boolean; projectId: string; currentRevision: string | null; refreshKey: number; onSaved?: () => void; onStateChange?: (state: FigureWorkspaceState) => void; onSelectAnalysis?: () => void;
}) {
  const wide = useMediaQuery("(min-width:1400px)");
  const overview = useInterfaceMode().mode === "overview";
  const zh = useLocale().startsWith("zh");
  const t = (en: string, cn: string) => zh ? cn : en;
  const language = zh ? "zh" : "en";
  const [catalog, setCatalog] = useState<FigureSources>({ charts: [] });
  const [history, setHistory] = useState<FigureBoardHistory>({ boards: [] });
  const historyRefresh = useRef({ projectId, refreshKey });
  const [editOpen, setEditOpen] = useState(false);
  const [readyPreview, setReadyPreview] = useState<FigureBoard | null>(null);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [saved, setSaved] = useState<FigureBoard | null>(null);
  const [draft, setDraft] = useState<FigureDraft>(emptyDraft);
  const [captureSample, setCaptureSample] = useState("");
  const [capturePanel, setCapturePanel] = useState(1);
  const [selectedPanel, setSelectedPanel] = useState(0);
  const [inspectorSource, setInspectorSource] = useState<"layout" | "preview">("layout");
  const [previewPanel, setPreviewPanel] = useState(0);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [showPanelPreviews, setShowPanelPreviews] = useState(false);
  const [inspectorTab, setInspectorTab] = useState("data");
  const [undoDrafts, setUndoDrafts] = useState<FigureDraft[]>([]);
  const [redoDrafts, setRedoDrafts] = useState<FigureDraft[]>([]);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [operation, setOperation] = useState<"load" | "save" | "capture" | "retry">("load");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const refusalText = useRefusalText();
  // Source chart plans by chart id: their axes frame the panel editor's limits and categories.
  const [sourcePlans, setSourcePlans] = useState<Record<string, RenderPlan | null>>({});
  const { title, axisMode, palette, fontSize, grid, pointRendering, labelStyle, fontFamily, categoryCounts, provenanceOnFigure, count, panels } = draft;
  useDesktopDraft(`figure:${projectId}`, projectId, { research: currentRevision, figure: saved?.id ?? null }, {
    draft, captureSample, capturePanel, selectedPanel, inspectorSource, previewPanel, inspectorOpen, showPanelPreviews, inspectorTab,
    undoDrafts, redoDrafts, dirty, editOpen,
  }, stored => {
    setDraft(storedDraft(stored.draft ?? stored)); setCaptureSample(stored.captureSample); setCapturePanel(stored.capturePanel);
    setSelectedPanel(stored.selectedPanel); setInspectorSource(stored.inspectorSource); setPreviewPanel(stored.previewPanel);
    setInspectorOpen(stored.inspectorOpen); setShowPanelPreviews(stored.showPanelPreviews); setInspectorTab(stored.inspectorTab);
    setUndoDrafts(stored.undoDrafts.map(storedDraft)); setRedoDrafts(stored.redoDrafts.map(storedDraft)); setDirty(stored.dirty); setEditOpen(stored.editOpen);
  }, !loading, busy || loading);
  useEffect(() => {
    let active = true;
    researchFigureApi.sources(projectId).then((result) => {
      if (active) setCatalog(result);
    }).catch((cause: unknown) => {
      if (active) setError(cause);
    });
    return () => { active = false; };
  }, [projectId, refreshKey, loadAttempt]);
  useEffect(() => {
    let active = true;
    researchFigureApi.list(projectId).then(async (result) => {
      const selected = desktopDraftBase(`figure:${projectId}`)?.figure ?? result.boards[0]?.id;
      const board = selected ? await researchFigureApi.get(projectId, selected) : null;
      if (!active) return;
      setHistory(result);
      if (board) {
        if (figureFilesReady(board)) setReadyPreview(board);
        setSaved(board); setDraft(draftFromSpec(board.spec));
      }
    }).catch((cause: unknown) => {
      if (active) setError(cause);
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [projectId, loadAttempt]);
  useEffect(() => {
    const previous = historyRefresh.current;
    historyRefresh.current = { projectId, refreshKey };
    // Initial/project loads already fetch history above. Refreshing the catalog
    // after a chart/template operation must not reset an edited layout.
    if (previous.projectId !== projectId || previous.refreshKey === refreshKey) return;
    let active = true;
    researchFigureApi.list(projectId).then((result) => { if (active) setHistory(result); })
      .catch((cause: unknown) => { if (active) setError(cause); });
    return () => { active = false; };
  }, [projectId, refreshKey]);
  const savedIdentifier = saved?.id;
  const artifactState = saved?.artifactState;
  useEffect(() => {
    if (!active || !savedIdentifier || !["queued", "running"].includes(artifactState ?? "")) return;
    const identifier = savedIdentifier;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const board = await researchFigureApi.get(projectId, identifier);
        if (disposed) return;
        // Polling updates the saved revision only; ongoing layout edits remain a draft.
        setSaved((previous) => previous?.id === identifier ? board : previous);
        if (figureFilesReady(board)) { setReadyPreview(board); onSaved?.(); }
        if (["queued", "running"].includes(board.artifactState ?? "")) timer = setTimeout(poll, 1000);
      } catch {
        if (!disposed) timer = setTimeout(poll, 2000);
      }
    }
    timer = setTimeout(poll, 500);
    return () => { disposed = true; clearTimeout(timer); };
  }, [active, projectId, savedIdentifier, artifactState, onSaved]);
  /** Every edit goes through here, so undo restores the whole figure as it was. */
  function update(next: FigureDraft) { setUndoDrafts((items) => [...items.slice(-49), draft]); setRedoDrafts([]); setDirty(true); setDraft(next); }
  function updatePanel(index: number, panel: PanelDraft) { update({ ...draft, panels: panels.map((item, i) => (i === index ? panel : item)) }); }
  function restore(value: FigureDraft) { setDraft(value); setSelectedPanel(0); setDirty(true); }
  function duplicate(index: number) {
    const result = duplicatePanel(draft, index);
    if (!result) return;
    update(result.draft); setSelectedPanel(result.target);
  }
  function accept(board: FigureBoard) {
    if (figureFilesReady(board)) setReadyPreview(board);
    setUndoDrafts([]); setRedoDrafts([]);
    setSaved(board); setDraft(draftFromSpec(board.spec));
    setDirty(false); setError(null);
  }
  async function run(action: () => Promise<void>, phase: "load" | "save" | "capture" | "retry" = "load") {
    setOperation(phase); setBusy(true); setError(null);
    try { await action(); }
    catch (cause) { setError(cause); }
    finally { setBusy(false); }
  }
  const blocked = loading || busy;
  const selectedIds = panels.slice(0, count).map((panel) => panel.chartId);
  const chartOf = (id: string) => catalog.charts.find((chart) => chart.id === id);
  const complete = panels.slice(0, count).every((panel) => panel.chartId && panelLayers(panel).every((id) => chartOf(id)));
  const problems = draftProblems(draft);
  const problemText: Record<DraftProblem["kind"], string> = {
    "limits-half": t("give both axis limits, or clear both", "请填写两个坐标轴界限，或都清空"),
    "limits-order": t("the lower axis limit must be below the upper", "坐标轴下限须小于上限"),
    "log-limit": t("a log axis needs limits above zero", "对数轴的界限须大于零"),
    "text-empty": t("a text annotation needs text", "文字标注不能为空"),
  };
  const names: Record<string, string> = {
    line: t("Line", "折线"), scatter: t("Scatter", "散点"), box: t("Box", "箱线"), heatmap: t("Correlation", "相关图"),
    ...Object.fromEntries(layerKinds.map((kind) => [kind, layers[kind].short[language]])),
  };
  const chartTitle = (id: string) => {
    const chart = chartOf(id);
    return chart ? chart.spec.title || names[chart.spec.kind] || chart.spec.kind : id.slice(0, 8);
  };
  const chartName = (id: string) => `${chartTitle(id)} · ${id.slice(0, 8)}`;
  // A panel's layers are research charts of its own chart's research revision on the same x axis:
  // the API compares the same things (paired charts by their two times, others by the revision's groups).
  const xAxisOf = (chart: FigureSources["charts"][number]) => layers[chart.spec.kind]?.xAxis === "pairTimes" ? `times:${chart.spec.pairTimes?.join("\n")}` : "groups";
  const fitsWith = (own: string) => (id: string) => {
    const chart = chartOf(id), base = chartOf(own);
    return !!chart && !!base && !chart.legacy && !base.legacy && chart.revisionId === base.revisionId && xAxisOf(chart) === xAxisOf(base);
  };
  const sameRevision = selectedIds.every((id) => catalog.charts.find((chart) => chart.id === id)?.revisionId === currentRevision);
  const inspectingPreview = overview && inspectorSource === "preview" && !!readyPreview;
  const inspectedIds = inspectingPreview ? chartIdsOf(readyPreview.spec) : selectedIds;
  const inspectedIndex = Math.min(inspectingPreview ? previewPanel : selectedPanel, inspectedIds.length - 1);
  const inspected = catalog.charts.find((c) => c.id === inspectedIds[inspectedIndex]);
  const inspectedBoard = inspectingPreview ? readyPreview : saved;
  const inspectedMetadata = inspectedBoard?.panels.find((panel) => panel.chartId === inspected?.id);
  const editedChart = !inspectingPreview && inspected && !inspected.legacy ? inspected.id : null;
  useEffect(() => {
    if (!active || !editedChart || editedChart in sourcePlans) return;
    let mounted = true;
    // A chart whose plan cannot be read still takes axis controls and notes, without the hints.
    researchChartApi.plan(projectId, editedChart).catch(() => null)
      .then((plan) => { if (mounted) setSourcePlans((plans) => ({ ...plans, [editedChart]: plan })); });
    return () => { mounted = false; };
  }, [active, editedChart, projectId, sourcePlans]);
  function inspectIndex(index: number) { if (inspectingPreview) setPreviewPanel(index); else setSelectedPanel(index); }
  const state = useMemo(() => figureWorkspaceState({ saved, preview: readyPreview, loading,
    saving: busy && operation === "save", retrying: busy && operation === "retry", saveFailed: !!error && operation === "save", dirty }),
    [saved, readyPreview, loading, busy, operation, error, dirty]);
  useEffect(() => { onStateChange?.(state); }, [onStateChange, state]);
  const previewBoard = overview ? readyPreview : figureFilesReady(saved) ? saved : null;
  const inspectorBody = <Stack spacing={2}>
    <Typography component="h3" variant="h6">{t("Selected panel", "选中面板")} {PANEL_LABELS[inspectedIndex]}</Typography>
    {overview ? <Tabs value={inspectorTab} onChange={(_, value: string) => setInspectorTab(value)} variant="fullWidth" aria-label={t("Panel inspector sections", "面板设置分区")}>
      <Tab value="data" aria-label={t("Panel data", "面板数据")} label={t("Data", "数据")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
      <Tab value="method" aria-label={t("Panel method", "面板方法")} label={t("Method", "方法")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
      <Tab value="appearance" aria-label={t("Panel appearance", "面板外观")} label={t("Appearance", "外观")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
    </Tabs> : <TextField select label={t("Inspector section", "设置分区")} value={inspectorTab} onChange={(e) => setInspectorTab(e.target.value)}>
      <MenuItem value="data">{t("Data", "数据")}</MenuItem><MenuItem value="method">{t("Method", "方法")}</MenuItem><MenuItem value="appearance">{t("Appearance", "外观")}</MenuItem>
    </TextField>}
    {inspectorTab === "data" && <>
      <TextField select label={t("Panel to inspect", "检查面板")} value={inspectedIndex} onChange={(e) => inspectIndex(Number(e.target.value))}>
        {inspectedIds.map((_, index) => <MenuItem key={index} value={index}>{PANEL_LABELS[index]}</MenuItem>)}
      </TextField>
      <Typography variant="caption">{inspectingPreview ? t("Saved preview binding", "已保存预览绑定") : dirty ? t("Draft panel binding", "草稿面板绑定") : t("Saved panel binding", "已保存面板绑定")}</Typography>
      {inspectingPreview && <Typography variant="caption" sx={{ overflowWrap: "anywhere" }}>{t("Preview revision", "预览修订")}: {inspectedBoard?.id}</Typography>}
      <Typography sx={{ overflowWrap: "anywhere" }}>{t("Source chart", "来源图表")}: {inspected?.spec.title || (inspected ? names[inspected.spec.kind] : "—")}</Typography>
      <Typography variant="caption" sx={{ overflowWrap: "anywhere" }}>{inspected?.id}</Typography>
      {inspectedMetadata && <Typography variant="caption">{t("Observations", "观测")}: {inspectedMetadata.observations} · {t("Independent units", "独立单位")}: {inspectedMetadata.independentUnits ?? t("Undetermined", "未确定")}</Typography>}
      <Typography sx={{ overflowWrap: "anywhere" }}>{t("Bound revision", "绑定版本")}: {inspected?.revisionId ?? "—"}</Typography><Typography>{t("Changing a binding never modifies the saved source chart.", "调整绑定不会修改已保存的来源图表。")}</Typography></>}
    {inspectorTab === "method" && <><Typography>{t("Chart method", "图表方法")}: {inspected ? names[inspected.spec.kind] : "—"}. {t("Parameters belong to the saved chart. Create a new chart in Analysis to change them.", "参数属于已保存图表，修改方法请在分析任务中生成新图表。")}</Typography>{onSelectAnalysis && <Button onClick={() => { setInspectorOpen(false); onSelectAnalysis(); }}>{t("Open Analysis to create a new version", "前往分析任务生成新版本")}</Button>}</>}
    {inspectorTab === "appearance" && inspected?.legacy && !inspectingPreview
      && <Typography>{t("Captured panels keep the axes they were captured with.", "捕获的面板保留捕获时的坐标轴。")}</Typography>}
    {inspectorTab === "appearance" && !inspected?.legacy && (inspectingPreview || !inspected
      ? <Typography>{t("Palette, font, grid and axis sharing apply to the whole figure. Preview and exports use the saved revision until you save.", "配色、字号、网格与坐标共享作用于整个组合图。保存前，预览及导出仍使用已保存版本。")}</Typography>
      : <ResearchFigurePanelEditor key={`${inspectedIndex}:${inspected.id}`} panel={panels[inspectedIndex]!} label={PANEL_LABELS[inspectedIndex]!}
        axes={sourcePlans[inspected.id]?.panels[0]?.axes ?? null} sharedX={shares(axisMode, "x")} sharedY={shares(axisMode, "y")}
        candidates={catalog.charts.filter((chart) => fitsWith(inspected.id)(chart.id) && !panelLayers(panels[inspectedIndex]!).includes(chart.id)).map((chart) => chart.id)}
        chartName={chartName} chartTitle={chartTitle}
        disabled={blocked} t={t} onChange={(panel) => updatePanel(inspectedIndex, panel)} />)}
  </Stack>;
  return <Paper variant="outlined" sx={{ p: 2 }} data-testid="figure-workspace">
    <Box sx={{ display: "grid", gridTemplateColumns: "minmax(0,1fr)", "@media (min-width:1400px)": { gridTemplateColumns: `minmax(0,1fr) ${overview ? "272px" : "240px"}` }, gap: 2, alignItems: "start" }}>
    <Stack spacing={overview ? 1 : 2} sx={{ minWidth: 0 }}>
      <Typography component="h2" variant="h5">{overview ? t("Saved preview", "已保存预览") : t("Figure layout", "组合图布局")}</Typography>
      {overview && <FigureStatus state={state} />}
      {overview && saved && <Typography variant="subtitle1">{figureOf(saved.spec).title || t("Untitled layout", "未命名布局")}</Typography>}
      {overview && <Box data-testid="saved-figure-summary" sx={{ display: "flex", gap: 1, flexWrap: "wrap", color: "text.secondary" }}>
        <Typography variant="caption">{t("Saved layout revision", "已保存布局修订")}: {saved?.id.slice(0, 12) ?? t("None", "暂无")}</Typography>
        <Typography variant="caption">{t("Panels", "面板")}: {saved ? chartIdsOf(saved.spec).length : 0}</Typography>
      </Box>}
      <Typography role="status" data-testid="figure-save-state" sx={overview ? { position: "absolute", width: "1px", height: "1px", overflow: "hidden", clipPath: "inset(50%)" } : undefined}>
        {loading ? t("Loading layout…", "正在加载布局…") : busy ? operation === "save" ? t("Saving layout…", "正在保存布局…") : operation === "capture" ? t("Capturing and validating chart…", "正在捕获并验证图表…") : t("Loading saved layout…", "正在加载已保存布局…") : error ? operation === "save" ? t("Save failed — draft retained", "保存失败：已保留草稿") : t("Request failed — draft retained", "请求失败：已保留草稿") : dirty || !saved ? t("Unsaved layout", "布局尚未保存") : t("Saved locally", "已保存到本地")}
      </Typography>
      {!!error && <Alert severity="error">{operation === "save" ? t("Save failed; layout draft retained. ", "保存失败，已保留布局草稿。") : t("Request failed; layout draft retained. ", "请求失败，已保留布局草稿。")}{refusalText(error)}{operation === "load" && !dirty && <Button onClick={() => { setError(null); setLoading(true); setLoadAttempt((n) => n + 1); }}>{t("Retry loading layout", "重新加载布局")}</Button>}</Alert>}
      <Button sx={{ "@media (min-width:1400px)": { display: "none" } }} aria-expanded={inspectorOpen} onClick={() => setInspectorOpen(!inspectorOpen)}>{t("Selected panel settings", "选中面板设置")}</Button>
      <Stack spacing={overview ? 1 : 2} sx={{ order: 1, minWidth: 0 }} data-testid="saved-figure-region">
      {saved?.artifactState === "queued" && <Alert severity="info" role="status">{t("Layout saved locally. Figure generation is queued.", "布局已保存到本地，组合图等待生成。")}</Alert>}
      {saved?.artifactState === "running" && <Alert severity="info" role="status">{t("Layout saved locally. Generating preview and export files…", "布局已保存到本地，正在生成预览和导出文件…")}</Alert>}
      {saved?.artifactRefusal && <Alert severity="error" data-testid="figure-refused">
        {t("This revision cannot be drawn: ", "此修订无法绘制：")}{refusalText({ code: saved.artifactRefusal, params: saved.artifactRefusalParams, message: saved.artifactError })} {t("Change the draft and save a new revision.", "请修改草稿后保存新的修订。")}
      </Alert>}
      {(saved?.artifactState === "failed" || saved?.artifactState === "unavailable") && !saved.artifactRefusal && <Alert severity="error">{t("Layout saved locally; figure generation failed. Retry uses the same saved inputs.", "布局已保存到本地，但组合图生成失败。重试将使用同一份已保存输入。")}
        <Button disabled={blocked} onClick={() => void run(async () => {
          const board = await researchFigureApi.retry(projectId, saved.id);
          setSaved(board);
          if (figureFilesReady(board)) { setReadyPreview(board); onSaved?.(); }
        }, "retry")}>{busy && operation === "retry" ? t("Retrying figure generation…", "正在重试生成组合图…") : t("Retry figure generation", "重试生成组合图")}</Button>
      </Alert>}
      {overview && !state.downloadsReady && <Stack direction="row" useFlexGap sx={{ flexWrap: "wrap", gap: 1 }} data-testid="figure-download-gate">
        <Typography sx={{ width: "100%" }}>{t("Waiting for complete figure · PNG / SVG / PDF", "等待完整图形就绪 · PNG / SVG / PDF")}</Typography>
        {["PNG", "SVG", "PDF"].map((format) => <Button disabled key={format}>{t("Download figure", "下载组合图")} {format}</Button>)}
      </Stack>}
      {overview && !previewBoard && !loading && <Paper variant="outlined" sx={{ p: { xs: 2, md: 4 }, textAlign: "center" }}>
        <Typography component="h3" variant="h6">{t("No saved preview yet", "尚无已保存预览")}</Typography>
        <Typography color="text.secondary">{t("Choose saved charts in the draft editor, then save a layout. The preview appears when generation completes.", "在草稿编辑区选择已保存图表并保存布局，生成完成后显示预览。")}</Typography>
        <Button onClick={() => setEditOpen(true)}>{t("Open draft editor", "打开草稿编辑区")}</Button>
        {!catalog.charts.length && onSelectAnalysis && <Button onClick={onSelectAnalysis}>{t("Open Analysis to create a new version", "前往分析任务生成新版本")}</Button>}
      </Paper>}
      {previewBoard && <>
        {overview && <Stack direction="row" useFlexGap sx={{ flexWrap: "wrap", gap: 1 }} aria-label={t("Saved panels", "已保存面板")}>
          {previewBoard.panels.map((panel, index) => <Button key={panel.label} size="small" variant="outlined" sx={{ px: 1, minWidth: 0, fontSize: 12 }}
            title={`${panel.observations} ${t("observations", "观测")} · ${panel.independentUnits ?? t("Undetermined", "未确定")} ${t("independent units", "独立单位")}`}
            aria-pressed={inspectingPreview && inspectedIndex === index} onClick={() => { setPreviewPanel(index); setInspectorSource("preview"); setInspectorOpen(!wide); }}>
            {t("Panel", "面板")} {panel.label} · {panel.observations} {t("obs", "观测")}
          </Button>)}
        </Stack>}
        {dirty && <Alert severity="warning">{t("The preview and downloads belong to the saved revision and exclude draft edits.", "组合图预览及下载属于已保存修订，不包含草稿修改。")}</Alert>}
        {overview && <Typography variant="caption" data-testid="figure-preview-revision" sx={{ display: previewBoard.id === saved?.id ? "none" : undefined }}>
          {previewBoard.id !== saved?.id ? t("Previous saved preview; current generation is not ready", "上一已保存预览；当前生成未就绪") : t("Preview of saved revision", "已保存修订预览")}: {previewBoard.id.slice(0, 12)}
        </Typography>}
        <ResearchFigureInspector key={previewBoard.id} projectId={projectId} boardId={previewBoard.id} active={active}
          downloadsEnabled={!overview || (state.downloadsReady && previewBoard.id === saved?.id)} />
      </>}
      <Typography variant="caption">{t(
        "Saving adds a history entry. Reopening restores the last saved layout; unsaved controls are a draft. Panel order never changes chart values or sample counts.",
        "保存会新增历史版本。重新打开时恢复上次保存的布局，未保存的控件修改属于草稿。调整顺序不会改变图表数值或样本数。")}</Typography>
      </Stack>
      <Accordion expanded={!overview || editOpen} onChange={(_, expanded) => setEditOpen(expanded)} disableGutters
        sx={{ order: 2, minWidth: 0, ...(!overview && { border: 0, "&::before": { display: "none" } }) }} data-testid="figure-draft-editor">
        <AccordionSummary expandIcon={<ExpandMoreRoundedIcon />} sx={{ display: overview ? "flex" : "none" }} aria-controls="figure-draft-controls" id="figure-draft-toggle">
          {t("Edit draft and panel layout", "编辑草稿与面板布局")}{dirty ? t(" · Unsaved", " · 未保存") : ""}
        </AccordionSummary>
        <AccordionDetails sx={{ p: overview ? 2 : 0 }} id="figure-draft-controls">
          <Stack spacing={2} sx={{ minWidth: 0 }}>
      <Stack direction="row" useFlexGap sx={{ flexWrap: "wrap", gap: 1 }}>
        <Button disabled={blocked || !undoDrafts.length} onClick={() => { const value = undoDrafts.at(-1)!; setRedoDrafts((items) => [...items, draft]); setUndoDrafts((items) => items.slice(0,-1)); restore(value); }}>{t("Undo layout edit", "撤销布局编辑")}</Button>
        <Button disabled={blocked || !redoDrafts.length} onClick={() => { const value = redoDrafts.at(-1)!; setUndoDrafts((items) => [...items, draft]); setRedoDrafts((items) => items.slice(0,-1)); restore(value); }}>{t("Redo layout edit", "重做布局编辑")}</Button>

      </Stack>
      {!catalog.charts.length && <Typography>{t(
        "Generate a scientific chart in Analysis to start a layout.",
        "先在分析任务中生成并保存科学图表，再选择面板。")}</Typography>}
      <Accordion>
        <AccordionSummary>{t("Capture from Quick plot (optional)", "从快速出图捕获（可选）")}</AccordionSummary>
        <AccordionDetails><Stack spacing={2}>
      <TextField select label={t("Main workspace panel to capture", "捕获主工作台面板")} value={capturePanel} onChange={(e) => setCapturePanel(Number(e.target.value))} disabled={blocked}>
        {[1, 2, 3, 4].map((n) => <MenuItem key={n} value={n}>{n}</MenuItem>)}
      </TextField>
      <TextField label={t("Sample column for cross-workspace linking (optional)", "跨工作台联动的样本列（可选）")} value={captureSample}
        disabled={blocked} onChange={(e) => setCaptureSample(e.target.value)} helperText={t("Declare a column whose labels identify the same samples as the applied research data. Without this mapping, legacy panels link by source row only.", "请指定与已应用研究数据使用相同样本标识的列。未声明时，旧图表仅按来源行联动，不推测样本身份。")} />
      <Button disabled={blocked} onClick={() => void run(async () => {
        await researchFigureApi.capture(projectId, capturePanel, captureSample.trim());
        setCatalog(await researchFigureApi.sources(projectId));
      }, "capture")}>{t("Capture saved main-workspace chart", "捕获主工作台已保存图表")}</Button>
      <Typography variant="caption">{t("Save the chart in the main workspace first. Capture supports line, scatter, box and correlation; its original panel assignments and 3D charts stay unchanged.", "请先在主工作台保存图表。可捕获折线、散点、箱线和相关图；原面板分配及独立 3D 图保持不变。")}</Typography>
        </Stack></AccordionDetails>
      </Accordion>
      <TextField select label={t("Saved layouts", "已保存布局")} value={saved?.id ?? ""}
        disabled={blocked || dirty || !history.boards.length}
        onChange={(e) => void run(async () => accept(await researchFigureApi.get(projectId, e.target.value)))}>
        {!saved && <MenuItem value="">{t("New layout", "新布局")}</MenuItem>}
        {history.boards.map((board) => <MenuItem key={board.id} value={board.id}>
          {figureOf(board.spec).title || t("Untitled layout", "未命名布局")} · {board.id.slice(0, 8)}
        </MenuItem>)}
      </TextField>
      {dirty && saved && <Button disabled={blocked} onClick={() => accept(saved)}>
        {t("Discard draft and restore saved layout", "放弃草稿并恢复已保存布局")}</Button>}
      <TextField label={t("Figure title", "组合图标题")} value={title} disabled={blocked}
        slotProps={{ htmlInput: { maxLength: 160 } }} onChange={(e) => update({ ...draft, title: e.target.value })} />
      <TextField select label={t("Panel count", "面板数量")} value={count} disabled={blocked}
        onChange={(e) => update(withCount(draft, Number(e.target.value)))}>
        {Array.from({ length: MAX_PANELS }, (_, i) => i + 1).map((n) => <MenuItem value={n} key={n}>{n}</MenuItem>)}
      </TextField>
      <FigureArrangementPicker count={count} value={draft.layout} disabled={blocked} t={t} onChange={(layout) => update({ ...draft, layout })} />
      <Accordion>
        <AccordionSummary>{t("Figure appearance and shared axes", "组合图外观与共享坐标")}</AccordionSummary>
        <AccordionDetails><Stack spacing={2}>
      <TextField select label={t("Axis sharing", "坐标轴共享")} value={axisMode} disabled={blocked}
        onChange={(e) => update({ ...draft, axisMode: e.target.value as FigureDraft["axisMode"] })}>
        {[["independent", t("Independent", "独立")], ["shared_x", t("Shared X", "共享 X")], ["shared_y", t("Shared Y", "共享 Y")], ["shared_xy", t("Shared X and Y", "共享 X 和 Y")]].map(([v, label]) => <MenuItem key={v} value={v}>{label}</MenuItem>)}
      </TextField>
      <TextField select label={t("Figure palette", "组合图配色")} value={palette} disabled={blocked} onChange={(e) => update({ ...draft, palette: e.target.value as FigureDraft["palette"] })}>
        <MenuItem value="blue_teal">{t("Blue / teal", "蓝／青")}</MenuItem><MenuItem value="grayscale">{t("Grayscale", "灰度")}</MenuItem>
      </TextField>
      <TextField select label={t("Panel labels", "面板标号")} value={labelStyle} disabled={blocked} onChange={(e) => update({ ...draft, labelStyle: e.target.value as FigureDraft["labelStyle"] })}>
        <MenuItem value="upper">A, B, C</MenuItem><MenuItem value="lower">a, b, c</MenuItem><MenuItem value="none">{t("None", "无")}</MenuItem>
      </TextField>
      <TextField select label={t("Point layer export", "点层导出")} value={pointRendering}
        disabled={blocked} onChange={(e) => update({ ...draft, pointRendering: e.target.value as FigureDraft["pointRendering"] })}
        helperText={t("Auto embeds all points at 300 dpi above 20,000 observations per panel. Text, axes and density stay vector; embedded points are not individually editable. Source data is complete.", "自动模式在每面板超过 20,000 个观测时以 300 dpi 嵌入全部点。文字、坐标轴、密度保留矢量；嵌入点无法逐个编辑，源数据完整保留。")}>
        <MenuItem value="auto">{t("Auto (large data hybrid)", "自动（大数据混合导出）")}</MenuItem>
        <MenuItem value="vector">{t("Force vector (slower for large data)", "强制全矢量（大数据较慢）")}</MenuItem>
        <MenuItem value="raster">{t("300 dpi point layer", "300 dpi 点层")}</MenuItem>
      </TextField>
      <TextField select label={t("Figure font", "组合图字体")} value={fontFamily} disabled={blocked} onChange={(e) => update({ ...draft, fontFamily: e.target.value as FigureFont })}>
        {FIGURE_FONTS.map((font) => <MenuItem key={font} value={font}>{font}</MenuItem>)}
      </TextField>
      <TextField select label={t("Figure font size", "组合图字号")} value={fontSize} disabled={blocked} onChange={(e) => update({ ...draft, fontSize: Number(e.target.value) })}>
        {[8, 10, 12, 14, 16].map((size) => <MenuItem key={size} value={size}>{size}</MenuItem>)}
      </TextField>
      <Button disabled={blocked} onClick={() => update({ ...draft, grid: !grid })}>{grid ? t("Hide figure grid", "隐藏组合图网格") : t("Show figure grid", "显示组合图网格")}</Button>
      <FormControlLabel label={t("Counts under category names", "分类名下显示计数")} disabled={blocked}
        control={<Switch checked={categoryCounts} onChange={(e) => update({ ...draft, categoryCounts: e.target.checked })} />} />
      <FormControlLabel label={t("Source details under each panel", "每个面板下显示来源信息")} disabled={blocked}
        control={<Switch checked={provenanceOnFigure} onChange={(e) => update({ ...draft, provenanceOnFigure: e.target.checked })} />} />
        </Stack></AccordionDetails>
      </Accordion>
      {saved?.composed && !dirty && <Button aria-pressed={showPanelPreviews} onClick={() => setShowPanelPreviews(!showPanelPreviews)}>
        {showPanelPreviews ? t("Hide source chart thumbnails", "收起来源图表缩略图") : t("Show source chart thumbnails", "显示来源图表缩略图")}
      </Button>}
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: count === 1 ? "1fr" : "repeat(2, minmax(0, 1fr))" }, gap: 2 }}>
        {selectedIds.map((id, index) => {
          const label = PANEL_LABELS[index]!;
          const chart = catalog.charts.find((item) => item.id === id);
          const spec = chart?.spec;
          const metadata = saved?.panels.find((panel) => panel.chartId === id);
          const noted = panels[index]!.notes.length, layered = panels[index]!.layers.length;
          // A control on a shared axis is not sent, so it does not count as the panel's own.
          const scaled = (["x", "y"] as const).some((dimension) => !shares(axisMode, dimension)
            && (panels[index]!.axes[dimension].scale !== "linear" || panels[index]!.axes[dimension].domain));
          return <Paper component="section" variant="outlined" key={label} sx={{ p: 1.5, minWidth: 0 }}
            aria-label={t(`Panel ${label}`, `面板 ${label}`)} data-testid={`figure-panel-${label}`}>
            <Stack spacing={1}>
              <Typography component="h3" variant="h6">{label}</Typography>
              <Button aria-pressed={selectedPanel === index} onClick={() => { setSelectedPanel(index); setInspectorSource("layout"); setInspectorOpen(!wide); }}>{t(`Inspect panel ${label}`, `检查面板 ${label}`)}</Button>
              <TextField select label={t(`Chart for panel ${label}`, `面板 ${label} 的图表`)} value={id} disabled={blocked}
                onChange={(e) => updatePanel(index, withChart(panels[index]!, e.target.value, fitsWith(e.target.value)))}>
                <MenuItem value="">{t("Choose saved chart", "选择已保存图表")}</MenuItem>
                {catalog.charts.map((item) => {
                  const itemSpec = item.spec;
                  return <MenuItem key={item.id} value={item.id}>{itemSpec.title || names[itemSpec.kind]} · {item.id.slice(0, 8)}</MenuItem>;
                })}
              </TextField>
              <TextField label={t(`Panel ${label} title (figure only)`, `面板 ${label} 标题（仅作用于组合图）`)}
                value={panels[index]!.title} disabled={blocked} slotProps={{ htmlInput: { maxLength: 160 } }}
                onChange={(e) => updatePanel(index, { ...panels[index]!, title: e.target.value })} />
              <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: "wrap" }}>
                <Button disabled={blocked || index === 0} onClick={() => update(movePanel(draft, index, -1))}
                  aria-label={t(`Move panel ${label} earlier`, `面板 ${label} 前移`)}>{t("Earlier", "前移")}</Button>
                <Button disabled={blocked || index === count - 1} onClick={() => update(movePanel(draft, index, 1))}
                  aria-label={t(`Move panel ${label} later`, `面板 ${label} 后移`)}>{t("Later", "后移")}</Button>
                <Button disabled={blocked || !id || (count === MAX_PANELS && selectedIds.every(Boolean))} onClick={() => duplicate(index)}>{t(`Duplicate panel ${label}`, `复制面板 ${label}`)}</Button>
                <Button disabled={blocked || !id} onClick={() => updatePanel(index, withChart(panels[index]!, "", () => false))}>{t(`Clear panel ${label}`, `清空面板 ${label}`)}</Button>
              </Stack>
              {chart && spec && <>
                {active && !loading && (!saved?.composed || dirty || showPanelPreviews) && !chart.legacy && <Box component="img" src={researchChartApi.artifact(projectId, chart.id, "png", false, true, 480)}
                  alt={`${label}: ${spec.title || names[spec.kind]}`}
                  sx={{ width: "100%", height: "auto", bgcolor: "white" }} />}
                {chart.legacy && <Typography>{t("Captured legacy panel; save to preview composition.", "已捕获旧图表面板；保存后预览组合图。")}</Typography> }
                <Typography variant="body2" sx={{ overflowWrap: "anywhere" }}>
                  {scaled ? t("Own axis settings", "自定坐标轴") : t("Independent axes", "独立坐标轴")}{layered ? ` · ${layered} ${t("layers", "个图层")}` : ""}{noted ? ` · ${noted} ${t(noted === 1 ? "annotation" : "annotations", "个标注")}` : ""} · {t("research revision", "研究版本")}: {chart.revisionId.slice(0, 8)}
                </Typography>
                {metadata && <Typography variant="caption">
                  {t("Observations", "观测")}: {metadata.observations} · {t("Independent units", "独立单位")}: {metadata.independentUnits ?? "?"} · {metadata.unit}
                </Typography>}
              </>}
            </Stack>
          </Paper>;
        })}
      </Box>
      {problems.length > 0 && <Alert severity="warning" data-testid="figure-draft-problems">
        {problems.map(({ panel, kind }) => <Typography key={`${panel}:${kind}`} variant="body2">{t(`Panel ${PANEL_LABELS[panel]}`, `面板 ${PANEL_LABELS[panel]}`)}: {problemText[kind]}</Typography>)}
      </Alert>}
      {complete && !sameRevision && <Alert severity="info">{t(
        "Some panels use historical data revisions. Saving preserves those exact chart bindings.",
        "部分面板使用历史数据版本；保存时会保留这些图表绑定，不会替换为当前数据。")}</Alert>}
      <Button variant="contained" disabled={blocked || !complete || problems.length > 0 || (!dirty && !!saved && (saved.composed || ["queued", "running", "failed"].includes(saved.artifactState ?? "")))}
        onClick={() => void run(async () => {
          const board = await researchFigureApi.save(projectId, specFromDraft(draft));
          accept(board);
          setHistory((previous) => ({ boards: [board, ...previous.boards] }));
          onSaved?.();
        }, "save")}>{t("Save new layout revision", "保存新的布局版本")}</Button>
          </Stack>
        </AccordionDetails>
      </Accordion>
    </Stack>
    <Paper id="panel-inspector" component="aside" variant="outlined" sx={{ p: 2, display: "none", position: "sticky", top: overview ? 104 : 56, minWidth: 0, "@media (min-width:1400px)": { display: "block" } }} aria-label={t("Selected panel inspector", "选中面板检查器")}>
      {wide && inspectorBody}
    </Paper>
    <Drawer slotProps={{ paper: { role: "dialog", "aria-label": t("Selected panel inspector", "选中面板检查器"), "aria-modal": true } }} anchor="right" open={inspectorOpen && !wide} onClose={() => setInspectorOpen(false)} sx={{ "@media (min-width:1400px)": { display: "none" } }}>
      <Stack sx={{ width: { xs: "min(340px, 90vw)", sm: 340 }, p: 2 }} spacing={2}>
        <Button onClick={() => setInspectorOpen(false)}>{t("Return to figure", "返回组合图")}</Button>
        {!wide && inspectorBody}
      </Stack>
    </Drawer>
    </Box>
  </Paper>;
}
