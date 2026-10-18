"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useLocale } from "next-intl";
import {
  Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Drawer, MenuItem, useMediaQuery, Paper, Stack, Tab, Tabs, TextField, Typography,
} from "@mui/material";
import ExpandMoreRoundedIcon from "@mui/icons-material/ExpandMoreRounded";
import { FigureStatus } from "./figure-status";
import { figureFilesReady, figureWorkspaceState, type FigureWorkspaceState } from "./figure-workspace-state";
import { ResearchFigureInspector } from "./research-figure-inspector";
import type { FigureBoard, FigureBoardHistory, FigureBoardSpec, FigureSources } from "@/domain/research-figure-board";
import { researchChartApi } from "@/lib/api/research-chart-api";
import { researchFigureApi } from "@/lib/api/research-figure-api";
import { useInterfaceMode } from "@/components/layout/interface-mode";

type LayoutDraft = { title: string; axisMode: FigureBoardSpec["axisMode"]; palette: "blue_teal" | "grayscale"; fontSize: number; grid: boolean; pointRendering: "auto" | "vector" | "raster"; count: number; panelTitles: string[]; slots: string[] };

export function ResearchFigureWorkspace({ projectId, active = true, currentRevision, refreshKey, onSaved, onStateChange, onSelectAnalysis }: {
  active?: boolean; projectId: string; currentRevision: string | null; refreshKey: number; onSaved?: () => void; onStateChange?: (state: FigureWorkspaceState) => void; onSelectAnalysis?: () => void;
}) {
  const wide = useMediaQuery("(min-width:1400px)");
  const overview = useInterfaceMode().mode === "overview";
  const zh = useLocale().startsWith("zh");
  const t = (en: string, cn: string) => zh ? cn : en;
  const [catalog, setCatalog] = useState<FigureSources>({ charts: [] });
  const [history, setHistory] = useState<FigureBoardHistory>({ boards: [] });
  const historyRefresh = useRef({ projectId, refreshKey });
  const [editOpen, setEditOpen] = useState(false);
  const [readyPreview, setReadyPreview] = useState<FigureBoard | null>(null);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [saved, setSaved] = useState<FigureBoard | null>(null);
  const [title, setTitle] = useState("");
  const [axisMode, setAxisMode] = useState<FigureBoardSpec["axisMode"]>("independent");
  const [palette, setPalette] = useState<"blue_teal" | "grayscale">("blue_teal");
  const [fontSize, setFontSize] = useState(10);
  const [grid, setGrid] = useState(true);
  const [pointRendering, setPointRendering] = useState<"auto" | "vector" | "raster">("auto");
  const [captureSample, setCaptureSample] = useState("");
  const [capturePanel, setCapturePanel] = useState(1);
  const [count, setCount] = useState(1);
  const [slots, setSlots] = useState(["", "", "", ""]);
  const [panelTitles, setPanelTitles] = useState(["", "", "", ""]);
  const [selectedPanel, setSelectedPanel] = useState(0);
  const [inspectorSource, setInspectorSource] = useState<"layout" | "preview">("layout");
  const [previewPanel, setPreviewPanel] = useState(0);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [showPanelPreviews, setShowPanelPreviews] = useState(false);
  const [inspectorTab, setInspectorTab] = useState("data");
  const [undoDrafts, setUndoDrafts] = useState<LayoutDraft[]>([]);
  const [redoDrafts, setRedoDrafts] = useState<LayoutDraft[]>([]);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [operation, setOperation] = useState<"load" | "save" | "capture" | "retry">("load");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    researchFigureApi.sources(projectId).then((result) => {
      if (active) setCatalog(result);
    }).catch((cause: unknown) => {
      if (active) setError(cause instanceof Error ? cause.message : "Chart list unavailable");
    });
    return () => { active = false; };
  }, [projectId, refreshKey, loadAttempt]);
  useEffect(() => {
    let active = true;
    researchFigureApi.list(projectId).then(async (result) => {
      const board = result.boards[0] ? await researchFigureApi.get(projectId, result.boards[0].id) : null;
      if (!active) return;
      setHistory(result);
      if (board) {
        if (figureFilesReady(board)) setReadyPreview(board);
        const spec = board.spec as FigureBoardSpec;
        setSaved(board); setTitle(spec.title); setCount(spec.panelChartIds.length);
        setPanelTitles([...(spec.panelTitles ?? []), ...Array<string>(4 - (spec.panelTitles?.length ?? 0)).fill("")]);
        setAxisMode(spec.axisMode); setPalette(spec.palette ?? "blue_teal"); setFontSize(spec.fontSize ?? 10); setGrid(spec.gridVisible ?? true); setPointRendering(spec.pointRendering ?? "auto");
        setSlots([...spec.panelChartIds, ...Array<string>(4 - spec.panelChartIds.length).fill("")]);
      }
    }).catch((cause: unknown) => {
      if (active) setError(cause instanceof Error ? cause.message : "Layout load failed");
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
      .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Layout load failed"); });
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
  function draft(): LayoutDraft { return { title, axisMode, palette, fontSize, grid, pointRendering, count, panelTitles: [...panelTitles], slots: [...slots] }; }
  function edited() { setUndoDrafts((items) => [...items.slice(-49), draft()]); setRedoDrafts([]); setDirty(true); }
  function restore(value: LayoutDraft) {
    setTitle(value.title); setAxisMode(value.axisMode); setPalette(value.palette); setFontSize(value.fontSize);
    setPanelTitles(value.panelTitles); setGrid(value.grid); setPointRendering(value.pointRendering); setCount(value.count); setSlots(value.slots); setSelectedPanel(0); setDirty(true);
  }
  function duplicate(index: number) {
    const next = [...slots]; const empty = next.slice(0,count).findIndex((v) => !v);
    const target = empty >= 0 ? empty : count;
    if (target >= 4) return;
    edited(); next[target] = slots[index]; setSlots(next);
    setPanelTitles((prev) => prev.map((value, i) => i === target ? prev[index] : value));
    if (target >= count) setCount(count === 1 ? 2 : 4);
    setSelectedPanel(target);
  }
  function accept(board: FigureBoard) {
    if (figureFilesReady(board)) setReadyPreview(board);
    setUndoDrafts([]); setRedoDrafts([]);
    const spec = board.spec as FigureBoardSpec;
    setSaved(board); setTitle(spec.title); setCount(spec.panelChartIds.length);
        setPanelTitles([...(spec.panelTitles ?? []), ...Array<string>(4 - (spec.panelTitles?.length ?? 0)).fill("")]);
        setAxisMode(spec.axisMode); setPalette(spec.palette ?? "blue_teal"); setFontSize(spec.fontSize ?? 10); setGrid(spec.gridVisible ?? true); setPointRendering(spec.pointRendering ?? "auto");
    setSlots([...spec.panelChartIds, ...Array<string>(4 - spec.panelChartIds.length).fill("")]);
    setDirty(false); setError(null);
  }
  async function run(action: () => Promise<void>, phase: "load" | "save" | "capture" | "retry" = "load") {
    setOperation(phase); setBusy(true); setError(null);
    try { await action(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Layout request failed"); }
    finally { setBusy(false); }
  }
  function move(index: number, delta: number) {
    const next = [...slots];
    [next[index], next[index + delta]] = [next[index + delta], next[index]];
    const titles = [...panelTitles];
    [titles[index], titles[index + delta]] = [titles[index + delta], titles[index]];
    setPanelTitles(titles); setSlots(next); edited();
  }
  const blocked = loading || busy;
  const selectedIds = slots.slice(0, count);
  const complete = selectedIds.every((id) => catalog.charts.some((chart) => chart.id === id));
  const names: Record<string, string> = {
    line: t("Line", "折线"), scatter: t("Scatter", "散点"), box: t("Box", "箱线"), heatmap: t("Correlation", "相关图"),
    grouped_strip: t("Raw points", "原始点图"), paired: t("Paired change", "配对变化图"), violin: t("Violin", "小提琴图"),
  };
  const sameRevision = selectedIds.every((id) => catalog.charts.find((chart) => chart.id === id)?.revisionId === currentRevision);
  const inspectingPreview = overview && inspectorSource === "preview" && !!readyPreview;
  const inspectedIds = inspectingPreview ? (readyPreview.spec as FigureBoardSpec).panelChartIds : selectedIds;
  const inspectedIndex = Math.min(inspectingPreview ? previewPanel : selectedPanel, inspectedIds.length - 1);
  const inspected = catalog.charts.find((c) => c.id === inspectedIds[inspectedIndex]);
  const inspectedBoard = inspectingPreview ? readyPreview : saved;
  const inspectedMetadata = inspectedBoard?.panels.find((panel) => panel.chartId === inspected?.id);
  function inspectIndex(index: number) { if (inspectingPreview) setPreviewPanel(index); else setSelectedPanel(index); }
  const state = useMemo(() => figureWorkspaceState({ saved, preview: readyPreview, loading,
    saving: busy && operation === "save", retrying: busy && operation === "retry", saveFailed: !!error && operation === "save", dirty }),
    [saved, readyPreview, loading, busy, operation, error, dirty]);
  useEffect(() => { onStateChange?.(state); }, [onStateChange, state]);
  const previewBoard = overview ? readyPreview : figureFilesReady(saved) ? saved : null;
  return <Paper variant="outlined" sx={{ p: 2 }} data-testid="figure-workspace">
    <Box sx={{ display: "grid", gridTemplateColumns: "minmax(0,1fr)", "@media (min-width:1400px)": { gridTemplateColumns: `minmax(0,1fr) ${overview ? "272px" : "240px"}` }, gap: 2, alignItems: "start" }}>
    <Stack spacing={overview ? 1 : 2} sx={{ minWidth: 0 }}>
      <Typography component="h2" variant="h5">{overview ? t("Saved preview", "已保存预览") : t("Figure layout", "组合图布局")}</Typography>
      {overview && <FigureStatus state={state} />}
      {overview && saved && <Typography variant="subtitle1">{(saved.spec as FigureBoardSpec).title || t("Untitled layout", "未命名布局")}</Typography>}
      {overview && <Box data-testid="saved-figure-summary" sx={{ display: "flex", gap: 1, flexWrap: "wrap", color: "text.secondary" }}>
        <Typography variant="caption">{t("Saved layout revision", "已保存布局修订")}: {saved?.id.slice(0, 12) ?? t("None", "暂无")}</Typography>
        <Typography variant="caption">{t("Panels", "面板")}: {(saved?.spec as FigureBoardSpec | undefined)?.panelChartIds.length ?? 0}</Typography>
        <Typography variant="caption">{t("Counts belong to each panel's source revision.", "计数按各面板来源修订分别列出。")}</Typography>
      </Box>}
      <Alert severity="info" sx={overview ? { display: "none" } : undefined}>{t(
        "Save the layout to keep your panel bindings. The composed figure generates in the background. Shared axes require matching meanings, units and categories. Selections inspect saved data only.",
        "保存布局后，面板配置会保留到本地，整页组合图在后台生成。共享坐标轴要求含义、单位及类别一致。点选仅查看已保存数据。")}</Alert>
      <Typography role="status" data-testid="figure-save-state" sx={overview ? { position: "absolute", width: "1px", height: "1px", overflow: "hidden", clipPath: "inset(50%)" } : undefined}>
        {loading ? t("Loading layout…", "正在加载布局…") : busy ? operation === "save" ? t("Saving layout…", "正在保存布局…") : operation === "capture" ? t("Capturing and validating chart…", "正在捕获并验证图表…") : t("Loading saved layout…", "正在加载已保存布局…") : error ? operation === "save" ? t("Save failed — draft retained", "保存失败：已保留草稿") : t("Request failed — draft retained", "请求失败：已保留草稿") : dirty || !saved ? t("Unsaved layout", "布局尚未保存") : t("Saved locally", "已保存到本地")}
      </Typography>
      {error && <Alert severity="error">{operation === "save" ? t("Save failed; layout draft retained. ", "保存失败，已保留布局草稿。") : t("Request failed; layout draft retained. ", "请求失败，已保留布局草稿。")}{error}{operation === "load" && !dirty && <Button onClick={() => { setError(null); setLoading(true); setLoadAttempt((n) => n + 1); }}>{t("Retry loading layout", "重新加载布局")}</Button>}</Alert>}
      <Button sx={{ "@media (min-width:1400px)": { display: "none" } }} aria-expanded={inspectorOpen} onClick={() => setInspectorOpen(!inspectorOpen)}>{t("Selected panel settings", "选中面板设置")}</Button>
      <Stack spacing={overview ? 1 : 2} sx={{ order: 1, minWidth: 0 }} data-testid="saved-figure-region">
      {saved?.artifactState === "queued" && <Alert severity="info" role="status">{t("Layout saved locally. Figure generation is queued.", "布局已保存到本地，组合图等待生成。")}</Alert>}
      {saved?.artifactState === "running" && <Alert severity="info" role="status">{t("Layout saved locally. Generating preview and export files…", "布局已保存到本地，正在生成预览和导出文件…")}</Alert>}
      {(saved?.artifactState === "failed" || saved?.artifactState === "unavailable") && <Alert severity="error">{t("Layout saved locally; figure generation failed. Retry uses the same saved inputs.", "布局已保存到本地，但组合图生成失败。重试将使用同一份已保存输入。")}
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
          {overview && <Alert severity="info">{t("Saving creates a new layout revision. Draft edits do not change the saved preview or exports.", "保存会创建新的布局修订。草稿编辑不会改变已保存预览或导出。")}</Alert>}
      <Stack direction="row" useFlexGap sx={{ flexWrap: "wrap", gap: 1 }}>
        <Button disabled={blocked || !undoDrafts.length} onClick={() => { const value = undoDrafts.at(-1)!; setRedoDrafts((items) => [...items, draft()]); setUndoDrafts((items) => items.slice(0,-1)); restore(value); }}>{t("Undo layout edit", "撤销布局编辑")}</Button>
        <Button disabled={blocked || !redoDrafts.length} onClick={() => { const value = redoDrafts.at(-1)!; setUndoDrafts((items) => [...items, draft()]); setRedoDrafts((items) => items.slice(0,-1)); restore(value); }}>{t("Redo layout edit", "重做布局编辑")}</Button>

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
          {(board.spec as FigureBoardSpec).title || t("Untitled layout", "未命名布局")} · {board.id.slice(0, 8)}
        </MenuItem>)}
      </TextField>
      {dirty && saved && <Button disabled={blocked} onClick={() => accept(saved)}>
        {t("Discard draft and restore saved layout", "放弃草稿并恢复已保存布局")}</Button>}
      <TextField label={t("Figure title", "组合图标题")} value={title} disabled={blocked}
        slotProps={{ htmlInput: { maxLength: 160 } }} onChange={(e) => { setTitle(e.target.value); edited(); }} />
      <TextField select label={t("Panel count", "面板数量")} value={count} disabled={blocked}
        onChange={(e) => { setCount(Number(e.target.value)); edited(); }}>
        {[1, 2, 4].map((n) => <MenuItem value={n} key={n}>{n}</MenuItem>)}
      </TextField>
      <Accordion>
        <AccordionSummary>{t("Figure appearance and shared axes", "组合图外观与共享坐标")}</AccordionSummary>
        <AccordionDetails><Stack spacing={2}>
      <TextField select label={t("Axis sharing", "坐标轴共享")} value={axisMode} disabled={blocked}
        onChange={(e) => { setAxisMode(e.target.value as FigureBoardSpec["axisMode"]); edited(); }}>
        {[["independent", t("Independent", "独立")], ["shared_x", t("Shared X", "共享 X")], ["shared_y", t("Shared Y", "共享 Y")], ["shared_xy", t("Shared X and Y", "共享 X 和 Y")]].map(([v, label]) => <MenuItem key={v} value={v}>{label}</MenuItem>)}
      </TextField>
      <TextField select label={t("Figure palette", "组合图配色")} value={palette} disabled={blocked} onChange={(e) => { setPalette(e.target.value as typeof palette); edited(); }}>
        <MenuItem value="blue_teal">{t("Blue / teal", "蓝／青")}</MenuItem><MenuItem value="grayscale">{t("Grayscale", "灰度")}</MenuItem>
      </TextField>
      <TextField select label={t("Point layer export", "点层导出")} value={pointRendering}
        disabled={blocked} onChange={(e) => { setPointRendering(e.target.value as typeof pointRendering); edited(); }}
        helperText={t("Auto embeds all points at 300 dpi above 20,000 observations per panel. Text, axes and density stay vector; embedded points are not individually editable. Source data is complete.", "自动模式在每面板超过 20,000 个观测时以 300 dpi 嵌入全部点。文字、坐标轴、密度保留矢量；嵌入点无法逐个编辑，源数据完整保留。")}>
        <MenuItem value="auto">{t("Auto (large data hybrid)", "自动（大数据混合导出）")}</MenuItem>
        <MenuItem value="vector">{t("Force vector (slower for large data)", "强制全矢量（大数据较慢）")}</MenuItem>
        <MenuItem value="raster">{t("300 dpi point layer", "300 dpi 点层")}</MenuItem>
      </TextField>
      <TextField select label={t("Figure font size", "组合图字号")} value={fontSize} disabled={blocked} onChange={(e) => { setFontSize(Number(e.target.value)); edited(); }}>
        {[8, 10, 12, 14, 16].map((size) => <MenuItem key={size} value={size}>{size}</MenuItem>)}
      </TextField>
      <Button disabled={blocked} onClick={() => { setGrid(!grid); edited(); }}>{grid ? t("Hide figure grid", "隐藏组合图网格") : t("Show figure grid", "显示组合图网格")}</Button>
        </Stack></AccordionDetails>
      </Accordion>
      {saved?.composed && !dirty && <Button aria-pressed={showPanelPreviews} onClick={() => setShowPanelPreviews(!showPanelPreviews)}>
        {showPanelPreviews ? t("Hide source chart thumbnails", "收起来源图表缩略图") : t("Show source chart thumbnails", "显示来源图表缩略图")}
      </Button>}
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: count === 1 ? "1fr" : "repeat(2, minmax(0, 1fr))" }, gap: 2 }}>
        {selectedIds.map((id, index) => {
          const label = "ABCD"[index];
          const chart = catalog.charts.find((item) => item.id === id);
          const spec = chart?.spec;
          const metadata = saved?.panels.find((panel) => panel.chartId === id);
          return <Paper component="section" variant="outlined" key={label} sx={{ p: 1.5, minWidth: 0 }}
            aria-label={t(`Panel ${label}`, `面板 ${label}`)} data-testid={`figure-panel-${label}`}>
            <Stack spacing={1}>
              <Typography component="h3" variant="h6">{label}</Typography>
              <Button aria-pressed={selectedPanel === index} onClick={() => { setSelectedPanel(index); setInspectorSource("layout"); setInspectorOpen(!wide); }}>{t(`Inspect panel ${label}`, `检查面板 ${label}`)}</Button>
              <TextField select label={t(`Chart for panel ${label}`, `面板 ${label} 的图表`)} value={id} disabled={blocked}
                onChange={(e) => { setSlots((previous) => previous.map((value, i) => i === index ? e.target.value : value)); edited(); }}>
                <MenuItem value="">{t("Choose saved chart", "选择已保存图表")}</MenuItem>
                {catalog.charts.map((item) => {
                  const itemSpec = item.spec;
                  return <MenuItem key={item.id} value={item.id}>{itemSpec.title || names[itemSpec.kind]} · {item.id.slice(0, 8)}</MenuItem>;
                })}
              </TextField>
              <TextField label={t(`Panel ${label} title (figure only)`, `面板 ${label} 标题（仅作用于组合图）`)}
                value={panelTitles[index]} disabled={blocked} slotProps={{ htmlInput: { maxLength: 160 } }}
                helperText={t("Blank keeps the source title. Saved source values and method stay unchanged.", "留空沿用来源标题，来源图表的数值和方法保持不变。")}
                onChange={(e) => { edited(); setPanelTitles((prev) => prev.map((value, i) => i === index ? e.target.value : value)); }} />
              <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: "wrap" }}>
                <Button disabled={blocked || index === 0} onClick={() => move(index, -1)}
                  aria-label={t(`Move panel ${label} earlier`, `面板 ${label} 前移`)}>{t("Earlier", "前移")}</Button>
                <Button disabled={blocked || index === count - 1} onClick={() => move(index, 1)}
                  aria-label={t(`Move panel ${label} later`, `面板 ${label} 后移`)}>{t("Later", "后移")}</Button>
                <Button disabled={blocked || !id || (count === 4 && selectedIds.every(Boolean))} onClick={() => duplicate(index)}>{t(`Duplicate panel ${label}`, `复制面板 ${label}`)}</Button>
                <Button disabled={blocked || !id} onClick={() => { edited(); setSlots((prev) => prev.map((v,i) => i === index ? "" : v)); }}>{t(`Clear panel ${label}`, `清空面板 ${label}`)}</Button>
              </Stack>
              {chart && spec && <>
                {active && !loading && (!saved?.composed || dirty || showPanelPreviews) && !chart.legacy && <Box component="img" src={researchChartApi.artifact(projectId, chart.id, "png", false, true, 480)}
                  alt={`${label}: ${spec.title || names[spec.kind]}`}
                  sx={{ width: "100%", height: "auto", bgcolor: "white" }} />}
                {chart.legacy && <Typography>{t("Captured legacy panel; save to preview composition.", "已捕获旧图表面板；保存后预览组合图。")}</Typography> }
                <Typography variant="body2" sx={{ overflowWrap: "anywhere" }}>
                  {t("Independent axes · research revision", "独立坐标轴 · 研究版本")}: {chart.revisionId.slice(0, 8)}
                </Typography>
                {metadata && <Typography variant="caption">
                  {t("Observations", "观测")}: {metadata.observations} · {t("Independent units", "独立单位")}: {metadata.independentUnits ?? "?"} · {metadata.unit}
                </Typography>}
              </>}
            </Stack>
          </Paper>;
        })}
      </Box>
      {complete && !sameRevision && <Alert severity="info">{t(
        "Some panels use historical data revisions. Saving preserves those exact chart bindings.",
        "部分面板使用历史数据版本；保存时会保留这些图表绑定，不会替换为当前数据。")}</Alert>}
      <Button variant="contained" disabled={blocked || !complete || (!dirty && !!saved && (saved.composed || ["queued", "running", "failed"].includes(saved.artifactState ?? "")))}
        onClick={() => void run(async () => {
          const board = await researchFigureApi.save(projectId, {
            schemaVersion: 1, title, axisMode, palette, fontSize, gridVisible: grid, pointRendering, panelTitles: panelTitles.slice(0,count), panelChartIds: selectedIds,
          });
          accept(board);
          setHistory((previous) => ({ boards: [board, ...previous.boards] }));
          onSaved?.();
        }, "save")}>{t("Save new layout revision", "保存新的布局版本")}</Button>
          </Stack>
        </AccordionDetails>
      </Accordion>
    </Stack>
    <Paper id="panel-inspector" component="aside" variant="outlined" sx={{ p: 2, display: "none", position: "sticky", top: overview ? 104 : 56, minWidth: 0, "@media (min-width:1400px)": { display: "block" } }} aria-label={t("Selected panel inspector", "选中面板检查器")}>
      <Stack spacing={2}>
        <Typography component="h3" variant="h6">{t("Selected panel", "选中面板")} {"ABCD"[inspectedIndex]}</Typography>
        {overview ? <Tabs value={inspectorTab} onChange={(_, value: string) => setInspectorTab(value)} variant="fullWidth" aria-label={t("Panel inspector sections", "面板设置分区")}>
          <Tab value="data" aria-label={t("Panel data", "面板数据")} label={t("Data", "数据")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
          <Tab value="method" aria-label={t("Panel method", "面板方法")} label={t("Method", "方法")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
          <Tab value="appearance" aria-label={t("Panel appearance", "面板外观")} label={t("Appearance", "外观")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
        </Tabs> : <TextField select label={t("Inspector section", "设置分区")} value={inspectorTab} onChange={(e) => setInspectorTab(e.target.value)}>
          <MenuItem value="data">{t("Data", "数据")}</MenuItem><MenuItem value="method">{t("Method", "方法")}</MenuItem><MenuItem value="appearance">{t("Appearance", "外观")}</MenuItem>
        </TextField>}
        {inspectorTab === "data" && <>
          <TextField select label={t("Panel to inspect", "检查面板")} value={inspectedIndex} onChange={(e) => inspectIndex(Number(e.target.value))}>
            {inspectedIds.map((_, index) => <MenuItem key={index} value={index}>{"ABCD"[index]}</MenuItem>)}
          </TextField>
          <Typography variant="caption">{inspectingPreview ? t("Saved preview binding", "已保存预览绑定") : dirty ? t("Draft panel binding", "草稿面板绑定") : t("Saved panel binding", "已保存面板绑定")}</Typography>
          {inspectingPreview && <Typography variant="caption" sx={{ overflowWrap: "anywhere" }}>{t("Preview revision", "预览修订")}: {inspectedBoard?.id}</Typography>}
          <Typography sx={{ overflowWrap: "anywhere" }}>{t("Source chart", "来源图表")}: {inspected?.spec.title || (inspected ? names[inspected.spec.kind] : "—")}</Typography>
          <Typography variant="caption" sx={{ overflowWrap: "anywhere" }}>{inspected?.id}</Typography>
          {inspectedMetadata && <Typography variant="caption">{t("Observations", "观测")}: {inspectedMetadata.observations} · {t("Independent units", "独立单位")}: {inspectedMetadata.independentUnits ?? t("Undetermined", "未确定")}</Typography>}
          <Typography sx={{ overflowWrap: "anywhere" }}>{t("Bound revision", "绑定版本")}: {inspected?.revisionId ?? "—"}</Typography><Typography>{t("Changing a binding never modifies the saved source chart.", "调整绑定不会修改已保存的来源图表。")}</Typography></>}
        {inspectorTab === "method" && <><Typography>{t("Chart method", "图表方法")}: {inspected ? names[inspected.spec.kind] : "—"}. {t("Parameters belong to the saved chart. Create a new chart in Analysis to change them.", "参数属于已保存图表，修改方法请在分析任务中生成新图表。")}</Typography>{onSelectAnalysis && <Button onClick={() => { setInspectorOpen(false); onSelectAnalysis(); }}>{t("Open Analysis to create a new version", "前往分析任务生成新版本")}</Button>}</>}
        {inspectorTab === "appearance" && <Typography>{t("Palette, font, grid and axis sharing apply to the whole figure. Preview and exports use the saved revision until you save.", "配色、字号、网格与坐标共享作用于整个组合图。保存前，预览及导出仍使用已保存版本。")}</Typography>}
        <Typography variant="caption">{t("Clear removes a panel binding only. Undo restores its previous binding and layout.", "清空只移除面板绑定，撤销可恢复此前绑定和布局。")}</Typography>
      </Stack>
    </Paper>
    <Drawer slotProps={{ paper: { role: "dialog", "aria-label": t("Selected panel inspector", "选中面板检查器"), "aria-modal": true } }} anchor="right" open={inspectorOpen && !wide} onClose={() => setInspectorOpen(false)} sx={{ "@media (min-width:1400px)": { display: "none" } }}>
      <Stack sx={{ width: { xs: "min(340px, 90vw)", sm: 340 }, p: 2 }} spacing={2}>
        <Button onClick={() => setInspectorOpen(false)}>{t("Return to figure", "返回组合图")}</Button>
      <Stack spacing={2}>
        <Typography component="h3" variant="h6">{t("Selected panel", "选中面板")} {"ABCD"[inspectedIndex]}</Typography>
        {overview ? <Tabs value={inspectorTab} onChange={(_, value: string) => setInspectorTab(value)} variant="fullWidth" aria-label={t("Panel inspector sections", "面板设置分区")}>
          <Tab value="data" aria-label={t("Panel data", "面板数据")} label={t("Data", "数据")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
          <Tab value="method" aria-label={t("Panel method", "面板方法")} label={t("Method", "方法")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
          <Tab value="appearance" aria-label={t("Panel appearance", "面板外观")} label={t("Appearance", "外观")} sx={{ minWidth: 0, px: 0.5, fontSize: 12 }} />
        </Tabs> : <TextField select label={t("Inspector section", "设置分区")} value={inspectorTab} onChange={(e) => setInspectorTab(e.target.value)}>
          <MenuItem value="data">{t("Data", "数据")}</MenuItem><MenuItem value="method">{t("Method", "方法")}</MenuItem><MenuItem value="appearance">{t("Appearance", "外观")}</MenuItem>
        </TextField>}
        {inspectorTab === "data" && <>
          <TextField select label={t("Panel to inspect", "检查面板")} value={inspectedIndex} onChange={(e) => inspectIndex(Number(e.target.value))}>
            {inspectedIds.map((_, index) => <MenuItem key={index} value={index}>{"ABCD"[index]}</MenuItem>)}
          </TextField>
          <Typography variant="caption">{inspectingPreview ? t("Saved preview binding", "已保存预览绑定") : dirty ? t("Draft panel binding", "草稿面板绑定") : t("Saved panel binding", "已保存面板绑定")}</Typography>
          {inspectingPreview && <Typography variant="caption" sx={{ overflowWrap: "anywhere" }}>{t("Preview revision", "预览修订")}: {inspectedBoard?.id}</Typography>}
          <Typography sx={{ overflowWrap: "anywhere" }}>{t("Source chart", "来源图表")}: {inspected?.spec.title || (inspected ? names[inspected.spec.kind] : "—")}</Typography>
          <Typography variant="caption" sx={{ overflowWrap: "anywhere" }}>{inspected?.id}</Typography>
          {inspectedMetadata && <Typography variant="caption">{t("Observations", "观测")}: {inspectedMetadata.observations} · {t("Independent units", "独立单位")}: {inspectedMetadata.independentUnits ?? t("Undetermined", "未确定")}</Typography>}
          <Typography sx={{ overflowWrap: "anywhere" }}>{t("Bound revision", "绑定版本")}: {inspected?.revisionId ?? "—"}</Typography><Typography>{t("Changing a binding never modifies the saved source chart.", "调整绑定不会修改已保存的来源图表。")}</Typography></>}
        {inspectorTab === "method" && <><Typography>{t("Chart method", "图表方法")}: {inspected ? names[inspected.spec.kind] : "—"}. {t("Parameters belong to the saved chart. Create a new chart in Analysis to change them.", "参数属于已保存图表，修改方法请在分析任务中生成新图表。")}</Typography>{onSelectAnalysis && <Button onClick={() => { setInspectorOpen(false); onSelectAnalysis(); }}>{t("Open Analysis to create a new version", "前往分析任务生成新版本")}</Button>}</>}
        {inspectorTab === "appearance" && <Typography>{t("Palette, font, grid and axis sharing apply to the whole figure. Preview and exports use the saved revision until you save.", "配色、字号、网格与坐标共享作用于整个组合图。保存前，预览及导出仍使用已保存版本。")}</Typography>}
        <Typography variant="caption">{t("Clear removes a panel binding only. Undo restores its previous binding and layout.", "清空只移除面板绑定，撤销可恢复此前绑定和布局。")}</Typography>
      </Stack>
      </Stack>
    </Drawer>
    </Box>
  </Paper>;
}
