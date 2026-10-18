"use client";

import {
  Accordion, AccordionDetails, AccordionSummary, Alert, Autocomplete, Box, Button,
  Chip, Drawer, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, Tabs, Tab, TextField, Typography,
} from "@mui/material";
import { useLocale } from "next-intl";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import type { PreparationPlan, ResearchPage, ResearchPreview, ResearchView } from "@/domain/research-preparation";
import { researchApi } from "@/lib/api/research-api";

import { rememberResearchNavigation, type ResearchTask } from "./research-navigation";
import { ResearchProjectList } from "./research-project-list";
import { FigureStatus } from "./figure-status";
import type { FigureWorkspaceState } from "./figure-workspace-state";
import { OverviewProjectContents } from "./overview-project-contents";
import { useInterfaceMode } from "@/components/layout/interface-mode";
import { useWorkspaceStore } from "./workspace-store";

const SourceDataTools = dynamic(() => import("./source-data-tools").then((module) => module.SourceDataTools));

const ResearchChartWorkspace = dynamic(() => import("./research-chart-workspace").then((module) => module.ResearchChartWorkspace));
const ResearchReuseWorkspace = dynamic(() => import("./research-reuse-workspace").then((module) => module.ResearchReuseWorkspace));
const ResearchFigureWorkspace = dynamic(() => import("./research-figure-workspace").then((module) => module.ResearchFigureWorkspace));

const emptyPlan: PreparationPlan = {
  mapping: { value: "", defaultUnit: "unspecified" }, steps: [], exclusions: [],
  analysisUnit: "observation", validatePairs: false, pairTimes: [],
};
const roles = [
  ["value", "Measurement value", "测量值"], ["sample", "Sample ID", "样本编号"],
  ["experimentalUnit", "Independent experimental unit", "独立实验单位"],
  ["technicalRepeat", "Technical repeat", "技术重复编号"], ["unit", "Unit column", "单位列"],
  ["state", "Measurement state", "测量状态"], ["group", "Group", "分组"],
  ["parentSample", "Parent sample", "母样本编号"], ["time", "Time point", "时间点"],
  ["batch", "Batch", "批次"], ["pair", "Pair key", "配对键"], ["limit", "Limit value", "检出／定量限"],
  ["reported", "Reported text", "原始报告文本"], ["measurementType", "Measurement type", "测量类型"],
  ["role", "Sample / blank / standard / control", "样本／空白／标准／对照"],
  ["notes", "Notes", "实验备注"], ["recordLink", "Local record reference", "本地实验记录引用"],
] as const;
const states: Record<string, [string, string]> = {
  observed: ["Observed", "已测得"], missing: ["Missing", "缺失"],
  not_measured: ["Not measured", "未测量"], failed: ["Failed", "测量失败"],
  below_detection: ["Below detection", "低于检出限"],
  below_quantification: ["Below quantification", "低于定量限"],
  out_of_range: ["Out of range", "超量程"], saturated: ["Saturated", "饱和"],
};
const splitLines = (value: string) => value.split("\n").map((part) => part.trim()).filter(Boolean);

export function ResearchWorkspace({ projectId, active = true, initialTask = "data", onFigureStateChange, onTaskChange }: { projectId: string; active?: boolean; initialTask?: ResearchTask; onFigureStateChange?: (state: FigureWorkspaceState) => void; onTaskChange?: (task: ResearchTask) => void }) {
  const zh = useLocale().startsWith("zh");
  const overview = useInterfaceMode().mode === "overview";
  const importedFile = useWorkspaceStore((state) => state.selectedFile?.sourceFile);
  const text = (en: string, cn: string) => zh ? cn : en;
  const previewAbort = useRef<AbortController | null>(null);
  const [previewRunning, setPreviewRunning] = useState(false);
  const [cancelled, setCancelled] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [operation, setOperation] = useState<"loading" | "computing" | "saving">("loading");
  const [projectsOpen, setProjectsOpen] = useState(false);
  const [projectsCollapsed, setProjectsCollapsed] = useState(false);
  const [figureState, setFigureState] = useState<FigureWorkspaceState | null>(null);
  const [task, setTask] = useState<ResearchTask>(initialTask);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});
  const [openedTasks, setOpenedTasks] = useState<Set<string>>(() => new Set([initialTask]));
  const [rawValues, setRawValues] = useState(false);
  const [rowSearch, setRowSearch] = useState("");
  const [chartGeneration, setChartGeneration] = useState(0);
  const [figureGeneration, setFigureGeneration] = useState(0);
  const [reuseGeneration, setReuseGeneration] = useState(0);
  const [view, setView] = useState<ResearchView | null>(null);
  const [preview, setPreview] = useState<ResearchPreview | null>(null);
  const [plan, setPlan] = useState<PreparationPlan>(emptyPlan);
  const [files, setFiles] = useState<File[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [sourceRows, setSourceRows] = useState<number | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [kind, setKind] = useState("filter");
  const [field, setField] = useState("");
  const [values, setValues] = useState("");
  const [nameField, setNameField] = useState("measurement");
  const [valueField, setValueField] = useState("reading");
  const [targetUnit, setTargetUnit] = useState("V");
  const [selected, setSelected] = useState("");
  const [reason, setReason] = useState("");
  const [pages, setPages] = useState<Partial<Record<"samples" | "observations", ResearchPage & { key: string }>>>({});


  useEffect(() => {
    let active = true;
    researchApi.get(projectId).then((result) => {
      if (!active) return;
      setView(result);
      if (result.data) {
        setPlan(result.data.plan as PreparationPlan);
        setColumns(result.data.columns);
      }
    }).catch((cause: unknown) => {
      if (active) setError(cause instanceof Error ? cause.message : "Research load failed.");
    }).finally(() => { if (active) setBusy(false); });
    return () => { active = false; previewAbort.current?.abort(); };
  }, [projectId]);

  useEffect(() => { onTaskChange?.(task); }, [task, onTaskChange]);
  useEffect(() => { if (figureState) onFigureStateChange?.(figureState); }, [figureState, onFigureStateChange]);

  function selectTask(value: ResearchTask) {
    setOpenedTasks((old) => new Set(old).add(value));
    setTask(value);
    rememberResearchNavigation(true, value);
  }
  function inspectSources(chosen: File[]) {
    if (!chosen.length) return;
    setPreview(null); setDirty(true);
    void run(async () => {
      const inspected = await researchApi.inspect(projectId, chosen);
      setFiles(chosen); setColumns(inspected.columns); setSourceRows(inspected.rows);
    });
  }

  async function run(action: () => Promise<void>, phase: "loading" | "computing" | "saving" = "loading") {
    setOperation(phase);
    setBusy(true); setError(null);
    try { await action(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Research request failed."); }
    finally { setBusy(false); }
  }
  function edit(next: PreparationPlan) {
    setCancelled(false);
    setPlan(next); setPreview(null); setDirty(true);
  }
  function accept(next: ResearchView) {
    setCancelled(false);
    setView(next); setPreview(null); setFiles([]); setDirty(false); setSelected("");
    if (next.data) {
      setPlan(next.data.plan as PreparationPlan);
      setColumns(next.data.columns);
    }
  }
  const data = preview?.data ?? view?.data;
  const pageKey = preview?.token ?? view?.revisionId ?? "";
  const observationPage = pages.observations?.key === pageKey ? pages.observations : null;
  const samplePage = pages.samples?.key === pageKey ? pages.samples : null;
  const pageObservations = observationPage?.observations ?? data?.observations ?? [];
  const visibleObservations = pageObservations.filter((row) => !rowSearch || JSON.stringify(row).toLocaleLowerCase().includes(rowSearch.toLocaleLowerCase()));
  const visibleSamples = samplePage?.samples ?? data?.samples ?? [];
  function pageControls(kind: "samples" | "observations", total: number) {
    const current = kind === "samples" ? samplePage : observationPage;
    const offset = current?.offset ?? 0;
    const size = kind === "samples" ? 200 : 100;
    const move = (next: number) => void run(async () => {
      const result = await researchApi.page(projectId, kind, next, view?.revisionId ?? null, preview?.token);
      setPages((previous) => ({ ...previous, [kind]: { ...result, key: pageKey } }));
      setSelected("");
    });
    return <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
      <Button disabled={busy || offset === 0} onClick={() => move(Math.max(0, offset - size))}>{text("Previous page", "上一页")}</Button>
      <Typography variant="body2">{total ? offset + 1 : 0}–{Math.min(offset + size, total)} / {total}</Typography>
      <Button disabled={busy || offset + size >= total} onClick={() => move(offset + size)}>{text("Next page", "下一页")}</Button>
    </Stack>;
  }

  const allColumns = [...new Set([...columns, ...(data?.columns ?? []), ...plan.steps.flatMap((step) =>
    step.kind === "wide_to_long" ? [String(step.nameField), String(step.valueField)] : [])])];
  const roleInput = ([key, en, cn]: typeof roles[number]) => (
    <Autocomplete key={key} freeSolo options={allColumns}
      inputValue={plan.mapping[key] ?? ""}
      onInputChange={(_, value) => edit({ ...plan, mapping: { ...plan.mapping, [key]: value || null } })}
      renderInput={(params) => <TextField {...params} label={text(en, cn)} size="small" />}
    />
  );

  function addStep() {
    let step: Record<string, unknown>;
    if (kind === "filter") step = { kind, field, values: splitLines(values), keep: true };
    else if (kind === "convert_unit") step = {
      kind, valueField: plan.mapping.value, unitField: plan.mapping.unit, toUnit: targetUnit,
      limitField: plan.mapping.limit ?? null,
    };
    else if (kind === "wide_to_long") step = { kind, fields: splitLines(values), nameField, valueField };
    else step = { kind, indexFields: splitLines(values), nameField, valueField };
    edit({ ...plan, steps: [...plan.steps, step] });
  }

  return (
    <Stack spacing={2} data-testid="research-workspace" sx={{ minWidth: 0, "& .MuiButton-root": { minHeight: 44 }, "@media (prefers-reduced-motion: reduce)": { "& *, & *::before, & *::after": { transition: "none !important", animation: "none !important" } } }}>
      <Typography component={overview ? "h2" : "h1"} variant="h5" sx={overview ? { position: "absolute", width: "1px", height: "1px", m: "-1px", p: 0, overflow: "hidden", clipPath: "inset(50%)", whiteSpace: "nowrap" } : undefined}>{text("Research workspace", "研究工作台")}</Typography>
      <Alert severity="info" sx={overview ? { display: task === "figures" ? "none" : undefined, py: 0, bgcolor: "transparent", "& .MuiAlert-message": { fontSize: 12 } } : undefined}>
        {text("Research revisions belong to this project. Quick plot continues to show the original imported file.",
          "研究版本保存在当前项目中；切回快速出图时，仍显示原始导入文件。")}
      </Alert>
      <Paper variant="outlined" sx={{ position: "sticky", top: 0, zIndex: 2, bgcolor: "background.paper", ...(overview && { borderRadius: 0, borderWidth: "0 0 1px" }) }}>
        <Tabs value={task} onChange={(_, value: ResearchTask) => selectTask(value)} variant="scrollable" scrollButtons="auto" aria-label={text("Research tasks", "研究任务")} sx={overview ? { "& .MuiTabs-list": { justifyContent: { sm: "center" } } } : undefined}>
          {[["data", "Data", "数据"], ["quality", "Quality", "质量"], ["analysis", "Analysis", "分析"], ["figures", "Figures", "图表"], ["outputs", "Outputs", "输出"]].map(([id,en,cn]) =>
            <Tab key={id} value={id} id={`research-tab-${id}`} aria-controls={`research-region-${id}`} label={text(en,cn)} />)}
        </Tabs>
        {overview && task === "figures" && figureState && <Box data-testid="figure-status-sticky" sx={{ px: 1.5, pb: 1 }}><FigureStatus state={figureState} compact /></Box>}
      </Paper>
      <Button aria-expanded={projectsOpen} onClick={() => setProjectsOpen(true)} sx={{ "@media (min-width:1400px)": { display: "none" } }}>{overview ? text("Project contents", "项目内容") : text("Project list", "项目列表")}</Button>
      <Drawer slotProps={{ paper: { role: "dialog", "aria-label": text("Project contents", "项目内容"), "aria-modal": true } }} open={projectsOpen} onClose={() => setProjectsOpen(false)}>
        <Stack spacing={2} sx={{ width: 280, p: 2 }}>
          <Button onClick={() => setProjectsOpen(false)}>{text("Return to active task", "返回当前任务")}</Button>
          {overview ? <OverviewProjectContents data={data ?? null} revisionId={view?.revisionId ?? null} preview={Boolean(preview)} figureState={figureState} onSelectTask={(next) => { selectTask(next); setProjectsOpen(false); }} /> : <ResearchProjectList projectId={projectId} />}
        </Stack>
      </Drawer>
      <Box sx={{ display: "grid", gridTemplateColumns: "minmax(0,1fr)", gap: 2, alignItems: "start", "@media (min-width:1400px)": { gridTemplateColumns: `${projectsCollapsed ? "64px" : overview ? "220px" : "180px"} minmax(0,1fr)` } }}>
        <Paper component="aside" variant="outlined" aria-label={overview ? text("Project contents", "项目内容") : text("Project list", "项目列表")} sx={{ p: 1, position: "sticky", top: overview ? 104 : 56, display: "none", "@media (min-width:1400px)": { display: "block" } }}>
          <Button aria-expanded={!projectsCollapsed} onClick={() => setProjectsCollapsed(!projectsCollapsed)}>{projectsCollapsed ? text("Open", "展开") : text("Collapse", "收起")}</Button>
          {!projectsCollapsed && (overview ? <OverviewProjectContents data={data ?? null} revisionId={view?.revisionId ?? null} preview={Boolean(preview)} figureState={figureState} onSelectTask={selectTask} /> : <ResearchProjectList projectId={projectId} />)}
        </Paper>
        <Stack spacing={2} sx={{ minWidth: 0 }}>
      {error && <Alert severity="error" action={
        <Button onClick={() => void run(async () => { const latest = await researchApi.get(projectId); setView(latest); if (!dirty && !preview) accept(latest); })}>
          {text("Reload saved status", "刷新已保存状态")}
        </Button>
      }>{error}</Alert>}
      {view && !view.supported ? <Alert severity="info">
        {text("Research preparation currently requires the local SQLite profile.", "研究数据准备目前适用于本地 SQLite 版本。")}
      </Alert> : <>
      <Box hidden={overview && task === "figures"}>
      <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
        <Chip role="status" label={cancelled ? text("Preview cancelled — draft retained", "已取消预览：已保留草稿") : error ? text("Save or computation failed — draft retained", "保存或计算失败：已保留草稿") : busy ? operation === "saving" ? text("Saving locally…", "正在保存到本地…") : operation === "computing" ? text("Computing…", "正在计算…") : text("Loading…", "正在加载…") : preview ? text("Preview — not applied", "预览：尚未应用")
          : dirty ? text("Changes need preview", "修改后请重新预览") : text("Saved state", "已保存状态")} />
        {view?.revisionId && <Chip label={text("Revision ", "版本 ") + view.revisionId.slice(0, 8)} />}
        <Button disabled={busy || !view?.canUndo || dirty} onClick={() => void run(async () =>
          accept(await researchApi.history(projectId, "undo", view!.revisionId!)))}>
          {text("Undo research change", "撤销研究修改")}
        </Button>
        <Button disabled={busy || !view?.canRedo || dirty} onClick={() => void run(async () =>
          accept(await researchApi.history(projectId, "redo", view!.revisionId!)))}>
          {text("Redo", "重做")}
        </Button>
      </Stack>
      <Button onClick={() => setDetailsOpen(true)}>{text("Save and computation details", "保存与计算详情")}</Button>
      <Drawer anchor="right" open={detailsOpen} onClose={() => setDetailsOpen(false)}>
        <Stack spacing={2} sx={{ width: "min(360px, 90vw)", p: 2, overflowWrap: "anywhere" }}>
          <Button onClick={() => setDetailsOpen(false)}>{text("Return to active task", "返回当前任务")}</Button>
          <Typography component="h2" variant="h6">{text("Save and computation details", "保存与计算详情")}</Typography>
          <Typography>{text("Applied revision", "已应用版本")}: {view?.revisionId ?? text("None", "暂无")}</Typography>
          <Typography>{text("Draft", "草稿")}: {preview ? text("Preview only; apply to save", "仅预览，应用后保存") : dirty ? text("Needs preview", "需要重新预览") : text("No pending preparation changes", "没有待应用的数据准备修改")}</Typography>
          <Typography>{text("Preview cancellation prevents applying its result. Server work may finish. Saved charts, figures and outputs retain their own revisions.", "取消预览会阻止应用其结果，服务端可能继续计算。已保存图表、组合图和输出保留各自的版本。")}</Typography>
          {error && <Alert severity="error">{error}</Alert>}
          {detailsOpen && <SourceDataTools />}
          <Typography variant="caption">{text("Reloading saved status preserves the current draft. Discarding is a separate action.", "刷新已保存状态会保留当前草稿，放弃草稿是单独的操作。")}</Typography>
        </Stack>
      </Drawer>
      {previewRunning && <Button onClick={() => { previewAbort.current?.abort(); setCancelled(true); }}>{text("Cancel preparation preview", "取消数据准备预览")}</Button>}
      {cancelled && <Typography variant="caption">{text("No data revision was applied. Server processing may finish its discarded preview.", "未应用数据版本。服务端可能继续完成已放弃的预览计算。")}</Typography>}
      {(dirty || preview) && <Button disabled={busy} onClick={() => { if (view) accept(view); }}>{text("Discard preparation draft", "放弃数据准备草稿")}</Button>}
      </Box>
      <Box role="tabpanel" id="research-region-data" aria-labelledby="research-tab-data" hidden={task !== "data"}>
      {active && task === "data" && <Box component="fieldset" disabled={busy} sx={{ m: 0, p: 0, border: 0, minWidth: 0 }}>
        <Stack spacing={2}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography component="h2" variant="h6">{text("1. Sources and field roles", "1. 数据来源与字段含义")}</Typography>
              <Typography variant="body2">{text(
                "Choose 1–5 UTF-8 CSV/TSV files with identical headers. Maximum: 50 MiB each, 100 MiB and 100,000 rows combined. Blank identity fields remain unknown.",
                "选择 1–5 个表头一致的 UTF-8 CSV/TSV 文件；单文件不超过 50 MiB，总计不超过 100 MiB、10 万行。未指定的样本身份保持未知。")}</Typography>
              {overview && importedFile && /\.(csv|tsv)$/i.test(importedFile.name) && <Button variant="outlined" disabled={busy} onClick={() => inspectSources([importedFile])}>
                {text("Use the file just imported", "使用刚导入的文件")}: {importedFile.name}
              </Button>}
              <Button component="label" variant="outlined">
                {text("Choose research files", "选择研究数据文件")}
                <input hidden multiple type="file" accept=".csv,.tsv"
                  aria-label={text("Research source files", "研究数据文件")}
                  onChange={(event) => {
                    const chosen = Array.from(event.target.files ?? []);
                    event.target.value = "";
                    if (!chosen.length) return;
                    inspectSources(chosen);
                  }} />
              </Button>
              <Typography variant="body2">
                {files.length ? files.map((file) => file.name).join(" + ")
                  : data?.sources.map((source) => source.name).join(" + ") ?? text("No research files yet.", "尚未选择研究文件。")}
                {sourceRows !== null && ` · ${sourceRows} ${text("source rows", "原始行")}`}
              </Typography>
              <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr 1fr" }, gap: 2 }}>
                {roles.slice(0, 6).map(roleInput)}
              </Box>
              <Accordion expanded={expandedSections.metadata ?? false} onChange={(_, value) => setExpandedSections((old) => ({ ...old, metadata: value }))}>
                <AccordionSummary>{text("More metadata, limits and experimental records", "分组、配对、检出限与实验记录")}</AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr 1fr" }, gap: 2 }}>
                    {roles.slice(6).map(roleInput)}
                    <TextField label={text("Unit when no unit column is mapped", "未映射单位列时使用的单位")}
                      value={plan.mapping.defaultUnit ?? "unspecified"}
                      onChange={(event) => edit({ ...plan, mapping: { ...plan.mapping, defaultUnit: event.target.value } })} />
                  </Box>
                </AccordionDetails>
              </Accordion>
              <TextField select label={text("Analysis unit", "分析单位")} value={plan.analysisUnit}
                onChange={(event) => edit({ ...plan, analysisUnit: event.target.value as PreparationPlan["analysisUnit"] })}>
                <MenuItem value="observation">{text("Observation — each reading", "观测值：每次读数")}</MenuItem>
                <MenuItem value="sample">{text("Sample — mean of declared technical repeats", "样本：先汇总已声明的技术重复")}</MenuItem>
                <MenuItem value="experimental_unit">{text("Independent experimental unit", "独立实验单位")}</MenuItem>
              </TextField>
            </Stack>
          </Paper>
          <Accordion expanded={expandedSections.preparation ?? false} onChange={(_, value) => setExpandedSections((old) => ({ ...old, preparation: value }))}>
            <AccordionSummary>{text("2. Preparation steps and explicit pairing", "2. 数据准备步骤与显式配对")}</AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
                {plan.steps.map((step, index) => <Stack key={index} direction="row" spacing={2} sx={{ alignItems: "center" }}>
                  <Typography>{index + 1}. {String(step.kind)} {String(step.field ?? step.toUnit ?? "")}</Typography>
                  <Button onClick={() => edit({ ...plan, steps: plan.steps.filter((_, at) => at !== index) })}>{text("Remove", "移除")}</Button>
                </Stack>)}
                <TextField select label={text("Operation", "处理操作")} value={kind} onChange={(event) => setKind(event.target.value)}>
                  <MenuItem value="filter">{text("Keep selected field values", "保留指定字段值")}</MenuItem>
                  <MenuItem value="convert_unit">{text("Convert mapped value, unit and limit", "换算已映射的测量值、单位及检出限")}</MenuItem>
                  <MenuItem value="wide_to_long">{text("Wide to long", "宽表转长表")}</MenuItem>
                  <MenuItem value="long_to_wide">{text("Long to wide", "长表转宽表")}</MenuItem>
                </TextField>
                {kind === "filter" && <TextField label={text("Filter field", "筛选字段")} value={field} onChange={(e) => setField(e.target.value)} />}
                {kind !== "convert_unit" && <TextField multiline minRows={2}
                  label={kind === "filter" ? text("Values to keep — one per line", "保留的值：每行一个")
                    : kind === "wide_to_long" ? text("Measurement columns — one per line", "测量列名：每行一个")
                    : text("Identifier columns — one per line", "标识列名：每行一个")}
                  value={values} onChange={(e) => setValues(e.target.value)} />}
                {(kind === "wide_to_long" || kind === "long_to_wide") && <>
                  <TextField label={text("Measurement-name field", "测量名称字段")} value={nameField} onChange={(e) => setNameField(e.target.value)} />
                  <TextField label={text("Measurement-value field", "测量数值字段")} value={valueField} onChange={(e) => setValueField(e.target.value)} />
                </>}
                {kind === "convert_unit" && <TextField select label={text("Target unit", "目标单位")} value={targetUnit} onChange={(e) => setTargetUnit(e.target.value)}>
                  {["V", "mV", "uV", "s", "min", "h", "g", "mg", "ug", "L", "mL", "uL", "mol/L", "mmol/L", "umol/L"]
                    .map((unit) => <MenuItem key={unit} value={unit}>{unit}</MenuItem>)}
                </TextField>}
                <Button variant="outlined" onClick={addStep}>{text("Add preparation step", "添加准备步骤")}</Button>
                <TextField select label={text("Pair validation", "配对校验")} value={plan.validatePairs ? "yes" : "no"}
                  onChange={(e) => edit({ ...plan, validatePairs: e.target.value === "yes" })}>
                  <MenuItem value="no">{text("No paired calculation", "不进行配对计算")}</MenuItem>
                  <MenuItem value="yes">{text("Explicit sample + pair + time keys", "使用显式样本、配对键和时间点")}</MenuItem>
                </TextField>
                {plan.validatePairs && [0, 1].map((index) => <TextField key={index}
                  label={index === 0 ? text("First time label", "起始时间标签") : text("Second time label (second minus first)", "结束时间标签（结束减起始）")}
                  value={plan.pairTimes[index] ?? ""} onChange={(e) => {
                    const times = [...plan.pairTimes]; times[index] = e.target.value;
                    edit({ ...plan, pairTimes: times });
                  }} />)}
              </Stack>
            </AccordionDetails>
          </Accordion>
          <Stack direction="row" spacing={2}>
            <Button variant="outlined" disabled={!plan.mapping.value || (!files.length && !view?.data)}
              onClick={() => void run(async () => {
                const controller = new AbortController(); previewAbort.current = controller;
                setPreviewRunning(true); setCancelled(false);
                try {
                  const result = await researchApi.preview(projectId, plan, files, view?.revisionId ?? null, controller.signal);
                  if (!controller.signal.aborted) { setPreview(result); selectTask("quality"); setDirty(false); }
                } catch (cause) { if (!controller.signal.aborted) throw cause; }
                finally { previewAbort.current = null; setPreviewRunning(false); }
              }, "computing")}>{text("Preview preparation", "预览数据准备")}</Button>
            <Button variant="contained" disabled={!preview}
              onClick={() => void run(async () => accept(await researchApi.apply(projectId, preview!.token, preview!.expectedRevision)), "saving")}>
              {text("Apply preview and save revision", "应用预览并保存版本")}
            </Button>
          </Stack>
        </Stack>
      </Box>}
      </Box>
      <Box role="tabpanel" id="research-region-quality" aria-labelledby="research-tab-quality" hidden={task !== "quality"}>
      <Stack spacing={2}>
      {!data && <Alert severity="info">{text("Apply or preview data in the Data task to inspect quality.", "请先在数据任务中预览或应用数据，再检查质量。")}</Alert>}
      {preview && <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: "wrap" }}>
        <Button variant="contained" disabled={busy} onClick={() => void run(async () => accept(await researchApi.apply(projectId, preview.token, preview.expectedRevision)), "saving")}>{text("Apply preview and save revision", "应用预览并保存版本")}</Button>
        <Button disabled={busy} onClick={() => selectTask("data")}>{text("Review field mapping", "返回检查字段映射")}</Button>
      </Stack>}
      {active && task === "quality" && data && <>
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Stack spacing={1}>
            <Typography component="h2" variant="h6">{text("Samples, observations and QC", "样本、观测与 QC")}</Typography>
            <Stack direction="row" useFlexGap sx={{ flexWrap: "wrap", gap: 1 }}>
              {roles.filter(([key]) => (data.plan as PreparationPlan).mapping[key]).map(([key,en,cn]) => <Chip key={key} label={`${text(en,cn)}: ${(data.plan as PreparationPlan).mapping[key]}`} />)}
            </Stack>
            <Typography data-testid="research-counts">
              {text("Independent units", "独立实验单位")}: {data.summary.independentUnits ?? text("unknown", "未知")}
              {" · "}{text("Samples", "样本")}: {data.summary.samples}
              {" · "}{text("Observations", "观测值")}: {data.summary.observations}
              {" · "}{text("Excluded", "已排除")}: {data.summary.excluded}
            </Typography>
            <Typography variant="body2">
              {text("Known units / unknown identities", "已知独立单位／身份未知观测")}: {data.summary.knownIndependentUnits} / {data.summary.unknownIdentity}
              {" · "}{text("Included observed readings", "纳入的已测得读数")}: {data.summary.includedObserved}
            </Typography>
            <Stack direction="row" sx={{ flexWrap: "wrap", gap: 1 }}>
              {Object.entries(data.summary.states).map(([state, count]) => <Chip key={state}
                label={`${states[state]?.[zh ? 1 : 0] ?? state}: ${count}`} />)}
            </Stack>
            {data.summary.withheldReasons.length ? <Alert severity="warning">
              {text("Summary withheld: ", "暂不计算汇总：")}{data.summary.withheldReasons.join(" ")}
            </Alert> : <Typography data-testid="research-mean">
              {text("Descriptive mean", "描述性均值")}: {data.summary.mean === null ? "—" : Number(data.summary.mean.toPrecision(8))}
              {" · "}{text("Analysis n", "分析 n")}: {data.summary.analysisN ?? "—"}
            </Typography>}
            {data.history.map((entry, index) => <Typography key={index} variant="body2">
              {index + 1}. {String(entry.step.kind)}: {entry.inputRows} → {entry.outputRows} {text("rows", "行")}
            </Typography>)}
          </Stack>
        </Paper>
        <Accordion expanded={expandedSections.samples ?? false} onChange={(_, value) => setExpandedSections((old) => ({ ...old, samples: value }))}>
          <AccordionSummary>{text("Sample inventory", "样本清单")} ({visibleSamples.length}/{data.summary.samples})</AccordionSummary>
          <AccordionDetails>
            {pageControls("samples", data.summary.samples)}
            <TableContainer><Table size="small" sx={{ minWidth: 600 }}>
              <TableHead><TableRow>
                {[text("Sample", "样本"), text("Independent unit", "独立单位"), text("Parent", "母样本"), text("Group", "分组")]
                  .map((label) => <TableCell key={label}>{label}</TableCell>)}
              </TableRow></TableHead>
              <TableBody>{active && task === "quality" && visibleSamples.map((sample) => <TableRow key={sample.id}>
                <TableCell>{sample.label}</TableCell><TableCell>{sample.experimentalUnit ?? "—"}</TableCell>
                <TableCell>{sample.parentSample ?? "—"}</TableCell><TableCell>{sample.group ?? "—"}</TableCell>
              </TableRow>)}</TableBody>
            </Table></TableContainer>
          </AccordionDetails>
        </Accordion>
        <Typography variant="body2">
          {text("Displayed observations / total", "显示观测／总数")}: {visibleObservations.length}/{data.summary.observations}.
          {" "}{text("Summaries use the full prepared input, not only these displayed rows.", "汇总使用完整准备数据，而不是只使用显示行。")}
        </Typography>
        {pageControls("observations", data.summary.observations)}
        <Typography variant="caption">{text("Scroll the table horizontally to inspect all source and QC fields.", "表格可横向滚动，查看完整来源和 QC 字段。")}</Typography>
        <TextField label={text("Search displayed rows", "搜索当前页数据行")} value={rowSearch} onChange={(e) => setRowSearch(e.target.value)} helperText={text("This filters the displayed page only; n and saved data do not change.", "仅筛选当前显示页，不改变 n 或已保存数据。")}/>
        <Button aria-pressed={rawValues} onClick={() => setRawValues(!rawValues)}>{rawValues ? text("Show derived values", "显示处理后值") : text("Show reported values", "显示原始报告值")}</Button>
        <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 560, "& th:first-of-type, & td:first-of-type": { position: "sticky", left: 0, width: 120, minWidth: 120, maxWidth: 120, bgcolor: "background.paper", zIndex: 1 }, "& th:nth-of-type(2), & td:nth-of-type(2)": { position: "sticky", left: 120, bgcolor: "background.paper", zIndex: 1 } }}>
          <Table stickyHeader size="small" sx={{ minWidth: 900, tableLayout: "fixed", "& td": { overflowWrap: "anywhere" } }} aria-label={text("Research observations", "研究观测表")}>
            <colgroup><col style={{ width: 120 }} /><col style={{ width: 200 }} /><col /><col /><col /><col /></colgroup>
            <TableHead><TableRow>
              {[text("Select", "选择"), text("Sample / unit", "样本／独立单位"), text("Batch / time / repeat", "批次／时间／重复"),
                text("Value / state", "数值／状态"), text("Source cell", "来源单元格"), text("Notes / exclusion", "备注／排除原因")]
                .map((label) => <TableCell key={label}>{label}</TableCell>)}
            </TableRow></TableHead>
            <TableBody>{active && task === "quality" && visibleObservations.map((observation) => <TableRow key={observation.id} selected={selected === observation.id}>
              <TableCell><Button size="small" aria-pressed={selected === observation.id}
                onClick={() => setSelected(selected === observation.id ? "" : observation.id)}>{text("Select", "选择")}</Button></TableCell>
              <TableCell>{observation.sampleLabel ?? "—"} / {observation.experimentalUnit ?? "—"}</TableCell>
              <TableCell>{observation.batch ?? "—"} / {observation.time ?? "—"} / {observation.technicalRepeat ?? "—"}</TableCell>
              <TableCell>{rawValues ? observation.reportedText || "—" : observation.value ?? "—"} {rawValues ? observation.reportedUnit : observation.unit}<br />
                {text("Reported", "原始报告")}: {observation.reportedText || "—"} {observation.reportedUnit}<br />
                {states[observation.state]?.[zh ? 1 : 0] ?? observation.state}
                {observation.limit !== null && ` (limit ${observation.limit})`}
              </TableCell>
              <TableCell>{observation.lineage.map((cell, index) => <Box key={index}>
                {data.sources.find((source) => source.id === cell.sourceId)?.name}: {cell.row}, {cell.column}
              </Box>)}</TableCell>
              <TableCell>{observation.notes} {observation.recordLink} {observation.exclusionReason}</TableCell>
            </TableRow>)}</TableBody>
          </Table>
        </TableContainer>
        <Typography variant="body2">{text("Selection only highlights a row. Exclusion needs a reason, preview and apply.",
          "选择只高亮行；排除数据需要填写原因、预览并应用。")}</Typography>
        <Stack direction={{ xs: "column", md: "row" }} spacing={1}>
          <TextField fullWidth label={text("Exclusion reason", "排除原因")} value={reason} onChange={(e) => setReason(e.target.value)} />
          <Button disabled={busy || !selected || !reason.trim()} onClick={() => edit({
            ...plan, exclusions: [...plan.exclusions.filter((item) => item.observationId !== selected),
              { observationId: selected, reason: reason.trim() }],
          })}>{text("Exclude selected observation", "排除选中观测")}</Button>
          <Button disabled={busy || !selected} onClick={() => edit({
            ...plan, exclusions: plan.exclusions.filter((item) => item.observationId !== selected),
          })}>{text("Remove exclusion", "撤销排除")}</Button>
        </Stack>
      </>}
      </Stack></Box>
      </>}
      <Box role="tabpanel" id="research-region-analysis" aria-labelledby="research-tab-analysis" hidden={task !== "analysis"}>
      {openedTasks.has("analysis") && view?.supported && <ResearchChartWorkspace active={active && task === "analysis"} refreshKey={reuseGeneration} projectId={projectId} revisionId={view.revisionId}
        mapping={(view.data?.plan as PreparationPlan | undefined)?.mapping} pending={busy || dirty || preview !== null} onSaved={() => setChartGeneration((value) => value + 1)} />}
      </Box>
      <Box role="tabpanel" id="research-region-figures" aria-labelledby="research-tab-figures" hidden={task !== "figures"}>
      {openedTasks.has("figures") && view?.supported && <ResearchFigureWorkspace active={active && task === "figures"} projectId={projectId} currentRevision={view.revisionId}
        onStateChange={setFigureState} onSelectAnalysis={() => selectTask("analysis")} refreshKey={chartGeneration} onSaved={() => setFigureGeneration((n) => n + 1)} />}
      </Box>
      <Box role="tabpanel" id="research-region-outputs" aria-labelledby="research-tab-outputs" hidden={task !== "outputs"}>
      {openedTasks.has("outputs") && view?.supported && <ResearchReuseWorkspace projectId={projectId} revision={view.revisionId}
        pending={busy || dirty || preview !== null} refreshKey={figureGeneration} onApplied={async () => {
          accept(await researchApi.get(projectId)); setReuseGeneration((n) => n + 1); setChartGeneration((n) => n + 1);
          setFigureGeneration((n) => n + 1);
        }} />}
      </Box>
        </Stack>
      </Box>
    </Stack>
  );
}
