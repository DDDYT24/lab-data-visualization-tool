"use client";

import { useEffect, useState } from "react";
import { useLocale } from "next-intl";
import { Alert, Box, Button, Checkbox, FormControlLabel, MenuItem, Paper, Stack, TextField, Typography } from "@mui/material";
import type { FigureBoardHistory, FigureBoardSpec } from "@/domain/research-figure-board";
import { researchApi } from "@/lib/api/research-api";
import { researchFigureApi } from "@/lib/api/research-figure-api";
import { researchReuseApi as api, type TemplateRecord, type ReplacementPreview, type OutputPreview, type OutputNotes, type SavedOutput } from "@/lib/api/research-reuse-api";

export function ResearchReuseWorkspace({ projectId, revision, pending, refreshKey, onApplied }: {
  projectId: string; revision: string | null; pending: boolean; refreshKey: number; onApplied: () => Promise<void>;
}) {
  const zh = useLocale().startsWith("zh");
  const t = (en: string, cn: string) => zh ? cn : en;
  const [boards, setBoards] = useState<FigureBoardHistory["boards"]>([]);
  const [templates, setTemplates] = useState<TemplateRecord[]>([]);
  const [outputs, setOutputs] = useState<SavedOutput[]>([]);
  const [board, setBoard] = useState("");
  const [template, setTemplate] = useState("");
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [check, setCheck] = useState<ReplacementPreview | null>(null);
  const [outputCheck, setOutputCheck] = useState<OutputPreview | null>(null);
  const [notes, setNotes] = useState<OutputNotes | null>(null);
  const [reviewed, setReviewed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  useEffect(() => {
    let active = true;
    Promise.all([researchFigureApi.list(projectId), api.templates(projectId), api.outputs(projectId)]).then(([b, tp, o]) => {
      if (!active) return;
      setBoards(b.boards); setTemplates(tp.templates); setOutputs(o.outputs);
    }).catch((e: unknown) => { if (active) setError(e instanceof Error ? e.message : "Load failed"); });
    return () => { active = false; };
  }, [projectId, refreshKey]);
  async function run(action: () => Promise<void>) {
    setBusy(true); setError(""); setMessage("");
    try { await action(); } catch (e) { setError(e instanceof Error ? e.message : "Request failed"); }
    finally { setBusy(false); }
  }
  const selected = templates.find((item) => item.id === template);
  const fields = [...new Set(selected?.spec.panels.flatMap((panel) => panel.sourceColumns) ?? [])];
  const blocked = busy || pending;
  function selectTemplate(id: string) {
    setTemplate(id); setCheck(null); setMapping({});
  }
  return <Paper variant="outlined" sx={{ p: 2 }} data-testid="reuse-workspace"><Stack spacing={2}>
    <Typography component="h2" variant="h5">{t("Templates and research outputs", "模板与研究输出")}</Typography>
    <Typography>{t("Use saved figures. Templates retain settings and category labels, but exclude measurements and old row exclusions.", "基于已保存组合图操作。模板保留设置和类别标签，不包含测量数据或此前逐行排除决定。")}</Typography>
    {pending && <Alert severity="info">{t("Apply or discard the data preparation draft before replacing data.", "替换数据前，请先应用或放弃数据准备草稿。")}</Alert>}
    {error && <Alert severity="error">{error}</Alert>}
    <Typography role="status">{busy ? t("Working…", "处理中…") : message}</Typography>
    <TextField select label={t("Saved figure for template / output", "用于模板／输出的已保存组合图")} value={board} disabled={busy}
      onChange={(e) => { setBoard(e.target.value); setOutputCheck(null); setNotes(null); setReviewed(false); }}>
      <MenuItem value="">{t("Choose a saved figure", "选择已保存组合图")}</MenuItem>
      {boards.map((b) => <MenuItem key={b.id} value={b.id}>{(b.spec as FigureBoardSpec).title || t("Untitled", "未命名")} · {b.id.slice(0, 8)}</MenuItem>)}
    </TextField>
    <TextField label={t("Template name", "模板名称")} value={name} disabled={busy} slotProps={{ htmlInput: { maxLength: 100 } }} onChange={(e) => setName(e.target.value)} />
    <Button disabled={busy || !board || !name.trim()} onClick={() => void run(async () => {
      const saved = await api.create(projectId, board, name.trim()); setTemplates((items) => [saved, ...items]); selectTemplate(saved.id);
      setMessage(t("Template saved locally.", "模板已保存到本地。"));
    })}>{t("Save workflow template", "保存工作流模板")}</Button>
    <Button component="label" disabled={busy}>{t("Import template JSON", "导入模板 JSON")}<input hidden type="file" accept=".json" data-testid="template-import" disabled={busy} onChange={(e) => {
      const upload = e.target.files?.[0]; e.target.value = ""; if (!upload) return;
      void run(async () => { if (upload.size > 1_000_000) throw new Error(t("Template exceeds 1 MB.", "模板超过 1 MB。")); const saved = await api.import(projectId, JSON.parse(await upload.text())); setTemplates((items) => [saved, ...items]); selectTemplate(saved.id); });
    }} /></Button>
    <TextField select label={t("Reusable template", "可复用模板")} value={template} disabled={blocked} onChange={(e) => selectTemplate(e.target.value)}>
      <MenuItem value="">{t("Choose a template", "选择模板")}</MenuItem>
      {templates.map((item) => <MenuItem key={item.id} value={item.id}>{item.spec.name} · {item.id.slice(0, 8)}</MenuItem>)}
    </TextField>
    {selected && <Button component="a" href={api.templateUrl(projectId, selected.id)} download="workflow-template.json">{t("Download template JSON", "下载模板 JSON")}</Button>}
    <Button component="label" disabled={blocked || !selected}>{t("Choose replacement CSV / TSV", "选择替换数据 CSV／TSV")}<input hidden type="file" accept=".csv,.tsv" data-testid="replacement-file" disabled={blocked || !selected} onChange={(e) => {
      const upload = e.target.files?.[0]; e.target.value = ""; setCheck(null); setFile(null); setColumns([]); setMapping({}); if (!upload) return;
      void run(async () => { const data = await researchApi.inspect(projectId, [upload]); setColumns(data.columns); setFile(upload); });
    }} /></Button>
    {file && <Typography>{file.name}</Typography>}
    {file && fields.map((field) => <TextField key={field} select label={`${t("Map", "映射")} ${field}`} value={mapping[field] ?? (columns.includes(field) ? field : "")} disabled={blocked}
      onChange={(e) => { setMapping({ ...mapping, [field]: e.target.value }); setCheck(null); }}>
      <MenuItem value="">{t("Unresolved — dependent panel will be checked", "未解决——预览时检查依赖面板")}</MenuItem>
      {columns.filter((c) => !c.startsWith("__labviz_")).map((c) => <MenuItem key={c} value={c}>{c}</MenuItem>)}
    </TextField>)}
    <Button disabled={blocked || !file || !template} onClick={() => void run(async () => {
      setCheck(null); const explicit = Object.fromEntries(fields.map((f) => [f, mapping[f] ?? (columns.includes(f) ? f : "")]).filter(([, value]) => value));
      setCheck(await api.preview(projectId, template, file!, explicit, revision));
    })}>{t("Validate replacement", "校验替换数据")}</Button>
    {check && <Stack spacing={1} data-testid="replacement-preview">
      <Typography>{t("Old row exclusions are not copied. Review QC and source identity again.", "不会复制旧的逐行排除决定，请重新核对 QC 和样本身份。")}</Typography>
      {check.panels.map((p) => <Alert key={p.panel} severity={p.status === "ready" ? "success" : "warning"}>
        {p.panel}: {p.status} · {p.observations} · {p.unit}. {p.message}
        <Box component="pre" sx={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", fontSize: 12 }}>{JSON.stringify(p.styles, null, 2)}</Box>
      </Alert>)}
      {!check.figureReady && <Alert severity="warning">{check.figureError || t("Only ready panels will be saved; no incomplete figure will be created.", "只保存已就绪面板，不生成不完整组合图。")}</Alert>}
      <Button variant="contained" disabled={blocked || check.readyCount === 0 || check.expectedRevision !== revision} onClick={() => void run(async () => {
        await api.apply(projectId, check.token, check.expectedRevision); setCheck(null); await onApplied();
        setBoards((await researchFigureApi.list(projectId)).boards); setMessage(t("Replacement saved as new revisions.", "替换结果已保存为新版本。"));
      })}>{t("Apply validated replacement", "应用已校验替换")}</Button>
    </Stack>}
    <Typography component="h3" variant="h6">{t("Freeze research output", "冻结研究输出")}</Typography>
    <Button disabled={busy || !board} onClick={() => void run(async () => {
      setOutputCheck(null); setReviewed(false); const result = await api.previewOutput(projectId, board); setOutputCheck(result); setNotes(result.notes);
    })}>{t("Preview included output files", "预览输出内容")}</Button>
    {outputCheck && notes && <Stack spacing={2} data-testid="output-preview">
      <Alert severity="info">{t("This ZIP is a research output, not a restorable project backup. It includes identifiers, sample labels, groups, source references, exclusion reasons and your notes. Original uploads are not included.", "ZIP 是研究输出，不是可恢复项目的备份。包含标识符、样本标签、分组、来源引用、排除理由及你的说明，不包含原始上传文件。")}</Alert>
      <Typography>{t("Scientific CSV separates reported text/unit from derived values and includes exclusions. Legacy CSV contains captured chart inputs after prior exclusions. Panel counts can overlap.", "科学图表 CSV 区分原始报告文本／单位与处理后数值，保留排除记录。旧图表 CSV 包含此前排除后的冻结输入。各面板计数可能重叠。")}</Typography>
      {outputCheck.manifest.panels.map((p) => <Typography key={p.label}>{p.label}: {p.sourceCsvRows} {t("source rows", "来源行")} · {p.revisionId.slice(0, 12)}</Typography>)}
      {outputCheck.files.map((f) => <Typography variant="body2" key={f.name}>{f.name} · {f.bytes ?? t("set at freeze", "冻结时确定")}</Typography>)}
      {([["legendEn", "Figure legend (English)", "图注（英文）"], ["legendZh", "Figure legend (Chinese)", "图注（中文）"], ["methodsEn", "Methods (English)", "方法（英文）"], ["methodsZh", "Methods (Chinese)", "方法（中文）"]] as const).map(([key, en, cn]) => <TextField key={key} multiline minRows={2} label={t(en, cn)} value={notes[key]} disabled={busy} slotProps={{ htmlInput: { maxLength: 20000 } }} onChange={(e) => { setNotes({ ...notes, [key]: e.target.value }); setReviewed(false); }} />)}
      <FormControlLabel control={<Checkbox checked={reviewed} disabled={busy} onChange={(e) => setReviewed(e.target.checked)} />} label={t("I reviewed the included data, identifiers and notes", "我已核对包含的数据、标识符及说明")} />
      <Button variant="contained" disabled={busy || !reviewed} onClick={() => void run(async () => {
        const saved = await api.freeze(projectId, outputCheck.token, notes); setOutputs((items) => [saved, ...items]); setOutputCheck(null); setNotes(null); setReviewed(false);
        setMessage(t("Output frozen locally. Later edits will not overwrite it.", "输出已冻结到本地，后续编辑不会覆盖此版本。"));
      })}>{t("Save frozen output", "保存冻结输出")}</Button>
    </Stack>}
    <Typography component="h3" variant="h6">{t("Saved output versions", "已保存输出版本")}</Typography>
    {outputs.map((o) => <Box key={o.id} sx={{ overflowWrap: "anywhere" }}><Button component="a" href={api.download(projectId, o.id)} download>{t("Download research ZIP", "下载研究 ZIP")} · {o.id.slice(0, 8)}</Button><Typography variant="caption" sx={{ display: "block" }}>{o.createdAt} · SHA-256: {o.sha256}</Typography></Box>)}
  </Stack></Paper>;
}
