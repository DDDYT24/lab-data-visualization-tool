"use client";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocale } from "next-intl";
import { Accordion, AccordionDetails, AccordionSummary, Alert, Autocomplete, Box, Button, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from "@mui/material";
import ExpandMoreRoundedIcon from "@mui/icons-material/ExpandMoreRounded";
import { useInterfaceMode } from "@/components/layout/interface-mode";
import { figureMarkAt } from "@/domain/figure-point-hit";
import type { FigureInteraction } from "@/domain/research-figure-board";
import { researchFigureApi } from "@/lib/api/research-figure-api";

const SourceRows = memo(function SourceRows({ rows, selected, choose }: {
  rows: FigureInteraction["rows"]; selected: boolean; choose: (next: { row?: string; mark?: string }) => void;
}) {
  return <TableBody>{rows.map((row) => <TableRow key={`${row.panel}:${row.id}`} selected={selected}>
    <TableCell>{row.panel}</TableCell><TableCell><Button onClick={() => choose({ row: row.id })}>{row.sampleLabel || row.id.slice(-12)}</Button></TableCell>
    <TableCell sx={{ minWidth: 220, overflowWrap: "anywhere" }}>{row.values ? JSON.stringify(row.values) : `${row.value} ${row.unit}`}</TableCell>
    <TableCell>{row.lineage.map((r) => `${r.sourceId.slice(0, 8)}:${r.row}:${r.column}`).join("; ")}</TableCell>
  </TableRow>)}</TableBody>;
});

export function ResearchFigureInspector({ projectId, boardId, active: visible = true, downloadsEnabled = true }: { projectId: string; boardId: string; active?: boolean; downloadsEnabled?: boolean }) {
  const overview = useInterfaceMode().mode === "overview";
  const [detailsOpen, setDetailsOpen] = useState(false);
  const zh = useLocale().startsWith("zh");
  const t = (en: string, cn: string) => zh ? cn : en;
  const [enlarged, setEnlarged] = useState(false);
  const [geometry, setGeometry] = useState<FigureInteraction["marks"]>([]);
  const geometryKey = useRef<string | null>(null);
  const [view, setView] = useState<FigureInteraction | null>(null);
  const [selection, setSelection] = useState<{ mark?: string; row?: string }>({});
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [descending, setDescending] = useState(false);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    if (!visible) return;
    let active = true;
    const params = new URLSearchParams({ offset: String(offset), query, descending: String(descending), ...selection });
    const key = projectId + ":" + boardId;
    const fullGeometry = geometryKey.current !== key;
    if (!fullGeometry) params.set("highlightsOnly", "true");
    const controller = new AbortController();
    researchFigureApi.interaction(projectId, boardId, params, controller.signal).then((result) => {
      if (active) {
        setView(result);
        if (fullGeometry) { setGeometry(result.marks); geometryKey.current = key; }
        setError(null);
      }
    }).catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Figure unavailable"); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; controller.abort(); };
  }, [projectId, boardId, selection, query, descending, offset, visible, retryKey]);
  useEffect(() => {
    const timer = setTimeout(() => { if (queryInput !== query) { setLoading(true); setQuery(queryInput); setOffset(0); } }, 300);
    return () => clearTimeout(timer);
  }, [queryInput, query]);
  const choose = useCallback((next: { mark?: string; row?: string }) => { if (next.mark || next.row) setDetailsOpen(true); setLoading(true); setSelection(next); setOffset(0); }, []);
  const compactPoints = geometry.length > 500;
  const staticMarks = useMemo(() => visible ? geometry.filter((mark) => !compactPoints || mark.kind !== "point").map((mark) => <circle key={mark.id} data-testid={`figure-mark-${mark.id}`} cx={mark.cx * 1000} cy={mark.cy * 1000}
    r={mark.kind === "point" ? 5 : 9} fill="transparent" stroke={mark.kind === "point" ? "transparent" : "#008577"} strokeWidth={1.5}
    role="button" tabIndex={-1} aria-label={`${mark.panel}: ${mark.kind} ${mark.label} (${mark.inputCount})`}
    onClick={(event) => { if (compactPoints) return; event.stopPropagation(); choose({ mark: mark.id }); }} style={{ cursor: "pointer" }} />) : [], [geometry, choose, visible, compactPoints]);
  if (!visible) return null;
  return <Stack spacing={overview ? 1 : 2} data-testid="figure-inspector">
    <Typography component="h3" variant="h6" sx={overview ? { position: "absolute", width: "1px", height: "1px", clipPath: "inset(50%)", overflow: "hidden" } : undefined}>{t("Saved figure and linked observations", "已保存组合图与观测联动")}</Typography>
    {error && <Alert severity="error" action={<Button onClick={() => { setLoading(true); setRetryKey((value) => value + 1); }}>{t("Retry source lookup", "重试来源查询")}</Button>}>{error}</Alert>}
    <Typography variant="caption" sx={overview ? { display: "none" } : undefined}>{t("Screen preview. Enlarge loads the original; downloads retain the complete 300 dpi figure.", "屏幕预览；放大后加载原图，下载保留完整 300 dpi 组合图。")}</Typography>
    <Button title={t("Screen preview; downloads retain the complete 300 dpi figure.", "屏幕预览；下载保留完整 300 dpi 组合图。")} sx={overview ? { alignSelf: "flex-start" } : undefined} onClick={() => setEnlarged(!enlarged)}>{enlarged ? t("Fit figure to screen", "组合图适应屏幕") : t("Enlarge figure for reading", "放大组合图阅读")}</Button>
    <Box sx={{ overflowX: "auto", maxWidth: "100%" }}>
    <Box sx={{ position: "relative", width: "100%", minWidth: enlarged ? 1000 : undefined, maxWidth: overview && !enlarged ? "calc(max(300px, 100dvh - 560px) * 1.272727)" : undefined, mx: "auto", bgcolor: "white" }}>
      <Box component="img" alt={t("Saved composed figure", "已保存组合图")} src={visible ? researchFigureApi.artifact(projectId, boardId, "png", false, !enlarged) : undefined} sx={{ width: "100%", display: "block" }} />
      <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label={t("Figure selection overlay", "组合图选择层")}
        onClick={(event) => {
          if (!compactPoints) return;
          const bounds = event.currentTarget.getBoundingClientRect();
          if (!bounds.width || !bounds.height) return;
          const mark = figureMarkAt(geometry, (event.clientX - bounds.left) * 1000 / bounds.width,
            (event.clientY - bounds.top) * 1000 / bounds.height);
          if (mark) choose({ mark });
        }} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
        {compactPoints && <rect width={1000} height={1000} fill="transparent" aria-hidden="true" style={{ cursor: "pointer" }} />}
        {staticMarks}
        {view?.marks.filter((mark) => mark.highlighted).map((mark) => <circle key={mark.id}
          cx={mark.cx * 1000} cy={mark.cy * 1000} r={mark.kind === "point" ? 5 : 9}
          fill="#ffb300" fillOpacity={0.55} stroke="#8d5000" strokeWidth={1.5}
          aria-hidden="true" style={{ pointerEvents: "none" }} />)}
      </svg>
    </Box>
    </Box>
    <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: "wrap" }}>
      {downloadsEnabled && ["svg", "png", "pdf"].map((format) => <Button key={format} component="a" href={researchFigureApi.artifact(projectId, boardId, format, true)}>{t("Download figure", "下载组合图")} {format.toUpperCase()}</Button>)}
      <Button onClick={() => choose({})}>{t("Clear highlight", "清除高亮")}</Button>
    </Stack>
    <Typography role="status">{loading ? t("Loading…", "加载中…") : `${t("Selected inputs", "所选输入")}: ${view?.selectedInputs ?? 0} · ${t("Matching panel rows", "匹配面板数据行")}: ${view?.totalRows ?? 0}`}</Typography>
    <Accordion expanded={!overview || detailsOpen} onChange={(_, expanded) => setDetailsOpen(expanded)} disableGutters
      sx={{ minWidth: 0, ...(!overview && { border: 0, "&::before": { display: "none" } }) }} data-testid="figure-source-dock">
      <AccordionSummary expandIcon={<ExpandMoreRoundedIcon />} sx={{ display: overview ? "flex" : "none" }} aria-controls="figure-source-details" id="figure-source-toggle">
        {t("Linked source rows", "关联来源行")} · {t("Selected inputs", "所选输入")}: {view?.selectedInputs ?? 0}
      </AccordionSummary>
      <AccordionDetails id="figure-source-details" sx={{ p: overview ? 2 : 0 }}><Stack spacing={2} sx={{ minWidth: 0 }}>
    <Alert severity="info">{t("Selection only highlights data. Search filters this table only. Exclude observations in Research preparation with a reason, then create new chart and figure revisions.", "点选仅高亮数据。搜索仅筛选此表。排除数据需回到研究数据准备中填写理由，再生成新的图表和布局版本。")}</Alert>
    <Autocomplete options={geometry}
      value={geometry.find((mark) => mark.id === selection.mark) ?? null}
      getOptionLabel={(mark) => `${mark.panel}: ${mark.kind} · ${mark.label} (${mark.inputCount})`}
      isOptionEqualToValue={(option, value) => option.id === value.id}
      filterOptions={(options, state) => {
        const query = state.inputValue.toLocaleLowerCase();
        return options.filter((mark) => `${mark.panel}: ${mark.kind} · ${mark.label} (${mark.inputCount})`.toLocaleLowerCase().includes(query)).slice(0, 100);
      }}
      onChange={(_, mark) => choose(mark ? { mark: mark.id } : {})}
      clearText={t("Clear mark", "清除标记")} openText={t("Open mark search", "打开标记搜索")}
      closeText={t("Close mark search", "关闭标记搜索")} noOptionsText={t("No matching mark", "没有匹配标记")}
      renderInput={(params) => <TextField {...params} label={t("Select a mark (keyboard)", "选择标记（键盘）")}
        helperText={t("Search any displayed mark; up to 100 suggestions are shown. The paginated source table contains all inputs.", "可搜索任何已显示标记，每次最多展示 100 个建议。分页来源表包含全部输入。")}/>} />

    {view?.selectedKind && view.selectedKind !== "point" && view.selectedKind !== "row" && <Alert severity="info">{t("This aggregate/density mark represents the contributing input subset, not an additional observation.", "此汇总／密度标记代表贡献数据集合，不是额外观测。")}</Alert>}
    {view?.sampled && <Alert severity="info">{t("Point hit targets or legacy previews are sampled. The paginated table retains all input rows; selection never changes n or saved exports.", "点选目标或旧图表预览经过抽样。分页表仍保留完整输入数据；点选不会改变 n 或已保存导出。")}</Alert>}
    <TextField label={t("Filter source table", "筛选来源表")} value={queryInput} onChange={(e) => setQueryInput(e.target.value)} />
    <Button onClick={() => { setLoading(true); setDescending(!descending); setOffset(0); }}>{t("Reverse source order", "反转来源排序")}</Button>
    <Typography variant="caption">{t("On narrow screens, scroll the source table horizontally to read all columns.", "窄屏下可横向滚动来源表，查看完整列。")}</Typography>
    <TableContainer sx={{ maxHeight: 420, overflowX: "auto" }}><Table size="small" sx={{ minWidth: 640 }} aria-label={t("Figure source observations", "组合图来源观测")}>
      <TableHead><TableRow>{[t("Panel", "面板"), t("Sample / row", "样本／数据行"), t("Value", "数值"), t("Source", "来源")].map((name) => <TableCell key={name}>{name}</TableCell>)}</TableRow></TableHead>
      {view && <SourceRows rows={view.rows} selected={view.selectedInputs > 0} choose={choose} />}
    </Table></TableContainer>
    <Stack direction="row" spacing={1}><Button disabled={loading || offset === 0} onClick={() => { setLoading(true); setOffset(Math.max(0, offset - 100)); }}>{t("Previous rows", "上一页")}</Button>
      <Button disabled={loading || offset + 100 >= (view?.totalRows ?? 0)} onClick={() => { setLoading(true); setOffset(offset + 100); }}>{t("Next rows", "下一页")}</Button></Stack>
      </Stack></AccordionDetails>
    </Accordion>
  </Stack>;
}
