"use client";

import { useEffect, useState } from "react";
import { useLocale } from "next-intl";
import {
  Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Checkbox, FormControlLabel,
  MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  TextField, Typography,
} from "@mui/material";
import type { ChartHistory, ScientificChart, ScientificChartSpec } from "@/domain/research-charts";
import { researchChartApi } from "@/lib/api/research-chart-api";

export function ResearchChartWorkspace({ projectId, revisionId, active = true, pending, onSaved, refreshKey = 0, mapping }: {
  mapping?: Record<string, string | null>;
  active?: boolean; projectId: string; revisionId: string | null; pending: boolean; onSaved?: () => void; refreshKey?: number;
}) {
  const zh = useLocale().startsWith("zh");
  const t = (en: string, cn: string) => zh ? cn : en;
  const [pointRendering, setPointRendering] = useState<"auto" | "vector" | "raster">("auto");
  const [kind, setKind] = useState<ScientificChartSpec["kind"]>("grouped_strip");
  const [title, setTitle] = useState("");
  const [seed, setSeed] = useState("labviz-strip-v1");
  const [summary, setSummary] = useState<ScientificChartSpec["summary"]>("none");
  const [summaryUnit, setSummaryUnit] = useState<ScientificChartSpec["summaryUnit"]>("experimental_unit");
  const [first, setFirst] = useState("before");
  const [second, setSecond] = useState("after");
  const [bandwidth, setBandwidth] = useState("");
  const [scale, setScale] = useState<ScientificChartSpec["densityScale"]>("equal_width");
  const [points, setPoints] = useState(true);
  const [history, setHistory] = useState<ChartHistory>({ charts: [] });
  const [chart, setChart] = useState<ScientificChart | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!active) return;
    let mounted = true;
    researchChartApi.list(projectId).then(async (list) => {
      if (!mounted) return;
      const saved = list.charts[0] ? await researchChartApi.get(projectId, list.charts[0].id) : null;
      if (mounted) { setHistory(list); setChart(saved); }
    }).catch((cause: unknown) => {
      if (mounted) setError(cause instanceof Error ? cause.message : "Chart load failed");
    }).finally(() => { if (mounted) setBusy(false); });
    return () => { mounted = false; };
  }, [projectId, refreshKey, active]);
  async function run(action: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await action(); } catch (cause) { setError(cause instanceof Error ? cause.message : "Chart failed"); }
    finally { setBusy(false); }
  }
  const names = {
    grouped_strip: t("Grouped raw points", "分组原始点图"),
    paired: t("Paired change", "配对变化图"),
    violin: t("Violin distribution", "小提琴分布图"),
  };
  const savedSpec = chart?.spec as ScientificChartSpec | undefined;
  const missingPairRoles = kind === "paired" && ["sample", "pair", "time"].some((key) => !mapping?.[key]);
  function downloadExample(example: ScientificChartSpec["kind"]) {
    const rows = ["sample,group,repeat,value,unit,time,pair"];
    for (let s = 1; s <= 6; s++) {
      if (example === "paired") {
        rows.push(`s${s},A,1,${s},mV,before,p${s}`, `s${s},A,1,${s + 2},mV,after,p${s}`);
      } else {
        for (let r = 1; r <= 3; r++) rows.push(`s${s},A,${r},${s + r - 2},mV,baseline,`);
      }
    }
    const url = URL.createObjectURL(new Blob([rows.join("\n") + "\n"], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `labviz-${example}-example.csv`;
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <Paper variant="outlined" sx={{ p: 2 }} data-testid="research-charts">
    <Stack spacing={2}>
      <Typography variant="h5" component="h2">{t("Scientific charts", "科学图表")}</Typography>
      <Typography variant="body2">{t(
        "Charts use an applied research revision. Each saved chart freezes its data, method and PNG/SVG/PDF exports.",
        "图表使用已应用的研究版本。每次保存都会固定数据、方法和 PNG／SVG／PDF 导出。")}</Typography>
      {(pending || !revisionId) && <Alert severity="info">{t(
        "Apply your research preparation before creating a chart. Existing charts keep their original revision.",
        "请先应用研究数据准备，再创建图表。历史图表仍绑定原版本。")}</Alert>}
      {error && <Alert severity="error">{t("Chart unavailable: ", "图表暂不可用：")}{error}</Alert>}
      <Box component="fieldset" disabled={busy} sx={{ border: 0, p: 0, m: 0, minWidth: 0 }}>
        <Stack spacing={2}>
          <TextField select label={t("Chart family", "图表类型")} value={kind}
            onChange={(event) => setKind(event.target.value as ScientificChartSpec["kind"])}>
            {Object.entries(names).map(([key, label]) => <MenuItem key={key} value={key}>{label}</MenuItem>)}
          </TextField>
          {kind === "paired" && <Alert severity={missingPairRoles ? "warning" : "info"}>
            {t("Paired charts require mapped sample, pair and time fields. Review Data before generating. Summary and density settings are retained for other chart families; they do not apply to paired charts.", "配对图需要映射样本、配对键和时间字段，请先在数据任务中检查。汇总与密度设置会保留供其他图表类型使用，不应用于配对图。")}
            <br />{["sample", "pair", "time"].map((key) => `${key}: ${mapping?.[key] || t("unmapped", "未映射")}`).join(" · ")}
          </Alert>}
          <TextField label={t("Chart title", "图表标题")} value={title} slotProps={{ htmlInput: { maxLength: 160 } }}
            onChange={(event) => setTitle(event.target.value)} />
          {kind === "paired" ? <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
            <TextField label={t("First time label", "第一个时间点")} value={first} onChange={(e) => setFirst(e.target.value)} />
            <TextField label={t("Second time label", "第二个时间点")} value={second} onChange={(e) => setSecond(e.target.value)} />
          </Stack> : <>
            <TextField label={t("Jitter seed", "抖动种子")} value={seed} onChange={(e) => setSeed(e.target.value)} />
            <TextField select label={t("Summary mark", "汇总标记")} value={summary}
              onChange={(e) => setSummary(e.target.value as ScientificChartSpec["summary"])}>
              <MenuItem value="none">{t("None", "无")}</MenuItem><MenuItem value="mean">{t("Mean", "均值")}</MenuItem>
              <MenuItem value="median">{t("Median", "中位数")}</MenuItem>
            </TextField>
            {summary !== "none" && <TextField select label={t("Summary unit", "汇总单位")} value={summaryUnit}
              onChange={(e) => setSummaryUnit(e.target.value as ScientificChartSpec["summaryUnit"])}>
              <MenuItem value="experimental_unit">{t("Independent experimental unit", "独立实验单位")}</MenuItem>
              <MenuItem value="sample">{t("Sample", "样本")}</MenuItem><MenuItem value="observation">{t("Observation", "观测")}</MenuItem>
            </TextField>}
          </>}
          {kind === "violin" && <>
            <TextField label={t("Bandwidth (blank = Scott)", "带宽（留空使用 Scott）")} type="number" value={bandwidth}
              onChange={(e) => setBandwidth(e.target.value)} slotProps={{ htmlInput: { min: 0, step: "any" } }} />
            <TextField select label={t("Density scale", "密度缩放")} value={scale}
              onChange={(e) => setScale(e.target.value as ScientificChartSpec["densityScale"])}>
              <MenuItem value="equal_width">{t("Equal maximum width", "等最大宽度")}</MenuItem>
              <MenuItem value="common_density">{t("Common density scale", "统一密度比例")}</MenuItem>
            </TextField>
            <FormControlLabel control={<Checkbox checked={points} onChange={(e) => setPoints(e.target.checked)} />}
              label={t("Show observed points", "显示原始观测点")} />
            <Alert severity="info">{t(
              "Density describes observations, including technical repeats. It does not increase independent n. At least 3 non-constant readings per group are required.",
              "密度描述观测值（包括技术重复），不增加独立样本数。每组至少需要 3 个非恒定读数。")}</Alert>
          </>}
          <TextField select label={t("Point layer export", "点层导出")} value={pointRendering}
            onChange={(e) => setPointRendering(e.target.value as typeof pointRendering)}
            helperText={t("Auto embeds all points at 300 dpi above 20,000 observations per panel. Text, axes and density stay vector; embedded points are not individually editable. Source data is complete.", "自动模式在每面板超过 20,000 个观测时以 300 dpi 嵌入全部点。文字、坐标轴、密度保留矢量；嵌入点无法逐个编辑，源数据完整保留。")}>
            <MenuItem value="auto">{t("Auto (large data hybrid)", "自动（大数据混合导出）")}</MenuItem>
            <MenuItem value="vector">{t("Force vector (slower for large data)", "强制全矢量（大数据较慢）")}</MenuItem>
            <MenuItem value="raster">{t("300 dpi point layer", "300 dpi 点层")}</MenuItem>
          </TextField>
          <Button variant="contained" disabled={busy || pending || !revisionId || missingPairRoles}
            onClick={() => void run(async () => {
              const created = await researchChartApi.create(projectId, revisionId!, {
                schemaVersion: 1, kind, title, locale: zh ? "zh" : "en", jitterSeed: seed, jitterWidth: 0.3,
                showPoints: kind === "violin" ? points : true,
                summary: kind === "paired" ? "none" : summary, summaryUnit,
                pairTimes: kind === "paired" ? [first, second] : [],
                bandwidth: kind === "violin" && bandwidth !== "" ? Number(bandwidth) : null,
                densityScale: scale, pointRendering,
              });
              setChart(created); onSaved?.(); setHistory(await researchChartApi.list(projectId));
            })}>{busy ? t("Working…", "处理中…") : t("Generate and save chart", "生成并保存图表")}</Button>
        </Stack>
      </Box>
      <Accordion>
        <AccordionSummary>{t("Examples and mapping guide", "示例与字段映射说明")}</AccordionSummary>
        <AccordionDetails>
          <Typography variant="body2">{t(
            "Download a synthetic example, choose it in Sources, map value → value, sample and independent unit → sample, group → group, repeat → repeat, unit → unit, time → time, pair → pair, then preview and apply. Raw/violin: 6 units, 18 readings, unit mean 3.5. Paired: 6 pairs, every after − before = 2.",
            "下载合成示例，在来源中选择文件；测量值映射 value，样本和独立单位均映射 sample，分组 group，技术重复 repeat，单位 unit，时间 time，配对键 pair，随后预览并应用。原始点／小提琴：6 个独立单位、18 次读数，单位均值 3.5。配对：6 对，after − before 均为 2。")}</Typography>
          {Object.entries(names).map(([key, label]) => <Button key={key}
            onClick={() => downloadExample(key as ScientificChartSpec["kind"])}>{label} CSV</Button>)}
        </AccordionDetails>
      </Accordion>
      <TextField select label={t("Saved charts", "已保存图表")} value={chart?.id ?? ""}
        disabled={busy || !history.charts.length}
        onChange={(e) => void run(async () => setChart(await researchChartApi.get(projectId, e.target.value)))}>
        {!history.charts.length && <MenuItem value="">{t("No saved charts", "暂无图表")}</MenuItem>}
        {history.charts.map((item) => {
          const spec = item.spec as ScientificChartSpec;
          return <MenuItem key={item.id} value={item.id}>{spec.title || names[spec.kind]} · {item.id.slice(0, 8)}</MenuItem>;
        })}
      </TextField>
      {chart && savedSpec && <>
        <Typography role="status">{t("Frozen chart: preview and downloads use the saved method and revision. Controls above configure a new chart.", "冻结图表：预览与下载使用已保存的方法和版本，上方控件用于配置新图表。")}</Typography>
        {chart.revisionId !== revisionId && <Alert severity="info">{t(
          "Historical chart: its data differ from the current research revision.",
          "历史图表：其数据版本与当前研究版本不同。")}</Alert>}
        <Typography variant="body2" data-testid="chart-counts">
          {t("Observations", "观测")}: {chart.observations} · {t("Excluded", "已排除")}: {chart.excluded} ·
          {t("Research revision", "研究版本")}: {chart.revisionId.slice(0, 8)}
        </Typography>
        {active && <Box component="img" src={researchChartApi.artifact(projectId, chart.id, "png", false, true)}
          alt={savedSpec.title || names[savedSpec.kind]} sx={{ width: "100%", height: "auto", bgcolor: "white" }} />}
        <Stack direction="row" sx={{ flexWrap: "wrap", gap: 1 }}>
          {(["png", "svg", "pdf", "json"] as const).map((format) => <Button component="a" key={format}
            href={researchChartApi.artifact(projectId, chart.id, format, true)} download>
            {t("Download", "下载")} {format.toUpperCase()}</Button>)}
        </Stack>
        <Typography variant="caption">{t(
          "JSON includes source identities, raw reported values, exclusions, all coordinates and density inputs/results. Image exports contain chart labels and counts.",
          "JSON 包含来源身份、原始报告值、排除记录、全部坐标及密度输入／结果。图片导出包含图中标签和数量。")}</Typography>
        <Typography variant="body2">{t("Frozen method", "已保存方法")}: {savedSpec.summary} / {savedSpec.summaryUnit} ·
          {t("Seed", "种子")}: {savedSpec.jitterSeed} · SHA256 {chart.resultSha256.slice(0, 12)}</Typography>
        {chart.groups.map((g, i) => <Typography variant="body2" key={i}>
          {g.label ?? t("Undeclared group", "未声明分组")} · {t("Independent units", "独立单位")}: {g.independentUnits ?? t("unknown", "未知")} ·
          {t("Known samples", "已知样本")}: {g.knownSamples} · {t("Observations", "观测")}: {g.observations}
          {g.summary !== null && ` · ${savedSpec.summary}=${Number(g.summary.toPrecision(8))}, n=${g.summaryN}`}
        </Typography>)}
        {chart.densityMethods.map((d, i) => <Typography variant="body2" key={i}>
          {d.group ?? t("Undeclared", "未声明")} · Gaussian KDE · h={Number(d.bandwidth.toPrecision(8))} ({d.bandwidthRule}) ·
          {d.scale} · {t("Support", "范围")}=[{d.support.map((v) => Number(v.toPrecision(6))).join(", ")}] ·
          {d.gridSize} {t("grid points; no boundary correction", "网格点；无边界校正")}
        </Typography>)}
        {!!chart.pairCount && <TableContainer><Table size="small" aria-label={t("Paired changes", "配对变化表")}>
          <TableHead><TableRow>{[t("Sample", "样本"), savedSpec.pairTimes[0], savedSpec.pairTimes[1],
            t("Second minus first", "第二次减第一次")].map((label, i) => <TableCell key={i}>{label}</TableCell>)}</TableRow></TableHead>
          <TableBody>{active && chart.pairs.map((pair) => <TableRow key={pair.sampleId}>
            <TableCell>{pair.sampleLabel}</TableCell><TableCell>{pair.first}</TableCell>
            <TableCell>{pair.second}</TableCell><TableCell>{pair.delta}</TableCell>
          </TableRow>)}</TableBody>
        </Table></TableContainer>}
        <Typography component="h3" variant="h6">{t("Chart source data", "图表来源数据")}</Typography>
        <TableContainer><Table size="small" sx={{ minWidth: 760 }} aria-label={t("Chart source data", "图表来源数据")}>
          <TableHead><TableRow>{[t("Sample / unit", "样本／独立单位"), t("Group / time", "分组／时间"),
            t("Value / reported", "值／原始报告"), t("State / exclusion", "状态／排除"), t("Source cells", "来源单元格")]
            .map((label) => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead>
          <TableBody>{active && chart.rows.map((row) => <TableRow key={row.id}>
            <TableCell>{row.sampleLabel ?? "?"} / {row.experimentalUnit ?? "?"}</TableCell>
            <TableCell>{row.group ?? "?"} / {row.time ?? "—"}</TableCell>
            <TableCell>{row.value ?? "—"} {row.unit}<br />{row.reportedText} {row.reportedUnit}</TableCell>
            <TableCell>{row.state} / {row.exclusionReason ?? "—"}</TableCell>
            <TableCell>{row.lineage.map((cell, i) => <Box key={i}>
              {chart.sources.find((source) => source.id === cell.sourceId)?.name}: {cell.row}, {cell.column}
            </Box>)}</TableCell>
          </TableRow>)}</TableBody>
        </Table></TableContainer>
        <Stack direction="row" spacing={1}>
          <Button disabled={busy || chart.offset === 0} onClick={() => void run(async () =>
            setChart(await researchChartApi.get(projectId, chart.id, Math.max(0, chart.offset - 100))))}>{t("Previous chart rows", "上一页图表数据")}</Button>
          <Typography>{chart.offset + 1}–{Math.min(chart.offset + 100, chart.totalRows)} / {chart.totalRows}</Typography>
          <Button disabled={busy || chart.offset + 100 >= chart.totalRows} onClick={() => void run(async () =>
            setChart(await researchChartApi.get(projectId, chart.id, chart.offset + 100)))}>{t("Next chart rows", "下一页图表数据")}</Button>
        </Stack>
      </>}
    </Stack>
  </Paper>;
}
