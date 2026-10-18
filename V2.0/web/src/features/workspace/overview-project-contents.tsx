"use client";

import { Button, Chip, Divider, Stack, Typography } from "@mui/material";
import { useLocale } from "next-intl";
import type { ResearchView } from "@/domain/research-preparation";
import { FigureStatus } from "./figure-status";
import type { FigureWorkspaceState } from "./figure-workspace-state";
import type { ResearchTask } from "./research-navigation";

export function OverviewProjectContents({ data, revisionId, preview, figureState, onSelectTask }: {
  data: ResearchView["data"]; revisionId: string | null; preview: boolean; figureState?: FigureWorkspaceState | null;
  onSelectTask: (task: ResearchTask) => void;
}) {
  const zh = useLocale().startsWith("zh");
  const text = (en: string, cn: string) => zh ? cn : en;
  const methods: Record<string, string> = {
    filter: text("Filter", "筛选"), convert_unit: text("Unit conversion", "单位换算"),
    wide_to_long: text("Wide to long", "宽表转长表"), long_to_wide: text("Long to wide", "长表转宽表"),
  };
  return <Stack spacing={1.5} data-testid="overview-project-contents" sx={{ minWidth: 0, "& .MuiButton-root": { justifyContent: "flex-start", textAlign: "left" } }}>
    <Typography component="h2" variant="subtitle1" sx={{ fontWeight: 700 }}>{text("Project contents", "项目内容")}</Typography>
    {preview && <Chip color="warning" variant="outlined" label={text("Unsaved preview", "未保存预览")} size="small" />}
    <Typography variant="subtitle2">{text("Samples and observations", "样本与观测")}</Typography>
    {data ? <>
      <Typography variant="caption">{text("Independent units", "独立实验单位")}: {data.summary.independentUnits ?? text("Not declared", "未确定")}</Typography>
      <Typography variant="caption">{text("Samples", "样本记录")}: {data.summary.samples} · {text("Observations", "观测")}: {data.summary.observations}</Typography>
      {data.samples.slice(0, 6).map((sample) => <Typography key={sample.id} variant="caption" sx={{ pl: 1, borderLeft: "2px solid", borderColor: "secondary.main", overflowWrap: "anywhere" }}>{sample.label}{sample.group ? ` · ${sample.group}` : ""}</Typography>)}
      {data.summary.samples > 6 && <Typography variant="caption" color="text.secondary">{text("First six loaded sample records", "仅列出已加载的前六个样本")}</Typography>}
    </> : <Typography variant="caption" color="text.secondary">{text("Prepare data to see declared samples.", "准备数据后显示已声明的样本。")}</Typography>}
    <Button onClick={() => onSelectTask("data")}>{text("Open sample data", "查看样本数据")}</Button>
    <Divider />
    <Typography variant="subtitle2">{text("Saved data revision", "已保存数据修订")}</Typography>
    <Typography variant="caption" title={revisionId ?? undefined} sx={{ overflowWrap: "anywhere" }}>{revisionId?.slice(0, 12) ?? text("No saved research revision", "尚无已保存研究修订")}</Typography>
    <Typography variant="subtitle2">{text("Source files", "来源文件")}</Typography>
    {data?.sources.map((source) => <Typography key={source.id} variant="caption" sx={{ overflowWrap: "anywhere" }}>{source.name}</Typography>)}
    <Divider />
    <Typography variant="subtitle2">{text("Processing steps", "处理步骤")}</Typography>
    {data?.history.length ? data.history.map(({ step }, i) => <Typography key={i} variant="caption">{i + 1}. {methods[String(step.kind)] ?? String(step.kind)}</Typography>)
      : <Typography variant="caption" color="text.secondary">{text("No transformation steps", "未添加转换步骤")}</Typography>}
    <Button onClick={() => onSelectTask("quality")}>{text("Quality and exclusions", "质量与排除记录")}</Button>
    <Divider />
    <Button onClick={() => onSelectTask("analysis")}>{text("Scientific charts", "科学图表")}</Button>
    {figureState && <>
      <Typography variant="subtitle2">{text("Current figure revision", "当前组合图修订")}</Typography>
      <Typography variant="caption" sx={{ overflowWrap: "anywhere" }}>{figureState.savedId?.slice(0, 12) ?? text("No saved figure revision", "尚无已保存组合图修订")}</Typography>
      <FigureStatus state={figureState} compact />
      {figureState.previewId && figureState.previewId !== figureState.savedId && <Typography variant="caption">{text("Previous preview", "上一预览")}: {figureState.previewId.slice(0, 12)}</Typography>}
    </>}
    <Button onClick={() => onSelectTask("figures")}>{text("Saved figure layouts", "已保存组合图")}</Button>
    <Button onClick={() => onSelectTask("outputs")}>{text("Templates and outputs", "模板与输出")}</Button>
  </Stack>;
}
