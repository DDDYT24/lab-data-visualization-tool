"use client";

import { Chip, Stack, Typography } from "@mui/material";
import { useLocale } from "next-intl";
import { figureGenerationLabels, figureLayoutLabels, type FigureWorkspaceState } from "./figure-workspace-state";

export function FigureStatus({ state, compact = false }: { state: FigureWorkspaceState; compact?: boolean }) {
  const language = useLocale().startsWith("zh") ? 1 : 0;
  return <Stack direction="row" useFlexGap spacing={1} role="status" aria-label={language ? "组合图保存与生成" : "Figure persistence and generation"}
    sx={{ flexWrap: "wrap", alignItems: "center", minWidth: 0 }} data-testid={compact ? "figure-status-compact" : "figure-status"}>
    {!compact && <Typography variant="caption">{language ? "布局保存" : "Layout persistence"}</Typography>}
    <Chip size="small" variant="outlined" color={state.layout === "saveFailed" ? "error" : state.layout === "draft" ? "warning" : "primary"} label={figureLayoutLabels[state.layout][language]} />
    {!compact && <Typography variant="caption">{language ? "完整生成" : "Complete generation"}</Typography>}
    <Chip size="small" variant="outlined" color={state.generation === "ready" ? "success" : state.generation === "failed" ? "error" : "default"} label={figureGenerationLabels[state.generation][language]} />
  </Stack>;
}
