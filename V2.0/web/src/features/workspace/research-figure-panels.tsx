"use client";

import { useEffect, useState } from "react";
import { Box, Skeleton } from "@mui/material";

import { RenderPlanChart } from "@/components/common/render-plan-chart";
import { densePointThreshold } from "@/domain/chart-render-contract";
import type { RenderPlan } from "@/domain/render-plan";
import type { FigureInteraction } from "@/domain/research-figure-board";
import { researchFigureApi } from "@/lib/api/research-figure-api";

/** Panels a browser can draw: they have a plan and hold no more points than a single chart. */
export function interactivePanels(panels: FigureInteraction["panels"]) {
  return (panels ?? []).filter((panel) => panel.planHash && (panel.observations ?? 0) <= densePointThreshold);
}

export function ResearchFigurePanels({ boardId, labels, onSelect, projectId, selected }: {
  boardId: string; labels: string[]; onSelect: (ids: string[]) => void; projectId: string; selected: string[];
}) {
  const key = `${boardId}:${labels.join("")}`;
  const [loaded, setLoaded] = useState<{ key: string; plans: (RenderPlan | null)[] } | null>(null);
  useEffect(() => {
    let mounted = true;
    void Promise.all(labels.map((label) => researchFigureApi.panelPlan(projectId, boardId, label).catch(() => null)))
      .then((plans) => { if (mounted) setLoaded({ key: `${boardId}:${labels.join("")}`, plans }); });
    return () => { mounted = false; };
  // The joined key stands for the labels array, which is rebuilt on every render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boardId, projectId, key]);
  if (loaded?.key !== key) return <Skeleton height={440} variant="rounded" />;
  return <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "minmax(0, 1fr)", lg: "repeat(2, minmax(0, 1fr))" } }}>
    {loaded.plans.map((plan) => plan && <RenderPlanChart key={plan.planHash} label={plan.panels[0]!.title}
      onSelect={onSelect} plan={plan} selected={selected} />)}
  </Box>;
}
