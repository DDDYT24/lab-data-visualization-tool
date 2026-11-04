"use client";

import { Box, Button, Skeleton, Stack, Typography } from "@mui/material";
import type { EChartsType } from "echarts";
import dynamic from "next/dynamic";
import { useLocale, useTranslations } from "next-intl";
import { useEffect, useMemo, useRef, useState } from "react";

import type { RenderPlan } from "@/domain/render-plan";

import { buildPlanOption, PLAN_ZOOM_IDS, planSelection } from "./render-plan-option";

const HEIGHT = 440;
const ZOOM_STEP = 25;
const RENDERER = { renderer: "canvas" } as const;
const FULL_VIEW = { start: 0, end: 100 };
const ReactECharts = dynamic(() => import("echarts-for-react"), {
  ssr: false,
  loading: () => <Skeleton height={HEIGHT} variant="rounded" />,
});

type ZoomWindow = { start: number; end: number };

export function zoomedWindow(view: ZoomWindow, step: number): ZoomWindow {
  const half = Math.min(50, Math.max(1, (view.end - view.start) / 2 + step / 2));
  const middle = Math.min(100 - half, Math.max(half, (view.start + view.end) / 2));
  return { start: middle - half, end: middle + half };
}

export function RenderPlanChart({ label, onSelect, plan, selected }: {
  label: string; onSelect: (observationIds: string[]) => void; plan: RenderPlan; selected: readonly string[];
}) {
  const t = useTranslations("common");
  const zh = useLocale().startsWith("zh");
  const instance = useRef<EChartsType | null>(null);
  const select = useRef(onSelect);
  const [view, setView] = useState<ZoomWindow>(FULL_VIEW);
  useEffect(() => { select.current = onSelect; }, [onSelect]);
  const zoomed = view.start > 0 || view.end < 100;
  const option = useMemo(() => buildPlanOption({
    plan, selected, zoomed,
    text: zh ? { mean: "均值", median: "中位数" } : { mean: "Mean", median: "Median" },
  }), [plan, selected, zh, zoomed]);
  // echarts-for-react rebuilds the chart when these handlers change identity.
  const events = useMemo(() => ({
    click: (params: { seriesIndex: number; dataIndex: number }) => {
      const ids = planSelection(plan, params.seriesIndex, params.dataIndex);
      if (ids.length) select.current(ids);
    },
    datazoom: () => {
      const zooms = (instance.current?.getOption().dataZoom ?? []) as Partial<ZoomWindow>[];
      const zoom = zooms.find((item) => (item.start ?? 0) > 0 || (item.end ?? 100) < 100);
      setView({ start: zoom?.start ?? 0, end: zoom?.end ?? 100 });
    },
  }), [plan]);
  const apply = (next: ZoomWindow) => instance.current?.dispatchAction({
    type: "dataZoom", batch: PLAN_ZOOM_IDS.map((dataZoomId) => ({ dataZoomId, ...next })),
  });
  const onChartReady = (chart: EChartsType) => { instance.current = chart; };
  return <Stack spacing={1}>
    <Box
      aria-label={label}
      data-plan-hash={plan.planHash}
      data-testid="render-plan-chart"
      role="img"
      sx={{ bgcolor: "white", minHeight: HEIGHT, touchAction: "pan-y", width: "100%" }}
    >
      <ReactECharts
        notMerge={false}
        onChartReady={onChartReady}
        onEvents={events}
        option={option}
        opts={RENDERER}
        style={{ height: HEIGHT, width: "100%" }}
      />
    </Box>
    <Stack direction="row" sx={{ flexWrap: "wrap", gap: 1 }}>
      <Button onClick={() => apply(zoomedWindow(view, -ZOOM_STEP))} size="small">{t("zoomIn")}</Button>
      <Button disabled={!zoomed} onClick={() => apply(zoomedWindow(view, ZOOM_STEP))} size="small">{t("zoomOut")}</Button>
      <Button disabled={!zoomed} onClick={() => apply(FULL_VIEW)} size="small">{t("resetView")}</Button>
    </Stack>
    {plan.disclosures.map((text) => <Typography key={text} variant="caption">{text}</Typography>)}
  </Stack>;
}
