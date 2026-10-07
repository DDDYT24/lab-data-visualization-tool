"use client";

import { Box, Button, Skeleton, Stack, Typography, useTheme } from "@mui/material";
import dynamic from "next/dynamic";
import { useTranslations } from "next-intl";
import { useEffect, useMemo, useState } from "react";

import type {
  ChartAnalysis,
  DataPreview,
  QualityFinding,
} from "@/domain/api-contract";
import type { ChartSpec } from "@/domain/chart-spec";

import { ApiStatePanel } from "./api-state-panel";
import { buildChartOption } from "./chart-option";

const ReactECharts = dynamic(() => import("echarts-for-react"), {
  ssr: false,
  loading: () => <Skeleton height={440} variant="rounded" />,
});

type ChartCanvasProps = {
  spec: ChartSpec;
  preview: DataPreview;
  analysis?: ChartAnalysis;
  excludedFindingIds?: string[];
  findings?: QualityFinding[];
};

const DEFAULT_SURFACE_VIEW = { alpha: 25, beta: 40, distance: 200 } as const;
type SurfaceView = { alpha: number; beta: number; distance: number };

export function ChartCanvas({
  analysis,
  spec,
  preview,
  excludedFindingIds = [],
  findings = [],
}: ChartCanvasProps) {
  const t = useTranslations("common");
  const theme = useTheme();
  const [surfaceSupport, setSurfaceSupport] = useState<
    "idle" | "loading" | "ready" | "error"
  >("idle");
  const [surfaceView, setSurfaceView] = useState<SurfaceView>({ ...DEFAULT_SURFACE_VIEW });
  const [showAxes, setShowAxes] = useState(true);
  const [showLegend, setShowLegend] = useState(true);
  const [lowCostFallback, setLowCostFallback] = useState(false);
  const isSurface = spec.type === "surface3d";
  const height = spec.panelCount > 2 ? 560 : 440;

  useEffect(() => {
    if (spec.type !== "surface3d" || surfaceSupport !== "idle") return;
    void import("echarts-gl").then(
      () => setSurfaceSupport("ready"),
      () => setSurfaceSupport("error"),
    );
  }, [spec.type, surfaceSupport]);

  useEffect(() => {
    if (spec.type !== "surface3d") return;
    const timer = window.setTimeout(() => {
      const device = navigator as Navigator & { deviceMemory?: number };
      setLowCostFallback(
        navigator.hardwareConcurrency <= 2 || (device.deviceMemory ?? 8) <= 2,
      );
    }, 0);
    return () => window.clearTimeout(timer);
  }, [spec.type]);

  const option = useMemo(
    () =>
      buildChartOption({
        analysis,
        colorMode: theme.palette.mode,
        excludedFindingIds,
        findings,
        preview,
        spec,
        showAxes,
        showLegend,
        surfacePointLimit: lowCostFallback ? 800 : undefined,
        surfaceView: isSurface ? surfaceView : undefined,
      }),
    [
      analysis,
      excludedFindingIds,
      findings,
      isSurface,
      lowCostFallback,
      preview,
      showAxes,
      showLegend,
      spec,
      surfaceView,
      theme.palette.mode,
    ],
  );
  const optionRecord = option as Record<string, unknown>;
  const surfaceComponents = ["grid3D", "xAxis3D", "yAxis3D", "zAxis3D"].filter(
    (component) => component in optionRecord,
  );
  const grid3D = optionRecord.grid3D as
    | { viewControl?: Record<string, unknown> }
    | undefined;
  const optionSeries = Array.isArray(optionRecord.series) ? optionRecord.series : [];
  const surfaceSeries = optionSeries[0] as { data?: unknown[] } | undefined;

  if (preview.rows.length === 0) {
    return (
      <ApiStatePanel
        compact
        description={t("noPreviewDescription")}
        kind="empty"
        title={t("noPreviewTitle")}
      />
    );
  }

  if (spec.type === "surface3d" && surfaceSupport === "error") {
    return (
      <ApiStatePanel
        compact
        description={t("surfaceUnavailableDescription")}
        kind="error"
        title={t("surfaceUnavailableTitle")}
      />
    );
  }

  if (
    spec.type === "surface3d" &&
    (surfaceSupport === "idle" || surfaceSupport === "loading")
  ) {
    return <Skeleton height={height} variant="rounded" />;
  }

  return (
    <Stack sx={{ width: "100%" }}>
      <Box
        aria-label={t("chartPreview", { title: spec.title, type: spec.type })}
        data-camera-controls={isSurface ? Boolean(grid3D?.viewControl) : undefined}
        data-chart-components={isSurface ? surfaceComponents.join(",") : undefined}
        data-surface-fallback={isSurface && lowCostFallback ? "low-cost" : undefined}
        data-surface-point-count={isSurface ? surfaceSeries?.data?.length ?? 0 : undefined}
        data-surface-support={isSurface ? surfaceSupport : undefined}
        data-touch-action={isSurface ? "pan-y" : undefined}
        data-surface-view={isSurface ? `${surfaceView.alpha}:${surfaceView.beta}:${surfaceView.distance}` : undefined}
        role="img"
        sx={{
          bgcolor: "background.paper",
          minHeight: height,
          touchAction: isSurface ? "pan-y" : undefined,
          width: "100%",
        }}
      >
        <ReactECharts
          notMerge
          option={option}
          opts={{ renderer: spec.type === "surface3d" ? "canvas" : "svg" }}
          style={{ height, width: "100%" }}
        />
      </Box>
      {isSurface ? (
        <Stack
          aria-label={t("surfaceControls")}
          direction={{ xs: "column", sm: "row" }}
          spacing={1}
          sx={{
            borderTop: 1,
            borderColor: "divider",
            p: 1.5,
            "& .MuiButton-root": { minHeight: 44 },
          }}
        >
          <Button
            onClick={() => setSurfaceView((view) => ({ ...view, beta: view.beta - 20 }))}
            size="small"
            variant="outlined"
          >
            {t("rotateLeft")}
          </Button>
          <Button
            onClick={() => setSurfaceView((view) => ({ ...view, beta: view.beta + 20 }))}
            size="small"
            variant="outlined"
          >
            {t("rotateRight")}
          </Button>
          <Button
            onClick={() => setSurfaceView((view) => ({ ...view, distance: Math.max(30, view.distance - 15) }))}
            size="small"
            variant="outlined"
          >
            {t("zoomIn")}
          </Button>
          <Button
            onClick={() => setSurfaceView((view) => ({ ...view, distance: Math.min(220, view.distance + 15) }))}
            size="small"
            variant="outlined"
          >
            {t("zoomOut")}
          </Button>
          <Button
            onClick={() => setSurfaceView({ ...DEFAULT_SURFACE_VIEW })}
            size="small"
            variant="outlined"
          >
            {t("resetView")}
          </Button>
          <Button onClick={() => setShowAxes((visible) => !visible)} size="small" variant="outlined">
            {t("toggleAxes")}
          </Button>
          <Button onClick={() => setShowLegend((visible) => !visible)} size="small" variant="outlined">
            {t("toggleLegend")}
          </Button>
        </Stack>
      ) : null}
      {isSurface ? (
        <Box sx={{ px: 1.5, pt: 1 }}>
          <Typography color="text.secondary" data-surface-gesture-hint variant="caption">
            {t("surfaceGestureHint")}
          </Typography>
        </Box>
      ) : null}
      {isSurface && lowCostFallback ? (
        <Box sx={{ px: 1.5, pb: 1.5 }}>
          <Typography color="text.secondary" variant="caption">
            {t("lowCostFallback")}
          </Typography>
        </Box>
      ) : null}
    </Stack>
  );
}
