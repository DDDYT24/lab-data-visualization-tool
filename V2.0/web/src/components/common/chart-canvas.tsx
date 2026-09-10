"use client";

import { Box, Skeleton } from "@mui/material";
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

export function ChartCanvas({
  analysis,
  spec,
  preview,
  excludedFindingIds = [],
  findings = [],
}: ChartCanvasProps) {
  const t = useTranslations("common");
  const [surfaceSupport, setSurfaceSupport] = useState<
    "idle" | "loading" | "ready" | "error"
  >("idle");
  const height = spec.panelCount > 2 ? 560 : 440;

  useEffect(() => {
    if (spec.type !== "surface3d" || surfaceSupport !== "idle") return;
    void import("echarts-gl").then(
      () => setSurfaceSupport("ready"),
      () => setSurfaceSupport("error"),
    );
  }, [spec.type, surfaceSupport]);

  const option = useMemo(
    () =>
      buildChartOption({
        analysis,
        excludedFindingIds,
        findings,
        preview,
        spec,
      }),
    [analysis, excludedFindingIds, findings, preview, spec],
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
  const isSurface = spec.type === "surface3d";

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
    <Box
      aria-label={t("chartPreview", { title: spec.title, type: spec.type })}
      data-camera-controls={isSurface ? Boolean(grid3D?.viewControl) : undefined}
      data-chart-components={isSurface ? surfaceComponents.join(",") : undefined}
      data-surface-point-count={isSurface ? surfaceSeries?.data?.length ?? 0 : undefined}
      data-surface-support={isSurface ? surfaceSupport : undefined}
      role="img"
      sx={{ bgcolor: "background.paper", minHeight: height, width: "100%" }}
    >
      <ReactECharts
        notMerge
        option={option}
        opts={{ renderer: spec.type === "surface3d" ? "canvas" : "svg" }}
        style={{ height, width: "100%" }}
      />
    </Box>
  );
}
