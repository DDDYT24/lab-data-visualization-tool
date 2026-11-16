import type { EChartsOption } from "echarts";

import type { ChartAnalysis } from "@/domain/api-contract";
import { gridColor } from "@/domain/chart-render-contract";
import type { ChartSpec } from "@/domain/chart-spec";

import { buildPlanOption, buildSurfacePlanOption, planAxisTooltip, planText, type SurfaceView } from "./render-plan-option";

// The preview draws the API's RenderPlan and nothing else: no statistic, bin or axis is computed
// here (see no-statistics.test.ts). The page adds only its own title, legend place and theme.

function legendOption(position: ChartSpec["export"]["legendPosition"]) {
  if (position === "none") return { show: false };
  if (position === "left" || position === "right") {
    return { orient: "vertical" as const, [position]: 8, top: "middle" };
  }
  return {
    left: "center",
    top: position === "bottom" ? undefined : 42,
    bottom: position === "bottom" ? 8 : undefined,
  };
}

function panelGrids(panelCount: number) {
  if (panelCount === 1) {
    return [{ left: 72, right: 38, top: 86, bottom: 62, containLabel: true }];
  }
  const columns = 2;
  const rows = Math.ceil(panelCount / columns);
  return Array.from({ length: panelCount }, (_, index) => ({
    left: `${5 + (index % columns) * 49}%`,
    top: `${16 + Math.floor(index / columns) * (76 / rows)}%`,
    width: "42%",
    height: `${62 / rows}%`,
    containLabel: true,
  }));
}

type ChartColorMode = "light" | "dark";
const DEFAULT_SURFACE_VIEW: SurfaceView = { alpha: 25, beta: 40, distance: 200 };

/** The option for a classic chart's plan, or null while there is no plan to draw. */
export function buildChartOption({
  analysis,
  colorMode = "light",
  spec,
  showAxes = true,
  showLegend = true,
  surfacePointLimit,
  surfaceView,
  zh = false,
}: {
  analysis?: ChartAnalysis;
  colorMode?: ChartColorMode;
  spec: ChartSpec;
  showAxes?: boolean;
  showLegend?: boolean;
  surfacePointLimit?: number;
  surfaceView?: SurfaceView;
  zh?: boolean;
}): EChartsOption | null {
  const plan = analysis?.plan;
  if (!plan) return null;
  const text = colorMode === "dark" ? "#ECECEC" : "#172033";
  const colors = { grid: colorMode === "dark" ? "#424242" : gridColor, text };
  const common = {
    animation: false,
    backgroundColor: spec.export.transparentBackground
      ? "transparent"
      : colorMode === "dark"
        ? "#2F2F2F"
        : spec.export.backgroundColor,
    legend: showLegend ? legendOption(spec.export.legendPosition) : { show: false },
    textStyle: { color: text, fontFamily: spec.export.fontFamily, fontSize: spec.export.fontSize },
    title: {
      left: "center",
      subtext: spec.subtitle || undefined,
      text: spec.title,
      textStyle: { fontSize: Math.max(16, spec.export.fontSize + 6), fontWeight: 600 },
    },
  };

  if (spec.type === "surface3d") {
    return {
      ...common,
      tooltip: {},
      ...buildSurfacePlanOption({
        colors: { ...colors, axis: colorMode === "dark" ? "#B4B4B4" : "#475467" },
        plan,
        pointLimit: surfacePointLimit,
        showAxes,
        showLegend,
        view: surfaceView ?? DEFAULT_SURFACE_VIEW,
      }),
    } as EChartsOption;
  }
  const planned = buildPlanOption({ colors, plan, text: planText(zh) });
  const shared = spec.type === "line" || spec.type === "scatter";
  return {
    ...(planned as EChartsOption), ...common, dataZoom: [],
    tooltip: (shared
      ? { trigger: "axis", formatter: planAxisTooltip(plan) }
      : planned.tooltip) as EChartsOption["tooltip"],
    // A heatmap keeps the plan's room for its colour scale.
    grid: spec.type === "heatmap" ? planned.grid : panelGrids(spec.panelCount),
    visualMap: planned.visualMap.map((scale) => ({ ...scale, show: showLegend })) as EChartsOption["visualMap"],
    // A box plot has no legend in its export; the labels sit on the axis.
    legend: planned.legend.show ? { ...common.legend, data: planned.legend.data } : { show: false },
  };
}
