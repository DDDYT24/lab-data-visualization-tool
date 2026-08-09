import renderContract from "../../../contracts/chart-render-v1.json";

import type { ChartSpec } from "./chart-spec";

export const chartRenderContract = renderContract;
export const grayscalePalette = renderContract.grayscalePalette;
export const groupPalette = renderContract.groupPalette;
export const gridColor = renderContract.gridColor;
export const confidenceBandOpacity = renderContract.confidenceBandOpacity;

export function chartLineType(
  style: ChartSpec["series"][number]["lineStyle"],
) {
  return renderContract.lineStyles[style].echarts;
}

export function chartSeriesColor({
  configuredColor,
  grayscale,
  grouped,
  index,
}: {
  configuredColor: string;
  grayscale: boolean;
  grouped: boolean;
  index: number;
}) {
  if (grayscale) return grayscalePalette[index % grayscalePalette.length];
  if (grouped) return groupPalette[index % groupPalette.length];
  return configuredColor;
}
