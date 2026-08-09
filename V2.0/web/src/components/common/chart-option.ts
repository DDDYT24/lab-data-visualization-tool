import type { CustomSeriesRenderItem, EChartsOption } from "echarts";

import type {
  ChartAnalysis,
  DataPreview,
  QualityFinding,
} from "@/domain/api-contract";
import {
  chartLineType,
  chartSeriesColor,
  confidenceBandOpacity,
  gridColor,
} from "@/domain/chart-render-contract";
import type { ChartSpec } from "@/domain/chart-spec";

type SeriesOptionRecord = Record<string, unknown>;

function axisName(title: string, unit: string) {
  return unit ? `${title} (${unit})` : title;
}

function visibleRows(
  preview: DataPreview,
  findings: QualityFinding[],
  excludedFindingIds: string[],
) {
  const excludedRows = new Set(
    findings
      .filter((finding) => excludedFindingIds.includes(finding.id))
      .flatMap((finding) => finding.rowIds.map(String)),
  );
  return preview.rows.filter((row) => !excludedRows.has(String(row.rowId)));
}

function numericValues(preview: DataPreview, field: string) {
  return preview.rows
    .map((row) => row[field])
    .filter((value): value is number => typeof value === "number");
}

function histogram(values: number[]) {
  if (values.length === 0) return [];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (minimum === maximum) return [[minimum, values.length]];
  const count = Math.min(16, Math.max(5, Math.ceil(Math.sqrt(values.length))));
  const width = (maximum - minimum) / count;
  const bins = Array.from({ length: count }, () => 0);
  for (const value of values) {
    const index = Math.min(Math.floor((value - minimum) / width), count - 1);
    bins[index] += 1;
  }
  return bins.map((value, index) => [minimum + width * (index + 0.5), value]);
}

function quantile(sorted: number[], fraction: number) {
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function boxValues(values: number[]) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return [
    sorted[0],
    quantile(sorted, 0.25),
    quantile(sorted, 0.5),
    quantile(sorted, 0.75),
    sorted.at(-1)!,
  ];
}

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

function analysisSeries(
  analysis: ChartAnalysis | undefined,
  spec: ChartSpec,
  panelAxisIndexes: Array<{ primary: number; secondary: number | null }>,
) {
  if (!analysis) return [];
  const derived: SeriesOptionRecord[] = [];
  for (const [resultIndex, result] of analysis.series.entries()) {
    const sourceSeries = spec.series.find((item) => item.field === result.field);
    if (!sourceSeries) continue;
    const seriesColor = chartSeriesColor({
      configuredColor: sourceSeries.color,
      grayscale: spec.export.grayscalePreview,
      grouped: Boolean(spec.groupField),
      index: resultIndex,
    });
    const panelIndex = Math.min(sourceSeries.panel, spec.panelCount) - 1;
    const yAxisIndex =
      sourceSeries.yAxis === "secondary"
        ? panelAxisIndexes[panelIndex].secondary ?? panelAxisIndexes[panelIndex].primary
        : panelAxisIndexes[panelIndex].primary;
    if (result.fit) {
      derived.push({
        data: result.fit.points.map((point) => [point.x, point.y]),
        lineStyle: { color: seriesColor, type: "dashed", width: 1.5 },
        name: `${result.label} fit`,
        showSymbol: false,
        silent: true,
        type: "line",
        xAxisIndex: panelIndex,
        yAxisIndex,
      });
      if (spec.fitting.confidenceBand) {
        const bounded = result.fit.points.filter(
          (point) => point.lower !== null && point.upper !== null,
        );
        const stack = `confidence-${sourceSeries.field}-${panelIndex}-${resultIndex}`;
        derived.push({
          data: bounded.map((point) => [point.x, point.lower]),
          lineStyle: { opacity: 0 },
          name: `${sourceSeries.label} confidence baseline`,
          showSymbol: false,
          silent: true,
          stack,
          stackStrategy: "all",
          type: "line",
          xAxisIndex: panelIndex,
          yAxisIndex,
        });
        derived.push({
          areaStyle: { color: seriesColor, opacity: confidenceBandOpacity },
          data: bounded.map((point) => [
            point.x,
            (point.upper ?? 0) - (point.lower ?? 0),
          ]),
          lineStyle: { opacity: 0 },
          name: `${sourceSeries.label} confidence band`,
          showSymbol: false,
          silent: true,
          stack,
          stackStrategy: "all",
          type: "line",
          xAxisIndex: panelIndex,
          yAxisIndex,
        });
      }
    }
    if (result.uncertainty) {
      const renderErrorBar: CustomSeriesRenderItem = (_params, api) => {
        const x = Number(api.value(0));
        const y = Number(api.value(1));
        const error = Number(api.value(2));
        const high = api.coord([x, y + error]);
        const low = api.coord([x, y - error]);
        if (!Array.isArray(high) || !Array.isArray(low)) return null;
        return {
          type: "group",
          children: [
            {
              type: "line",
              shape: { x1: high[0], y1: high[1], x2: low[0], y2: low[1] },
              style: { stroke: seriesColor, lineWidth: 1 },
            },
            {
              type: "line",
              shape: { x1: high[0] - 4, y1: high[1], x2: high[0] + 4, y2: high[1] },
              style: { stroke: seriesColor, lineWidth: 1 },
            },
            {
              type: "line",
              shape: { x1: low[0] - 4, y1: low[1], x2: low[0] + 4, y2: low[1] },
              style: { stroke: seriesColor, lineWidth: 1 },
            },
          ],
        };
      };
      derived.push({
        data: result.uncertainty.points.map((point) => [point.x, point.y, point.error]),
        name: `${result.label} uncertainty`,
        renderItem: renderErrorBar,
        silent: true,
        type: "custom",
        xAxisIndex: panelIndex,
        yAxisIndex,
      });
    }
  }
  return derived;
}

export function buildChartOption({
  analysis,
  excludedFindingIds,
  findings,
  preview,
  spec,
}: {
  analysis?: ChartAnalysis;
  excludedFindingIds: string[];
  findings: QualityFinding[];
  preview: DataPreview;
  spec: ChartSpec;
}): EChartsOption {
  const rows = visibleRows(preview, findings, excludedFindingIds);
  const colors = spec.series.map((series, index) =>
    chartSeriesColor({
      configuredColor: series.color,
      grayscale: spec.export.grayscalePreview,
      grouped: false,
      index,
    }),
  );
  const common = {
    animation: false,
    backgroundColor: spec.export.transparentBackground
      ? "transparent"
      : spec.export.backgroundColor,
    color: colors,
    legend: legendOption(spec.export.legendPosition),
    textStyle: {
      color: "#172033",
      fontFamily: spec.export.fontFamily,
      fontSize: spec.export.fontSize,
    },
    title: {
      left: "center",
      subtext: spec.subtitle || undefined,
      text: spec.title,
      textStyle: { fontSize: Math.max(16, spec.export.fontSize + 6), fontWeight: 600 },
    },
    tooltip: { trigger: "axis" as const },
  };

  if (spec.type === "histogram") {
    return {
      ...common,
      grid: { left: 72, right: 38, top: 86, bottom: 62, containLabel: true },
      xAxis: { name: axisName(spec.yAxis.title, spec.yAxis.unit), type: "value" },
      yAxis: { name: "Frequency", type: "value" },
      series: spec.series.map((series, index) => ({
        barGap: "5%",
        data:
          analysis?.preview.histograms
            .find((item) => item.field === series.field)
            ?.bins.map((bin) => [(bin.start + bin.end) / 2, bin.count]) ??
          histogram(numericValues({ ...preview, rows }, series.field)),
        itemStyle: { color: colors[index], opacity: 0.82 },
        name: series.label,
        type: "bar",
      })),
    };
  }

  if (spec.type === "box") {
    const boxes = analysis?.preview.boxes.length
      ? analysis.preview.boxes.map((box) => ({
          label: box.label,
          values: [box.minimum, box.q1, box.median, box.q3, box.maximum],
        }))
      : spec.series.flatMap((series) => {
          const values = boxValues(numericValues({ ...preview, rows }, series.field));
          return values ? [{ label: series.label, values }] : [];
        });
    return {
      ...common,
      grid: { left: 72, right: 38, top: 86, bottom: 62, containLabel: true },
      xAxis: { data: boxes.map((box) => box.label), type: "category" },
      yAxis: { name: axisName(spec.yAxis.title, spec.yAxis.unit), type: "value" },
      series: [
        {
          data: boxes.map((box) => box.values),
          name: "Distribution",
          type: "boxplot",
        },
      ],
    };
  }

  if (spec.type === "heatmap") {
    const heatmap = analysis?.preview.heatmaps[0];
    const labels = heatmap?.labels ?? spec.series.map((series) => series.label);
    const values = heatmap
      ? heatmap.matrix.flatMap((matrixRow, yIndex) =>
          matrixRow.map((value, xIndex) => [xIndex, yIndex, value]),
        )
      : [];
    const numeric = values
      .map((entry) => entry[2])
      .filter((value): value is number => typeof value === "number");
    return {
      ...common,
      grid: { left: 90, right: 78, top: 86, bottom: 74, containLabel: true },
      visualMap: {
        calculable: true,
        max: numeric.length ? Math.max(...numeric) : 1,
        min: numeric.length ? Math.min(...numeric) : 0,
        orient: "vertical",
        right: 8,
      },
      xAxis: {
        data: labels,
        name: "Correlation",
        type: "category",
      },
      yAxis: { data: labels, type: "category" },
      series: [{ data: values, name: "Correlation", type: "heatmap" }],
    };
  }

  if (spec.type === "surface3d") {
    const yField = spec.series[0]?.field;
    const zField = spec.series[1]?.field ?? spec.series[0]?.field;
    const fallbackData = rows.flatMap((row, index) => {
      const x = row[spec.xAxis.field];
      const y = yField ? row[yField] : index;
      const z = zField ? row[zField] : null;
      return typeof x === "number" && typeof y === "number" && typeof z === "number"
        ? [[x, y, z]]
        : [];
    });
    const data = analysis?.preview.surfacePoints.length
      ? analysis.preview.surfacePoints.map((point) => [point.x, point.y, point.z])
      : fallbackData;
    const zValues = data.map((point) => point[2]);
    return {
      ...common,
      grid3D: {
        axisLine: { lineStyle: { color: "#667085" } },
        boxDepth: 80,
        boxHeight: 80,
        boxWidth: 120,
        viewControl: { projection: "perspective" },
      },
      visualMap: {
        max: zValues.length ? Math.max(...zValues) : 1,
        min: zValues.length ? Math.min(...zValues) : 0,
      },
      xAxis3D: { name: axisName(spec.xAxis.title, spec.xAxis.unit), type: "value" },
      yAxis3D: { name: spec.series[0]?.label ?? "Y", type: "value" },
      zAxis3D: { name: spec.series[1]?.label ?? spec.yAxis.title, type: "value" },
      series: [{ data, shading: "lambert", type: "surface" }],
    } as EChartsOption;
  }

  const grids = panelGrids(spec.panelCount);
  const xAxes: Array<Record<string, unknown>> = grids.map((_, panelIndex) => ({
    axisLabel: { color: "#475467" },
    gridIndex: panelIndex,
    name: axisName(spec.xAxis.title, spec.xAxis.unit),
    nameGap: 30,
    nameLocation: "middle" as const,
    splitLine: {
      lineStyle: { color: gridColor },
      show: spec.export.gridVisible,
    },
    type: spec.type === "bar" ? ("category" as const) : ("value" as const),
  }));
  const yAxes: Array<Record<string, unknown>> = [];
  const panelAxisIndexes = grids.map((_, panelIndex) => {
    const primary = yAxes.length;
    yAxes.push({
      gridIndex: panelIndex,
      name: axisName(spec.yAxis.title, spec.yAxis.unit),
      nameGap: 42,
      nameLocation: "middle",
      splitLine: {
        lineStyle: { color: gridColor },
        show: spec.export.gridVisible,
      },
      type: "value",
    });
    let secondary: number | null = null;
    if (spec.secondaryYAxis.enabled) {
      secondary = yAxes.length;
      yAxes.push({
        gridIndex: panelIndex,
        name: axisName(spec.secondaryYAxis.title, spec.secondaryYAxis.unit),
        position: "right",
        splitLine: { show: false },
        type: "value",
      });
    }
    return { primary, secondary };
  });
  const plottedSeries =
    spec.groupField && analysis
      ? analysis.series.flatMap((result) => {
          const source = spec.series.find(
            (series) => series.field === result.field && series.panel === result.panel,
          );
          return source ? [{ result, source }] : [];
        })
      : spec.series.map((source) => ({
          result: analysis?.series.find(
            (item) => item.field === source.field && item.panel === source.panel,
          ),
          source,
        }));
  const barCategories = grids.map((_, panelIndex) => {
    const derivedCategories = plottedSeries
      .filter(({ source }) => source.panel === panelIndex + 1)
      .flatMap(({ result }) => result?.points.map((point) => String(point.x ?? "")) ?? []);
    return Array.from(
      new Set(
        derivedCategories.length
          ? derivedCategories
          : rows.map((row) => String(row[spec.xAxis.field] ?? row.rowId)),
      ),
    );
  });
  const sourceSeries: SeriesOptionRecord[] = plottedSeries.map(({ result, source }, index) => {
    const series = source;
    const panelIndex = Math.min(series.panel, spec.panelCount) - 1;
    const axisIndexes = panelAxisIndexes[panelIndex];
    const yAxisIndex =
      series.yAxis === "secondary"
        ? axisIndexes.secondary ?? axisIndexes.primary
        : axisIndexes.primary;
    const derivedPoints = result?.points;
    const points = derivedPoints
      ? derivedPoints
          .filter(
            (point): point is typeof point & { x: string | number } =>
              typeof point.x === "number" || typeof point.x === "string",
          )
          .map((point) => [point.x, point.y] as [string | number, number])
      : rows
          .map((row) => [row[spec.xAxis.field], row[series.field]])
          .filter(
            (point): point is [string | number, number] =>
              (typeof point[0] === "number" || typeof point[0] === "string") &&
              typeof point[1] === "number",
          );
    const seriesColor = chartSeriesColor({
      configuredColor: series.color,
      grayscale: spec.export.grayscalePreview,
      grouped: Boolean(spec.groupField),
      index: spec.groupField ? index : Math.max(spec.series.indexOf(series), 0),
    });
    const barValues = new Map(points.map((point) => [String(point[0]), point[1]]));
    return {
      data:
        spec.type === "bar"
          ? barCategories[panelIndex].map((category) => barValues.get(category) ?? null)
          : points,
      emphasis: { focus: "series" },
      itemStyle: { color: seriesColor },
      lineStyle: {
        color: seriesColor,
        type: chartLineType(series.lineStyle),
        width: spec.export.lineWidth,
      },
      name: result?.label ?? series.label,
      showSymbol: spec.export.markerSize > 1,
      symbolSize: spec.export.markerSize,
      type: spec.type,
      xAxisIndex: panelIndex,
      yAxisIndex,
    };
  });
  if (spec.type === "bar") {
    for (let panelIndex = 0; panelIndex < xAxes.length; panelIndex += 1) {
      xAxes[panelIndex].data = barCategories[panelIndex];
    }
  }

  return {
    ...common,
    grid: grids,
    series: [
      ...sourceSeries,
      ...analysisSeries(analysis, spec, panelAxisIndexes),
    ] as EChartsOption["series"],
    xAxis: xAxes,
    yAxis: yAxes,
  };
}
