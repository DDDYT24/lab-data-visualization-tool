import type { CustomSeriesRenderItem, EChartsOption } from "echarts";

import type {
  ChartAnalysis,
  DataPreview,
  QualityFinding,
} from "@/domain/api-contract";
import {
  chartLineType,
  chartRenderContract,
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

type ChartColorMode = "light" | "dark";

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
  colorMode = "light",
  excludedFindingIds,
  findings,
  preview,
  spec,
  showAxes = true,
  showLegend = true,
  surfacePointLimit,
  surfaceView,
}: {
  analysis?: ChartAnalysis;
  colorMode?: ChartColorMode;
  excludedFindingIds: string[];
  findings: QualityFinding[];
  preview: DataPreview;
  spec: ChartSpec;
  showAxes?: boolean;
  showLegend?: boolean;
  surfacePointLimit?: number;
  surfaceView?: {
    alpha: number;
    beta: number;
    distance: number;
  };
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
  const chartAxisColor = colorMode === "dark" ? "#B4B4B4" : "#475467";
  const chartGridColor = colorMode === "dark" ? "#424242" : gridColor;
  const common = {
    animation: false,
    backgroundColor: spec.export.transparentBackground
      ? "transparent"
      : colorMode === "dark"
        ? "#2F2F2F"
        : spec.export.backgroundColor,
    color: colors,
    legend: showLegend ? legendOption(spec.export.legendPosition) : { show: false },
    textStyle: {
      color: colorMode === "dark" ? "#ECECEC" : "#172033",
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
    // Draw the API's complete intervals: a value-axis bar only retains bin centers
    // and invents a width/gap that does not match the publication renderer.
    const histograms = spec.series.map((series, index) => ({
      series,
      color: colors[index],
      bins: analysis?.preview.histograms.find((item) => item.field === series.field)?.bins ?? [],
    }));
    const panels = Array.from({ length: spec.panelCount }, (_, index) => {
      const bins = histograms.filter((item) => item.series.panel === index + 1).flatMap((item) => item.bins);
      const minimum = bins.length ? Math.min(...bins.map((bin) => bin.start)) : 0;
      const maximum = bins.length ? Math.max(...bins.map((bin) => bin.end)) : 1;
      const padding = (maximum - minimum) * chartRenderContract.histogramAxisPadding;
      return {
        minimum: minimum - padding,
        maximum: maximum + padding,
        frequencyMaximum: Math.max(1, ...bins.map((bin) => bin.count * (1 + chartRenderContract.histogramAxisPadding))),
      };
    });
    return {
      ...common,
      tooltip: { trigger: "item", valueFormatter: (value) => String(value) },
      grid: panelGrids(spec.panelCount),
      xAxis: panels.map((panel, index) => ({
        gridIndex: index, name: axisName(spec.yAxis.title, spec.yAxis.unit), type: "value",
        min: panel.minimum, max: panel.maximum, scale: true, show: showAxes,
        nameLocation: "middle", nameGap: 35,
        axisLabel: { showMinLabel: false, showMaxLabel: false, formatter: (value: number) => Number(value.toPrecision(12)).toString() },
        splitLine: { show: spec.export.gridVisible, lineStyle: { color: chartGridColor } },
      })),
      yAxis: panels.map((panel, index) => ({
        gridIndex: index, name: "Frequency", type: "value", min: 0, max: panel.frequencyMaximum,
        nameLocation: "middle", nameGap: 45,
        axisLabel: { showMaxLabel: false, formatter: (value: number) => Number(value.toPrecision(12)).toString() },
        show: showAxes, splitLine: { show: spec.export.gridVisible, lineStyle: { color: chartGridColor } },
      })),
      series: histograms.map(({ series, bins, color }) => {
        const renderItem: CustomSeriesRenderItem = (_params, api) => {
          const start = api.coord([Number(api.value(0)), Number(api.value(2))]);
          const end = api.coord([Number(api.value(1)), 0]);
          if (!Array.isArray(start) || !Array.isArray(end)) return null;
          return {
            type: "rect",
            shape: { x: start[0], y: start[1], width: end[0] - start[0], height: end[1] - start[1] },
            style: { fill: color, opacity: chartRenderContract.histogramFillOpacity },
          };
        };
        return {
          clip: true, data: bins.map((bin) => [bin.start, bin.end, bin.count]),
          dimensions: ["Interval start", "Interval end", "Frequency"],
          encode: { x: [0, 1], y: 2, tooltip: [0, 1, 2] },
          itemStyle: { color, opacity: chartRenderContract.histogramFillOpacity },
          name: series.label, type: "custom", renderItem,
          xAxisIndex: series.panel - 1, yAxisIndex: series.panel - 1,
        };
      }),
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
    return {
      ...common,
      grid: { left: 90, right: 78, top: 86, bottom: 74, containLabel: true },
      visualMap: {
        show: showLegend,
        calculable: false,
        max: 1,
        min: -1,
        text: ["1", "-1"],
        inRange: { color: spec.export.grayscalePreview
          ? chartRenderContract.surfaceGrayscalePalette
          : chartRenderContract.correlationPalette },
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
    const surfaceDiagnostic = analysis?.preview.surfaceDiagnostics.find(
      (item) => item.panel === 1,
    );
    const analysisData = analysis?.preview.surfacePoints.length
      ? analysis.preview.surfacePoints.map((point) => [point.x, point.y, point.z])
      : fallbackData;
    const fullData = surfaceDiagnostic?.status === "valid" || !surfaceDiagnostic
      ? analysisData
      : [];
    const xs = [...new Set(fullData.map((point) => point[0]))].sort((a, b) => a - b);
    const ys = [...new Set(fullData.map((point) => point[1]))].sort((a, b) => a - b);
    const limit = Math.max(4, surfacePointLimit ?? fullData.length);
    let stride = 1;
    const selectAxis = (values: number[]) => values.filter(
      (_value, index) => index % stride === 0 || index === values.length - 1,
    );
    while (selectAxis(xs).length * selectAxis(ys).length > limit) stride += 1;
    const selectedX = new Set(selectAxis(xs));
    const selectedY = new Set(selectAxis(ys));
    const data = fullData.filter((point) => selectedX.has(point[0]) && selectedY.has(point[1]))
      .sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    // Keep the full-data scale even when the preview uses a reduced grid.
    const zValues = fullData.map((point) => point[2]);
    const minZ = zValues.length ? Math.min(...zValues) : 0;
    const maxZ = zValues.length ? Math.max(...zValues) : 1;
    return {
      ...common,
      grid3D: {
        top: 45,
        height: "70%",
        axisLabel: { show: showAxes, color: chartAxisColor },
        axisPointer: { show: showAxes },
        splitLine: { show: spec.export.gridVisible, lineStyle: { color: chartGridColor } },
        axisLine: { show: showAxes, lineStyle: { color: chartAxisColor } },
        boxDepth: 80,
        boxHeight: 80,
        boxWidth: 120,
        viewControl: {
          alpha: surfaceView?.alpha,
          autoRotate: false,
          beta: surfaceView?.beta,
          distance: surfaceView?.distance ?? 200,
          panSensitivity: 1,
          projection: "perspective",
          rotateSensitivity: 1,
          zoomSensitivity: 1,
        },
      },
      visualMap: {
        show: showLegend,
        dimension: 2,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        inRange: { color: spec.export.grayscalePreview
          ? chartRenderContract.surfaceGrayscalePalette
          : chartRenderContract.surfacePalette },
        max: maxZ,
        min: minZ,
        text: [String(Number(maxZ.toPrecision(5))), String(Number(minZ.toPrecision(5)))],
      },
      xAxis3D: {
        axisLine: { show: showAxes },
        name: showAxes ? axisName(spec.xAxis.title, spec.xAxis.unit) : "",
        type: "value",
      },
      yAxis3D: { axisLine: { show: showAxes }, name: showAxes ? spec.series[0]?.label ?? "Y" : "", type: "value" },
      zAxis3D: {
        axisLine: { show: showAxes },
        name: showAxes ? spec.series[1]?.label ?? spec.yAxis.title : "",
        type: "value",
      },
      series: [{
        data,
        dataShape: [selectedY.size, selectedX.size],
        shading: "color",
        type: "surface",
      }],
    } as EChartsOption;
  }

  const grids = panelGrids(spec.panelCount);
  const xAxes: Array<Record<string, unknown>> = grids.map((_, panelIndex) => ({
    axisLabel: { color: chartAxisColor },
    gridIndex: panelIndex,
    name: axisName(spec.xAxis.title, spec.xAxis.unit),
    nameGap: 30,
    nameLocation: "middle" as const,
    splitLine: {
      lineStyle: { color: chartGridColor },
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
        lineStyle: { color: chartGridColor },
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
