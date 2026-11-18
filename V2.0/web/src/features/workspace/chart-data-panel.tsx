"use client";

import {
  Alert,
  Checkbox,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  Typography,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material/Select";
import { useTranslations } from "next-intl";

import type { PreviewColumn } from "@/domain/api-contract";
import type { ChartSpec } from "@/domain/chart-spec";

import { useWorkspaceStore } from "./workspace-store";

export const seriesColors = ["#2563EB", "#0F766E", "#D97706", "#7C3AED", "#DC2626"];
const chartTypes: ChartSpec["type"][] = [
  "line",
  "scatter",
  "bar",
  "histogram",
  "box",
  "heatmap",
  "surface3d",
];

/** A surface takes X plus two different numeric fields: Y (series 1) and Z (series 2). */
export function surfaceSetup(chartSpec: ChartSpec) {
  const y = chartSpec.series[0]?.field ?? "";
  const z = chartSpec.series[1]?.field ?? "";
  const incomplete =
    chartSpec.type === "surface3d" &&
    (chartSpec.series.length !== 2 || !y || !z ||
      y === chartSpec.xAxis.field || z === chartSpec.xAxis.field || y === z);
  return { y, z, incomplete };
}

/** What the chart is computed from: its kind, fields, grouping, panels and second axis. */
export function ChartDataPanel({ numericColumns }: { numericColumns: PreviewColumn[] }) {
  const t = useTranslations("chart");
  const chartSpec = useWorkspaceStore((state) => state.chartSpec);
  const preview = useWorkspaceStore((state) => state.preview);
  const updateChart = useWorkspaceStore((state) => state.updateChart);
  const distributionChart = ["histogram", "box"].includes(chartSpec.type);
  const { y: surfaceYField, z: surfaceZField } = surfaceSetup(chartSpec);
  const groupingColumns =
    preview?.columns.filter(
      (column) =>
        column.field !== chartSpec.xAxis.field &&
        !chartSpec.series.some((series) => series.field === column.field),
    ) ?? [];

  const setSeriesFields = (event: SelectChangeEvent<string[]>) => {
    const fields =
      typeof event.target.value === "string"
        ? event.target.value.split(",")
        : event.target.value;
    const nextSeries = fields.map((field, index) => {
      const existing = chartSpec.series.find((series) => series.field === field);
      const column = numericColumns.find((item) => item.field === field);
      return (
        existing ?? {
          field,
          label: column?.label ?? field,
          color: seriesColors[index % seriesColors.length],
          lineStyle: "solid" as const,
          panel: 1,
          yAxis: "primary" as const,
        }
      );
    });
    if (nextSeries.length > 0) {
      updateChart({
        series: nextSeries,
        yAxis: {
          ...chartSpec.yAxis,
          field: nextSeries[0].field,
          title: nextSeries[0].label,
        },
      });
    }
  };

  const setSurfaceField = (index: 0 | 1, field: string) => {
    const column = numericColumns.find((item) => item.field === field);
    if (!column || field === chartSpec.xAxis.field) return;
    const nextSeries = chartSpec.series.slice(0, 2);
    const baseSeries = nextSeries[index] ?? {
      field,
      label: column.label,
      color: seriesColors[index % seriesColors.length],
      lineStyle: "solid" as const,
      panel: 1,
      yAxis: "primary" as const,
    };
    nextSeries[index] = {
      ...baseSeries,
      field,
      label: column.label,
      panel: 1,
      yAxis: "primary",
    };
    updateChart({
      series: nextSeries,
      ...(index === 0
        ? {
            yAxis: {
              ...chartSpec.yAxis,
              field,
              title: column.label,
              unit: column.unit ?? "",
            },
          }
        : {}),
    });
  };

  return (
    <Stack spacing={2.25}>
      <FormControl fullWidth size="small">
        <InputLabel id="chart-type-label">{t("chartType")}</InputLabel>
        <Select
          label={t("chartType")}
          labelId="chart-type-label"
          onChange={(event) => {
            const type = event.target.value as ChartSpec["type"];
            const singlePanel = ["heatmap", "surface3d"].includes(type);
            updateChart({
              type,
              ...(singlePanel
                ? {
                    panelCount: 1,
                    series: chartSpec.series
                      .slice(0, type === "surface3d" ? 2 : undefined)
                      .map((series) => ({ ...series, panel: 1 })),
                  }
                : {}),
              ...(!["line", "scatter", "bar"].includes(type)
                ? { groupField: null }
                : {}),
            });
          }}
          value={chartSpec.type}
        >
          {chartTypes.map((type) => (
            <MenuItem
              disabled={numericColumns.length < 2 && ["line", "scatter", "bar"].includes(type)}
              key={type}
              value={type}
            >
              {t(`types.${type}`)}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      {numericColumns.length === 1 ? (
        <Alert severity="info">{t("singleValueHelp")}</Alert>
      ) : null}
      {distributionChart ? null : (
        <FormControl fullWidth size="small">
          <InputLabel id="chart-x-field-label">{t("xField")}</InputLabel>
          <Select
            label={t("xField")}
            labelId="chart-x-field-label"
            onChange={(event) => {
              const column = numericColumns.find(
                (item) => item.field === event.target.value,
              );
              if (column) {
                const previousXColumn = numericColumns.find(
                  (item) => item.field === chartSpec.xAxis.field,
                );
                const nextSeries = chartSpec.series.map((series) =>
                  series.field === column.field && previousXColumn
                    ? {
                        ...series,
                        field: previousXColumn.field,
                        label: previousXColumn.label,
                      }
                    : series,
                );
                updateChart({
                  xAxis: {
                    field: column.field,
                    title: column.label,
                    unit: column.unit ?? "",
                  },
                  series: nextSeries,
                  yAxis: {
                    ...chartSpec.yAxis,
                    field: nextSeries[0].field,
                    title: nextSeries[0].label,
                    unit:
                      numericColumns.find(
                        (item) => item.field === nextSeries[0].field,
                      )?.unit ?? "",
                  },
                });
              }
            }}
            value={chartSpec.xAxis.field}
          >
            {numericColumns.map((column) => (
              <MenuItem key={column.field} value={column.field}>
                {column.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}
      {chartSpec.type === "surface3d" ? (
        <Stack spacing={1.5}>
          <FormControl fullWidth size="small">
            <InputLabel id="surface-y-field-label">{t("surfaceYField")}</InputLabel>
            <Select
              label={t("surfaceYField")}
              labelId="surface-y-field-label"
              onChange={(event) => setSurfaceField(0, event.target.value)}
              value={surfaceYField}
            >
              {!surfaceYField ? <MenuItem value="">{t("chooseField")}</MenuItem> : null}
              {numericColumns
                .filter(
                  (column) =>
                    column.field === surfaceYField ||
                    (column.field !== chartSpec.xAxis.field &&
                      column.field !== surfaceZField),
                )
                .map((column) => (
                  <MenuItem key={column.field} value={column.field}>
                    {column.label}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
          <FormControl disabled={!surfaceYField} fullWidth size="small">
            <InputLabel id="surface-z-field-label">{t("surfaceZField")}</InputLabel>
            <Select
              label={t("surfaceZField")}
              labelId="surface-z-field-label"
              onChange={(event) => setSurfaceField(1, event.target.value)}
              value={surfaceZField}
            >
              {!surfaceZField ? <MenuItem value="">{t("chooseField")}</MenuItem> : null}
              {numericColumns
                .filter(
                  (column) =>
                    column.field === surfaceZField ||
                    (column.field !== chartSpec.xAxis.field &&
                      column.field !== surfaceYField),
                )
                .map((column) => (
                  <MenuItem key={column.field} value={column.field}>
                    {column.label}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </Stack>
      ) : (
        <FormControl fullWidth size="small">
          <InputLabel id="chart-series-field-label">{t("seriesFields")}</InputLabel>
          <Select
            label={t("seriesFields")}
            labelId="chart-series-field-label"
            multiple
            onChange={setSeriesFields}
            renderValue={(selected) =>
              chartSpec.series
                .filter((series) => selected.includes(series.field))
                .map((series) => series.label)
                .join(", ")
            }
            value={chartSpec.series.map((series) => series.field)}
          >
            {numericColumns
              .filter((column) => distributionChart || column.field !== chartSpec.xAxis.field)
              .map((column) => (
                <MenuItem key={column.field} value={column.field}>
                  <Checkbox
                    checked={chartSpec.series.some(
                      (series) => series.field === column.field,
                    )}
                  />
                  {column.label}
                </MenuItem>
              ))}
          </Select>
        </FormControl>
      )}
      <FormControl
        disabled={!(["line", "scatter", "bar"] as ChartSpec["type"][]).includes(chartSpec.type)}
        fullWidth
        size="small"
      >
        <InputLabel id="chart-group-field-label">{t("groupField")}</InputLabel>
        <Select
          label={t("groupField")}
          labelId="chart-group-field-label"
          onChange={(event) =>
            updateChart({ groupField: event.target.value || null })
          }
          value={chartSpec.groupField ?? ""}
        >
          <MenuItem value="">{t("noGrouping")}</MenuItem>
          {groupingColumns.map((column) => (
            <MenuItem key={column.field} value={column.field}>
              {column.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      {distributionChart ? (
        <Typography color="text.secondary" variant="caption">
          {t("distributionPooled")}
        </Typography>
      ) : null}
      <FormControl fullWidth size="small">
        <InputLabel id="panel-count-label">{t("panelCount")}</InputLabel>
        <Select
          disabled={["heatmap", "surface3d"].includes(chartSpec.type)}
          label={t("panelCount")}
          labelId="panel-count-label"
          onChange={(event) => {
            const panelCount = Number(event.target.value) as 1 | 2 | 3 | 4;
            updateChart({
              panelCount,
              series: chartSpec.series.map((series) => ({
                ...series,
                panel: Math.min(series.panel, panelCount),
              })),
            });
          }}
          value={String(chartSpec.panelCount)}
        >
          {[1, 2, 3, 4].map((count) => (
            <MenuItem key={count} value={String(count)}>{count}</MenuItem>
          ))}
        </Select>
      </FormControl>
      {["heatmap", "surface3d"].includes(chartSpec.type) ? (
        <Typography color="text.secondary" variant="caption">
          {t("singlePanelType")}
        </Typography>
      ) : null}
      {chartSpec.panelCount > 1
        ? chartSpec.series.map((series) => (
            <FormControl fullWidth key={series.field} size="small">
              <InputLabel id={`panel-${series.field}`}>{t("seriesPanel", { label: series.label })}</InputLabel>
              <Select
                label={t("seriesPanel", { label: series.label })}
                labelId={`panel-${series.field}`}
                onChange={(event) =>
                  updateChart({
                    series: chartSpec.series.map((item) =>
                      item.field === series.field
                        ? { ...item, panel: Number(event.target.value) }
                        : item,
                    ),
                  })
                }
                value={String(series.panel)}
              >
                {Array.from({ length: chartSpec.panelCount }, (_, index) => (
                  <MenuItem key={index + 1} value={String(index + 1)}>{index + 1}</MenuItem>
                ))}
              </Select>
            </FormControl>
          ))
        : null}
      <FormControlLabel
        control={
          <Switch
            checked={chartSpec.secondaryYAxis.enabled}
            onChange={(event) =>
              updateChart({
                secondaryYAxis: {
                  ...chartSpec.secondaryYAxis,
                  enabled: event.target.checked,
                },
              })
            }
          />
        }
        label={t("secondaryAxis")}
      />
      {chartSpec.secondaryYAxis.enabled ? (
        <FormControl fullWidth size="small">
          <InputLabel id="secondary-field-label">{t("secondaryField")}</InputLabel>
          <Select
            label={t("secondaryField")}
            labelId="secondary-field-label"
            onChange={(event) => {
              const field = event.target.value;
              const column = numericColumns.find((item) => item.field === field);
              updateChart({
                secondaryYAxis: {
                  ...chartSpec.secondaryYAxis,
                  field,
                  title: column?.label ?? field,
                  unit: column?.unit ?? "",
                },
                series: chartSpec.series.map((series) => ({
                  ...series,
                  yAxis: series.field === field ? "secondary" : series.yAxis,
                })),
              });
            }}
            value={chartSpec.secondaryYAxis.field ?? ""}
          >
            {chartSpec.series.map((series) => (
              <MenuItem key={series.field} value={series.field}>{series.label}</MenuItem>
            ))}
          </Select>
        </FormControl>
      ) : null}
    </Stack>
  );
}
