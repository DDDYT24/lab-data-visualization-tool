"use client";

import {
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { useTranslations } from "next-intl";

import type { ChartSpec } from "@/domain/chart-spec";

import { useWorkspaceStore } from "./workspace-store";

/**
 * How the chart looks. None of these change what is computed: the API draws a new plan from the
 * analysis it already holds (analysis-cache-key.ts).
 */
export function ChartAppearancePanel() {
  const t = useTranslations("chart");
  const chartSpec = useWorkspaceStore((state) => state.chartSpec);
  const updateChart = useWorkspaceStore((state) => state.updateChart);
  const customSeriesColorDisabled =
    Boolean(chartSpec.groupField) || chartSpec.export.grayscalePreview ||
    chartSpec.type === "surface3d" || chartSpec.type === "heatmap";
  return (
    <Stack spacing={2.25}>
      <TextField
        fullWidth
        label={t("figureTitle")}
        onChange={(event) => updateChart({ title: event.target.value })}
        size="small"
        value={chartSpec.title}
      />
      <TextField
        fullWidth
        label={t("figureSubtitle")}
        onChange={(event) => updateChart({ subtitle: event.target.value })}
        size="small"
        value={chartSpec.subtitle}
      />
      <TextField
        fullWidth
        label={t("xAxis")}
        onChange={(event) =>
          updateChart({ xAxis: { ...chartSpec.xAxis, title: event.target.value } })
        }
        size="small"
        value={chartSpec.xAxis.title}
      />
      <TextField
        fullWidth
        label={t("yAxis")}
        onChange={(event) =>
          updateChart({ yAxis: { ...chartSpec.yAxis, title: event.target.value } })
        }
        size="small"
        value={chartSpec.yAxis.title}
      />
      <Stack direction="row" spacing={1}>
        <TextField
          fullWidth
          label={t("xUnit")}
          onChange={(event) =>
            updateChart({ xAxis: { ...chartSpec.xAxis, unit: event.target.value } })
          }
          size="small"
          value={chartSpec.xAxis.unit}
        />
        <TextField
          fullWidth
          label={t("yUnit")}
          onChange={(event) =>
            updateChart({ yAxis: { ...chartSpec.yAxis, unit: event.target.value } })
          }
          size="small"
          value={chartSpec.yAxis.unit}
        />
      </Stack>
      {chartSpec.type === "surface3d" || chartSpec.type === "heatmap" ? (
        <Typography color="text.secondary" variant="caption">
          {t("valueColorHelp")}
        </Typography>
      ) : chartSpec.groupField ? (
        <Typography color="text.secondary" variant="caption">
          {t("groupColorHelp")}
        </Typography>
      ) : chartSpec.export.grayscalePreview ? (
        <Typography color="text.secondary" variant="caption">
          {t("grayscaleColorHelp")}
        </Typography>
      ) : null}
      {chartSpec.series.map((series, index) => (
        <Stack key={series.field} spacing={1}>
          <TextField
            fullWidth
            label={t("seriesLabel", { index: index + 1 })}
            onChange={(event) =>
              updateChart({
                series: chartSpec.series.map((item) =>
                  item.field === series.field
                    ? { ...item, label: event.target.value }
                    : item,
                ),
              })
            }
            size="small"
            value={series.label}
          />
          <TextField
            fullWidth
            label={t("seriesColor", { label: series.label })}
            onChange={(event) =>
              updateChart({
                series: chartSpec.series.map((item) =>
                  item.field === series.field
                    ? { ...item, color: event.target.value }
                    : item,
                ),
              })
            }
            disabled={customSeriesColorDisabled}
            size="small"
            type="color"
            value={series.color}
          />
          <FormControl fullWidth size="small">
            <InputLabel id={`series-line-style-${index}`}>
              {t("lineStyle")}
            </InputLabel>
            <Select
              label={t("lineStyle")}
              labelId={`series-line-style-${index}`}
              onChange={(event) =>
                updateChart({
                  series: chartSpec.series.map((item) =>
                    item.field === series.field
                      ? {
                          ...item,
                          lineStyle: event.target.value as ChartSpec["series"][number]["lineStyle"],
                        }
                      : item,
                  ),
                })
              }
              value={series.lineStyle}
            >
              {(["solid", "dashed", "dotted", "dashdot"] as const).map(
                (style) => (
                  <MenuItem key={style} value={style}>
                    {t(`lineStyles.${style}`)}
                  </MenuItem>
                ),
              )}
            </Select>
          </FormControl>
        </Stack>
      ))}
      {/* The fit labels only matter once a fit is drawn. */}
      {chartSpec.fitting.model !== "none" ? (
        <>
          <FormControlLabel
            control={
              <Switch
                checked={chartSpec.fitting.showEquation}
                onChange={(event) =>
                  updateChart({
                    fitting: { ...chartSpec.fitting, showEquation: event.target.checked },
                  })
                }
              />
            }
            label={t("showEquation")}
          />
          <FormControlLabel
            control={
              <Switch
                checked={chartSpec.fitting.showRSquared}
                onChange={(event) =>
                  updateChart({
                    fitting: { ...chartSpec.fitting, showRSquared: event.target.checked },
                  })
                }
              />
            }
            label={t("showRSquared")}
          />
        </>
      ) : null}
    </Stack>
  );
}
