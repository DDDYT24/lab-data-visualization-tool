"use client";

import {
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  Typography,
} from "@mui/material";
import { useTranslations } from "next-intl";

import type { ChartSpec } from "@/domain/chart-spec";

import { useWorkspaceStore } from "./workspace-store";

/** How the chart is computed: the fit, its interval and the error bars. */
export function ChartMethodPanel({ numericColumns }: { numericColumns: Column[] }) {
  const t = useTranslations("chart");
  const chartSpec = useWorkspaceStore((state) => state.chartSpec);
  const updateChart = useWorkspaceStore((state) => state.updateChart);
  return (
    <Stack spacing={2}>
      <FormControl fullWidth size="small">
        <InputLabel id="fit-model-label">{t("fitModel")}</InputLabel>
        <Select
          disabled={["heatmap", "surface3d"].includes(chartSpec.type)}
          label={t("fitModel")}
          labelId="fit-model-label"
          onChange={(event) =>
            updateChart({
              fitting: {
                ...chartSpec.fitting,
                model: event.target.value as ChartSpec["fitting"]["model"],
                intervalKind: ["linear", "polynomial"].includes(event.target.value)
                  ? chartSpec.fitting.intervalKind
                  : "pointwise-mean",
              },
            })
          }
          value={chartSpec.fitting.model}
        >
          {(["none", "linear", "polynomial", "exponential", "logarithmic", "power"] as const).map(
            (model) => (
              <MenuItem key={model} value={model}>
                {t(`fitModels.${model}`)}
              </MenuItem>
            ),
          )}
        </Select>
      </FormControl>
      <FormControl fullWidth size="small">
        <InputLabel id="fit-method-label">{t("fitMethod")}</InputLabel>
        <Select
          disabled={chartSpec.fitting.model === "none"}
          label={t("fitMethod")}
          labelId="fit-method-label"
          onChange={(event) =>
            updateChart({
              fitting: {
                ...chartSpec.fitting,
                fitMethod: event.target.value as ChartSpec["fitting"]["fitMethod"],
                confidenceBand:
                  event.target.value === "robust-huber"
                    ? false
                    : chartSpec.fitting.confidenceBand,
                intervalKind:
                  event.target.value === "robust-huber"
                    ? "pointwise-mean"
                    : chartSpec.fitting.intervalKind,
              },
            })
          }
          value={chartSpec.fitting.fitMethod}
        >
          {(["ordinary-least-squares", "weighted-least-squares", "robust-huber"] as const).map(
            (method) => (
            <MenuItem key={method} value={method}>
              {t(`fitMethods.${method}`)}
            </MenuItem>
            ),
          )}
        </Select>
      </FormControl>
      {chartSpec.fitting.fitMethod === "weighted-least-squares" ? (
        <Typography color="text.secondary" variant="caption">
          {t("weightedFitHelp")}
        </Typography>
      ) : null}
      {chartSpec.fitting.fitMethod === "robust-huber" ? (
        <Typography color="text.secondary" variant="caption">
          {t("robustFitHelp")}
        </Typography>
      ) : null}
      {chartSpec.fitting.model === "polynomial" ? (
        <FormControl fullWidth size="small">
          <InputLabel id="polynomial-order-label">{t("polynomialOrder")}</InputLabel>
          <Select
            label={t("polynomialOrder")}
            labelId="polynomial-order-label"
            onChange={(event) =>
              updateChart({
                fitting: {
                  ...chartSpec.fitting,
                  polynomialOrder: Number(event.target.value) as 1 | 2 | 3,
                },
              })
            }
            value={String(chartSpec.fitting.polynomialOrder)}
          >
            <MenuItem value="1">1</MenuItem>
            <MenuItem value="2">2</MenuItem>
            <MenuItem value="3">3</MenuItem>
          </Select>
        </FormControl>
      ) : null}
      <FormControlLabel
        control={
          <Switch
            checked={chartSpec.fitting.confidenceBand}
            disabled={chartSpec.fitting.fitMethod === "robust-huber"}
            onChange={(event) =>
              updateChart({
                fitting: { ...chartSpec.fitting, confidenceBand: event.target.checked },
              })
            }
          />
        }
        label={t("confidenceBand")}
      />
      {chartSpec.fitting.confidenceBand ? (
        <>
          <FormControl fullWidth size="small">
            <InputLabel id="interval-kind-label">{t("intervalKind")}</InputLabel>
            <Select
              label={t("intervalKind")}
              labelId="interval-kind-label"
              onChange={(event) =>
                updateChart({
                  fitting: {
                    ...chartSpec.fitting,
                    intervalKind: event.target.value as ChartSpec["fitting"]["intervalKind"],
                    confidenceMethod:
                      event.target.value === "pointwise-mean"
                        ? chartSpec.fitting.confidenceMethod
                        : "student-t",
                  },
                })
              }
              value={chartSpec.fitting.intervalKind}
            >
              {(["pointwise-mean", "prediction", "simultaneous"] as const)
                .filter(
                  (kind) =>
                    kind === "pointwise-mean" ||
                    ["linear", "polynomial"].includes(chartSpec.fitting.model),
                )
                .map((kind) => (
                  <MenuItem key={kind} value={kind}>
                    {t(`intervalKinds.${kind}`)}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
          <FormControl fullWidth size="small">
            <InputLabel id="confidence-method-label">{t("confidenceMethod")}</InputLabel>
            <Select
              disabled={chartSpec.fitting.intervalKind !== "pointwise-mean"}
              label={t("confidenceMethod")}
              labelId="confidence-method-label"
              onChange={(event) =>
                updateChart({
                  fitting: {
                    ...chartSpec.fitting,
                    confidenceMethod: event.target.value as ChartSpec["fitting"]["confidenceMethod"],
                  },
                })
              }
              value={chartSpec.fitting.confidenceMethod}
            >
              {(["student-t", "bootstrap"] as const)
                .filter(
                  (method) =>
                    chartSpec.fitting.intervalKind === "pointwise-mean" ||
                    method === "student-t",
                )
                .map((method) => (
                  <MenuItem key={method} value={method}>
                    {t(`confidenceMethods.${method}`)}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </>
      ) : null}
      <FormControl fullWidth size="small">
        <InputLabel id="uncertainty-mode-label">{t("errorBars")}</InputLabel>
        <Select
          label={t("errorBars")}
          labelId="uncertainty-mode-label"
          onChange={(event) =>
            updateChart({
              uncertainty: {
                ...chartSpec.uncertainty,
                mode: event.target.value as ChartSpec["uncertainty"]["mode"],
              },
            })
          }
          value={chartSpec.uncertainty.mode}
        >
          {(["none", "standard-deviation", "standard-error", "confidence-interval", "column"] as const).map(
            (mode) => (
              <MenuItem key={mode} value={mode}>
                {t(`uncertaintyModes.${mode}`)}
              </MenuItem>
            ),
          )}
        </Select>
      </FormControl>
      {chartSpec.uncertainty.mode === "column" ? (
        <FormControl fullWidth size="small">
          <InputLabel id="error-field-label">{t("errorField")}</InputLabel>
          <Select
            label={t("errorField")}
            labelId="error-field-label"
            onChange={(event) =>
              updateChart({
                uncertainty: {
                  ...chartSpec.uncertainty,
                  errorField: event.target.value,
                },
              })
            }
            value={chartSpec.uncertainty.errorField ?? ""}
          >
            {numericColumns.map((column) => (
              <MenuItem key={column.field} value={column.field}>
                {column.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      ) : null}
      {(chartSpec.fitting.confidenceBand ||
        chartSpec.uncertainty.mode === "confidence-interval") ? (
        <FormControl fullWidth size="small">
          <InputLabel id="confidence-level-label">{t("confidenceLevel")}</InputLabel>
          <Select
            label={t("confidenceLevel")}
            labelId="confidence-level-label"
            onChange={(event) => {
              const confidenceLevel = Number(event.target.value) as 90 | 95 | 99;
              updateChart({
                fitting: { ...chartSpec.fitting, confidenceLevel },
                uncertainty: { ...chartSpec.uncertainty, confidenceLevel },
              });
            }}
            value={String(chartSpec.fitting.confidenceLevel)}
          >
            <MenuItem value="90">90%</MenuItem>
            <MenuItem value="95">95%</MenuItem>
            <MenuItem value="99">99%</MenuItem>
          </Select>
        </FormControl>
      ) : null}
    </Stack>
  );
}

type Column = NonNullable<ReturnType<typeof useWorkspaceStore.getState>["preview"]>["columns"][number];
