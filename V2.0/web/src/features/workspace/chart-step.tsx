"use client";

import ArrowBackRoundedIcon from "@mui/icons-material/ArrowBackRounded";
import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Button,
  Checkbox,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material/Select";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { ChartCanvas } from "@/components/common/chart-canvas";
import type { ChartAnalysis } from "@/domain/api-contract";
import type { ChartSpec } from "@/domain/chart-spec";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

import { StepLayout } from "./step-layout";
import { useWorkspaceStore } from "./workspace-store";

const seriesColors = ["#2563EB", "#0F766E", "#D97706", "#7C3AED", "#DC2626"];
const chartTypes: ChartSpec["type"][] = [
  "line",
  "scatter",
  "bar",
  "histogram",
  "box",
  "heatmap",
  "surface3d",
];

export function ChartStep({
  onBack,
  onContinue,
}: {
  onBack: () => void;
  onContinue: () => void;
}) {
  const t = useTranslations("chart");
  const projectId = useWorkspaceStore((state) => state.projectId);
  const chartSpec = useWorkspaceStore((state) => state.chartSpec);
  const issueActions = useWorkspaceStore((state) => state.issueActions);
  const preview = useWorkspaceStore((state) => state.preview);
  const quality = useWorkspaceStore((state) => state.quality);
  const updateChart = useWorkspaceStore((state) => state.updateChart);
  const numericColumns = preview?.columns.filter((column) => column.kind === "number") ?? [];
  const surfaceYField = chartSpec.series[0]?.field ?? "";
  const surfaceZField = chartSpec.series[1]?.field ?? "";
  const surfaceNeedsAnotherField =
    chartSpec.type === "surface3d" &&
    (chartSpec.series.length !== 2 ||
      !surfaceYField ||
      !surfaceZField ||
      surfaceYField === chartSpec.xAxis.field ||
      surfaceZField === chartSpec.xAxis.field ||
      surfaceYField === surfaceZField);
  const groupingColumns =
    preview?.columns.filter(
      (column) =>
        column.field !== chartSpec.xAxis.field &&
        !chartSpec.series.some((series) => series.field === column.field),
    ) ?? [];
  const excludedFindingIds = Object.entries(issueActions)
    .filter(([, action]) => ["exclude", "remove"].includes(action))
    .map(([findingId]) => findingId);
  const analysisKey = JSON.stringify({
    fitting: chartSpec.fitting,
    groupField: chartSpec.groupField,
    issueActions,
    panelCount: chartSpec.panelCount,
    secondaryYAxis: chartSpec.secondaryYAxis,
    series: chartSpec.series,
    type: chartSpec.type,
    uncertainty: chartSpec.uncertainty,
    xAxis: chartSpec.xAxis.field,
  });
  const analysisQuery = useQuery({
    queryKey: ["chart-analysis", projectId, analysisKey],
    queryFn: ({ signal }) => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.analyzeChart(projectId, chartSpec, signal);
    },
    enabled: Boolean(projectId) && !surfaceNeedsAnotherField,
    retry: 1,
    staleTime: 30_000,
  });
  const saveChart = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.saveChart(projectId, chartSpec);
    },
    onSuccess: onContinue,
  });

  if (!preview || !quality) {
    return (
      <ApiStatePanel
        description={t("loadingBody")}
        kind="loading"
        title={t("loadingTitle")}
      />
    );
  }

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

  const applyRecommendation = (
    recommendation: ChartAnalysis["recommendations"][number],
  ) => {
    const xColumn = numericColumns.find(
      (column) => column.field === recommendation.xField,
    );
    const yColumn = numericColumns.find(
      (column) => column.field === recommendation.yField,
    );
    const zColumn = numericColumns.find(
      (column) => column.field === recommendation.zField,
    );
    if (!xColumn || !yColumn || !zColumn) return;

    const recommendedSeries = [yColumn, zColumn].map((column, index) => {
      const existing = chartSpec.series.find(
        (series) => series.field === column.field,
      );
      return {
        ...(existing ?? chartSpec.series[0]),
        color: existing?.color ?? seriesColors[index % seriesColors.length],
        field: column.field,
        label: column.label,
        lineStyle: existing?.lineStyle ?? "solid",
        panel: 1,
        yAxis: "primary" as const,
      };
    });
    if (recommendation.chartType === "surface3d") {
      updateChart({
        fitting: { ...chartSpec.fitting, model: "none" },
        groupField: null,
        panelCount: 1,
        secondaryYAxis: {
          ...chartSpec.secondaryYAxis,
          enabled: false,
          field: null,
        },
        series: recommendedSeries,
        type: "surface3d",
        uncertainty: {
          ...chartSpec.uncertainty,
          errorField: null,
          mode: "none",
        },
        xAxis: {
          field: xColumn.field,
          title: xColumn.label,
          unit: xColumn.unit ?? "",
        },
        yAxis: {
          field: yColumn.field,
          title: yColumn.label,
          unit: yColumn.unit ?? "",
        },
      });
      return;
    }
    if (recommendation.chartType === "scatter") {
      updateChart({
        groupField: yColumn.field,
        panelCount: 1,
        series: [recommendedSeries[1]],
        type: "scatter",
        xAxis: {
          field: xColumn.field,
          title: xColumn.label,
          unit: xColumn.unit ?? "",
        },
        yAxis: {
          field: zColumn.field,
          title: zColumn.label,
          unit: zColumn.unit ?? "",
        },
      });
    }
  };

  const warnings = analysisQuery.data?.series.flatMap((series) => series.warnings) ?? [];
  const translateAnalysisText = (value: string) => {
    if (value.startsWith("The selected model form")) return t("evidenceMessages.modelForm");
    if (value.startsWith("Complete finite observations")) {
      return t("evidenceMessages.completeObservations");
    }
    if (value.startsWith("The supplied error column")) return t("evidenceMessages.errorColumn");
    if (value.startsWith("Bootstrap residual resampling")) return t("evidenceMessages.bootstrap");
    if (value.startsWith("The selected confidence level")) return t("evidenceMessages.confidence");
    if (value.startsWith("Residual summaries")) return t("evidenceMessages.residualDescriptive");
    if (value.startsWith("They do not establish")) return t("evidenceMessages.residualNoInference");
    if (value.startsWith("R² describes")) return t("evidenceMessages.rSquared");
    if (value.startsWith("The interval is pointwise")) return t("evidenceMessages.pointwise");
    if (value.startsWith("Ordinary least squares")) return t("evidenceMessages.ordinary");
    if (value.startsWith("Weighted fitting")) return t("evidenceMessages.weighted");
    if (value.startsWith("A possible residual trend")) return t("evidenceMessages.residualTrend");
    if (value.startsWith("Prediction intervals")) return t("evidenceMessages.prediction");
    if (value.startsWith("Simultaneous confidence bands")) {
      return t("evidenceMessages.simultaneous");
    }
    if (value.startsWith("Robust regression")) return t("evidenceMessages.robust");
    if (value.startsWith("No multiplicity correction")) return t("evidenceMessages.multiplicity");
    if (value.startsWith("Select weighted least squares")) {
      return t("evidenceMessages.weightedSelection");
    }
    if (value.startsWith("A confidence band is available")) {
      return t("evidenceMessages.confidenceRequiresFit");
    }
    if (value.startsWith("Residual diagnostics require")) {
      return t("evidenceMessages.residualRequiresFit");
    }
    if (value.startsWith("The selected fit could not")) return t("evidenceMessages.fitUnavailable");
    return value;
  };
  const renderAnalysisEvidence = (series: ChartAnalysis["series"][number]) => {
    const disclosures = series.disclosures ?? [];
    const assumptions = disclosures.flatMap((item) => item.assumptions ?? []).map(translateAnalysisText);
    const limitations = disclosures.flatMap((item) => item.limitations ?? []).map(translateAnalysisText);
    const residual = series.residualDiagnostic;
    const uniqueAssumptions = [...new Set(assumptions)];
    const uniqueLimitations = [...new Set(limitations)];
    const hasDeferredMethod = disclosures.some((item) => item.status === "deferred");
    return (
      <Stack spacing={0.5} sx={{ mt: 0.75 }}>
        {series.fit ? (
          <Typography variant="caption">
            {t("analysisEvidence", {
              excludedCount: series.fit.excludedCount ?? 0,
              sampleSize: series.fit.sampleSize ?? residual?.sampleSize ?? 0,
            })}
          </Typography>
        ) : null}
        {residual?.status === "supported" && residual.rmse !== null ? (
          <Typography variant="caption">
            {t("residualDiagnostic", {
              meanResidual: residual.meanResidual?.toFixed(4) ?? "—",
              rmse: residual.rmse.toFixed(4),
            })}
          </Typography>
        ) : null}
        {residual?.residualTrend === "possible-trend" ? (
          <Typography variant="caption">{t("residualTrend")}</Typography>
        ) : null}
        {hasDeferredMethod ? (
          <Typography variant="caption">{t("deferredMethods")}</Typography>
        ) : null}
        {uniqueAssumptions.length > 0 ? (
          <Typography variant="caption">
            <strong>{t("analysisAssumptions")}:</strong> {uniqueAssumptions.join(" ")}
          </Typography>
        ) : null}
        {uniqueLimitations.length > 0 ? (
          <Typography variant="caption">
            <strong>{t("limitations")}:</strong> {uniqueLimitations.join(" ")}
          </Typography>
        ) : null}
      </Stack>
    );
  };
  const recommendationMessage = (
    recommendation: ChartAnalysis["recommendations"][number],
  ) => {
    const params = {
      pointCount: Number(recommendation.reasonParams.pointCount ?? 0),
      xCount: Number(recommendation.reasonParams.xCount ?? 0),
      xField: recommendation.xField,
      yCount: Number(recommendation.reasonParams.yCount ?? 0),
      yField: recommendation.yField,
      zField: recommendation.zField,
    };
    switch (recommendation.reasonCode) {
      case "chart.recommendation.surface-grid-line":
        return t("recommendations.surfaceGridLine", params);
      case "chart.recommendation.surface-grid":
        return t("recommendations.surfaceGrid", params);
      case "chart.recommendation.scatter-grid":
        return t("recommendations.scatterGrid", params);
      default:
        return recommendation.reason;
    }
  };
  const surfaceDiagnostic =
    chartSpec.type === "surface3d"
      ? analysisQuery.data?.preview.surfaceDiagnostics.find((item) => item.panel === 1)
      : undefined;
  const surfaceDiagnosticMessage = surfaceDiagnostic
    ? (() => {
        switch (surfaceDiagnostic.status) {
          case "invalid-values":
            return t("surfaceDiagnostics.invalidValues", {
              count: surfaceDiagnostic.nonFinitePoints,
            });
          case "insufficient-points":
            return t("surfaceDiagnostics.insufficientPoints");
          case "collinear":
            return t("surfaceDiagnostics.collinear");
          case "duplicate-coordinates":
            return t("surfaceDiagnostics.duplicateCoordinates", {
              count: surfaceDiagnostic.duplicateCoordinateRows,
            });
          case "missing-grid":
            return t("surfaceDiagnostics.missingGrid", {
              count: surfaceDiagnostic.missingGridCells,
            });
          case "irregular-grid":
            return t("surfaceDiagnostics.irregularGrid");
          default:
            return null;
        }
      })()
    : null;
  const surfaceBlocked = Boolean(
    surfaceDiagnostic && surfaceDiagnostic.status !== "valid",
  );

  const canvas = (
    <Stack spacing={1.5}>
      <Paper sx={{ border: 1, borderColor: "divider", p: { xs: 1, md: 2 } }}>
        {surfaceNeedsAnotherField ? (
          <ApiStatePanel
            compact
            description={t("surfaceSetupDescription")}
            kind="empty"
            title={t("surfaceSetupTitle")}
          />
        ) : analysisQuery.isPending ? (
          <ApiStatePanel
            compact
            description={t("calculatingAnalysis")}
            kind="loading"
            title={t("renderingPreview")}
          />
        ) : analysisQuery.error ? (
          <ApiStatePanel
            actionLabel={t("retryAnalysis")}
            compact
            description={
              analysisQuery.error instanceof LabVizApiError
                ? analysisQuery.error.message
                : t("analysisFailed")
            }
            kind="error"
            onAction={() => void analysisQuery.refetch()}
            title={t("previewUnavailable")}
          />
        ) : (
          <ChartCanvas
            analysis={analysisQuery.data}
            excludedFindingIds={excludedFindingIds}
            findings={quality.findings}
            preview={preview}
            spec={chartSpec}
          />
        )}
      </Paper>
      {preview.sampled ? (
        <Alert severity="info">
          {t("sampled", {
            count: preview.rows.length,
            total: preview.totalRows,
          })}
        </Alert>
      ) : null}
      {analysisQuery.data?.recommendations.map((recommendation) => (
        <Paper
          aria-label={t("recommendationTitle", {
            chart: t(`types.${recommendation.chartType}`),
          })}
          key={`${recommendation.chartType}:${recommendation.xField}:${recommendation.yField}:${recommendation.zField}`}
          sx={{ border: 1, borderColor: "info.main", p: 2 }}
        >
          <Stack spacing={1.25}>
            <Typography component="h2" sx={{ fontWeight: 700 }} variant="body1">
              {t("recommendationTitle", {
                chart: t(`types.${recommendation.chartType}`),
              })}
            </Typography>
            <Typography color="text.secondary" variant="body2">
              {recommendationMessage(recommendation)}
            </Typography>
            {recommendation.chartType !== "heatmap" ? (
              <Button
                onClick={() => applyRecommendation(recommendation)}
                size="small"
                sx={{ alignSelf: "flex-start" }}
                variant="outlined"
              >
                {t("useRecommendation", {
                  chart: t(`types.${recommendation.chartType}`),
                })}
              </Button>
            ) : null}
          </Stack>
        </Paper>
      ))}
      {analysisQuery.data?.series
        .filter(
          (series) =>
            series.fit ||
            series.disclosures?.some((item) => item.status === "insufficient-data"),
        )
        .map((series) => (
          <Alert key={series.field} severity={series.fit ? "success" : "warning"}>
            <Typography sx={{ fontWeight: 700 }} variant="body2">
              {series.fit ? t("fitResult", { label: series.label }) : t("analysisFailed")}
            </Typography>
            {chartSpec.fitting.showEquation ? (
              <Typography component="span" variant="body2">
                {series.fit?.equation}
              </Typography>
            ) : null}
            {chartSpec.fitting.showRSquared ? (
              <Typography component="span" sx={{ ml: 1 }} variant="body2">
                R² = {series.fit?.rSquared.toFixed(4)}
              </Typography>
            ) : null}
            {renderAnalysisEvidence(series)}
          </Alert>
        ))}
      {warnings.map((warning) => (
        <Alert key={warning} severity="warning">
          {warning}
        </Alert>
      ))}
    </Stack>
  );

  const inspector = (
    <Stack spacing={1.5}>
      <Alert severity="info" sx={{ display: { md: "none" } }}>
        {t("continueOnDesktop")}
      </Alert>
      <Paper sx={{ border: 1, borderColor: "divider", p: 2.5 }}>
        <Stack spacing={2.25}>
          <Typography component="h2" variant="h3">
            {t("settings")}
          </Typography>
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
                <MenuItem key={type} value={type}>
                  {t(`types.${type}`)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
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
          {chartSpec.type === "surface3d" ? (
            <Stack spacing={1.5}>
              <Typography color="text.secondary" variant="caption">
                {t("surfaceAxisHelp")}
              </Typography>
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
                  .filter((column) => column.field !== chartSpec.xAxis.field)
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
          <Typography color="text.secondary" variant="caption">
            {t("groupHelp")}
          </Typography>
          {surfaceNeedsAnotherField ? (
            <Alert severity="warning">{t("surfaceSetupDescription")}</Alert>
          ) : null}
          {surfaceDiagnosticMessage ? (
            <Alert severity="warning">{surfaceDiagnosticMessage}</Alert>
          ) : null}
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

          <Accordion disableGutters elevation={0}>
            <AccordionSummary>{t("seriesAppearance")}</AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
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
              </Stack>
            </AccordionDetails>
          </Accordion>

          <Accordion disableGutters elevation={0}>
            <AccordionSummary>{t("analysis")}</AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
                <Typography color="text.secondary" variant="caption">
                  {t("analysisHelp")}
                </Typography>
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
                        },
                      })
                    }
                    value={chartSpec.fitting.fitMethod}
                  >
                    {(["ordinary-least-squares", "weighted-least-squares"] as const).map(
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
                <FormControlLabel
                  control={
                    <Switch
                      checked={chartSpec.fitting.confidenceBand}
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
                  <FormControl fullWidth size="small">
                    <InputLabel id="confidence-method-label">{t("confidenceMethod")}</InputLabel>
                    <Select
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
                      {(["student-t", "bootstrap"] as const).map((method) => (
                        <MenuItem key={method} value={method}>
                          {t(`confidenceMethods.${method}`)}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
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
            </AccordionDetails>
          </Accordion>

          <Accordion disableGutters elevation={0}>
            <AccordionSummary>{t("layout")}</AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
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
            </AccordionDetails>
          </Accordion>

          <Stack direction="row" spacing={1}>
            <Button onClick={onBack} startIcon={<ArrowBackRoundedIcon />} variant="outlined">
              {t("back")}
            </Button>
            <Button
              disabled={
                surfaceNeedsAnotherField ||
                surfaceBlocked ||
                Boolean(analysisQuery.error)
              }
              endIcon={<ArrowForwardRoundedIcon />}
              fullWidth
              loading={saveChart.isPending}
              onClick={() => saveChart.mutate()}
              variant="contained"
            >
              {t("customize")}
            </Button>
          </Stack>
          {saveChart.error ? (
            <Alert severity="error">
              {saveChart.error instanceof LabVizApiError
                ? saveChart.error.message
                : t("saveFailed")}
            </Alert>
          ) : null}
        </Stack>
      </Paper>
    </Stack>
  );

  return (
    <StepLayout
      canvas={canvas}
      description={t("description")}
      inspector={inspector}
      title={t("title")}
    />
  );
}
