"use client";

import ArrowBackRoundedIcon from "@mui/icons-material/ArrowBackRounded";
import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
import ExpandMoreRoundedIcon from "@mui/icons-material/ExpandMoreRounded";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Button,
  Paper,
  Stack,
  Tab,
  Tabs,
  Typography,
} from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { ChartCanvas } from "@/components/common/chart-canvas";
import { DataPreviewTable } from "@/components/common/data-preview-table";
import type { ChartAnalysis } from "@/domain/api-contract";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

import { analysisCacheKey, planAppearanceKey } from "./analysis-cache-key";
import { ChartAppearancePanel } from "./chart-appearance-panel";
import { ChartDataPanel, seriesColors, surfaceSetup } from "./chart-data-panel";
import { ChartMethodPanel } from "./chart-method-panel";
import { StepLayout } from "./step-layout";
import { useWorkspaceStore } from "./workspace-store";

// Data and method edits change what is computed; appearance edits only redraw the plan.
const sections = ["data", "method", "appearance"] as const;
type Section = (typeof sections)[number];

export function ChartStep({
  active = true,
  onBack,
  onContinue,
}: {
  active?: boolean;
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
  const [section, setSection] = useState<Section>("data");
  const numericColumns = preview?.columns.filter((column) => column.kind === "number") ?? [];
  const surfaceNeedsAnotherField = surfaceSetup(chartSpec).incomplete;
  const analysisKey = analysisCacheKey(chartSpec, issueActions);
  const analysisQuery = useQuery({
    queryKey: ["chart-analysis", projectId, analysisKey, planAppearanceKey(chartSpec)],
    queryFn: ({ signal }) => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.analyzeChart(projectId, chartSpec, signal);
    },
    // Restyling keeps the chart on screen until its new plan arrives.
    placeholderData: (previous, query) => (query?.queryKey[2] === analysisKey ? previous : undefined),
    enabled: active && Boolean(projectId) && !surfaceNeedsAnotherField,
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
    if (value.startsWith("Ordinary least squares")) return t("evidenceMessages.ordinary");
    if (value.startsWith("Weighted fitting")) return t("evidenceMessages.weighted");
    if (value.startsWith("Huber IRLS")) return t("evidenceMessages.robustAssumption");
    if (value.startsWith("Huber downweights")) return t("evidenceMessages.robustLimit");
    if (value.startsWith("The interval includes residual")) {
      return t("evidenceMessages.predictionAssumption");
    }
    if (value.startsWith("The Working-Hotelling band covers the family")) {
      return t("evidenceMessages.simultaneousAssumption");
    }
    if (value.startsWith("The prediction interval includes")) {
      return t("evidenceMessages.predictionLimit");
    }
    if (value.startsWith("The Working-Hotelling band covers the displayed")) {
      return t("evidenceMessages.simultaneousLimit");
    }
    if (value.startsWith("The selected confidence level is interpreted for one future")) {
      return t("evidenceMessages.predictionConfidence");
    }
    if (value.startsWith("The selected confidence level is interpreted as a Working-Hotelling")) {
      return t("evidenceMessages.simultaneousConfidence");
    }
    if (value.startsWith("This pointwise mean interval")) return t("evidenceMessages.pointwise");
    if (value.startsWith("This prediction interval")) return t("evidenceMessages.predictionLimit");
    if (value.startsWith("This simultaneous mean band")) {
      return t("evidenceMessages.simultaneousLimit");
    }
    if (value.startsWith("A possible residual trend")) return t("evidenceMessages.residualTrend");
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
    const intervalKind = series.fit?.intervalKind;
    const intervalMeaning =
      intervalKind === "pointwise-mean"
        ? t("intervalKinds.pointwise-mean")
        : intervalKind === "prediction"
          ? t("intervalKinds.prediction")
          : intervalKind === "simultaneous"
            ? t("intervalKinds.simultaneous")
            : null;
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
        {intervalMeaning ? (
          <Typography variant="caption">
            {t("intervalEvidence", { meaning: intervalMeaning })}
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
            active={active}
            analysis={analysisQuery.data}
            preview={preview}
            spec={chartSpec}
          />
        )}
      </Paper>
      <Accordion disableGutters elevation={0} sx={{ border: 1, borderColor: "divider" }}>
        <AccordionSummary expandIcon={<ExpandMoreRoundedIcon />}>
          <Typography sx={{ fontWeight: 700 }}>{t("dataTableTitle")}</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography color="text.secondary" sx={{ display: "block", mb: 1 }} variant="caption">
            {t("dataTableLegend")}
          </Typography>
          <DataPreviewTable
            active={active}
            findings={quality.findings}
            emphasizeIssues
            preview={preview}
          />
        </AccordionDetails>
      </Accordion>
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
          <Tabs
            aria-label={t("sections.label")}
            onChange={(_, value: Section) => setSection(value)}
            value={section}
            variant="fullWidth"
          >
            {sections.map((value) => (
              <Tab
                aria-controls={`chart-section-${value}`}
                id={`chart-tab-${value}`}
                key={value}
                label={t(`sections.${value}`)}
                sx={{ minWidth: 0 }}
                value={value}
              />
            ))}
          </Tabs>
          <div aria-labelledby={`chart-tab-${section}`} id={`chart-section-${section}`} role="tabpanel">
            {section === "data" ? (
              <ChartDataPanel numericColumns={numericColumns} />
            ) : section === "method" ? (
              <ChartMethodPanel numericColumns={numericColumns} />
            ) : (
              <ChartAppearancePanel />
            )}
          </div>
          {/* Shown on every tab: it is why Continue is disabled. */}
          {surfaceDiagnosticMessage ? (
            <Alert severity="warning">{surfaceDiagnosticMessage}</Alert>
          ) : null}
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
      inspector={inspector}
      title={t("title")}
    />
  );
}
