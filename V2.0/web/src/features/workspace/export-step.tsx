"use client";

import ArrowBackRoundedIcon from "@mui/icons-material/ArrowBackRounded";
import DownloadRoundedIcon from "@mui/icons-material/DownloadRounded";
import {
  Alert,
  Button,
  Divider,
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
import { useState } from "react";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { ChartCanvas } from "@/components/common/chart-canvas";
import type { ChartSpec } from "@/domain/chart-spec";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

import { StepLayout } from "./step-layout";
import { CloudProjectActions } from "./cloud-project-actions";
import { useWorkspaceStore } from "./workspace-store";

export function ExportStep({ onBack }: { onBack: () => void }) {
  const t = useTranslations("export");
  const chartSpec = useWorkspaceStore((state) => state.chartSpec);
  const projectId = useWorkspaceStore((state) => state.projectId);
  const preview = useWorkspaceStore((state) => state.preview);
  const quality = useWorkspaceStore((state) => state.quality);
  const issueActions = useWorkspaceStore((state) => state.issueActions);
  const updateExport = useWorkspaceStore((state) => state.updateExport);
  const chartSpecKey = JSON.stringify(chartSpec);
  const [preparedSpecKey, setPreparedSpecKey] = useState<string | null>(null);
  const excludedFindingIds = Object.entries(issueActions)
    .filter(([, action]) => ["exclude", "remove"].includes(action))
    .map(([findingId]) => findingId);
  const analysisQuery = useQuery({
    queryKey: [
      "chart-analysis",
      projectId,
      JSON.stringify({ chartSpec, issueActions }),
    ],
    queryFn: ({ signal }) => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.analyzeChart(projectId, chartSpec, signal);
    },
    enabled: Boolean(projectId),
    retry: 1,
    staleTime: 30_000,
  });
  const exportMutation = useMutation({
    mutationFn: ({ chart }: { chart: ChartSpec; key: string }) => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.requestExport(projectId, chart);
    },
    onSuccess: (_result, variables) => setPreparedSpecKey(variables.key),
  });
  const preparedExport =
    preparedSpecKey === chartSpecKey ? exportMutation.data : undefined;
  const currentExportError =
    exportMutation.variables?.key === chartSpecKey
      ? exportMutation.error
      : null;

  if (!preview || !quality) {
    return (
      <ApiStatePanel
        description={t("loadingDescription")}
        kind="loading"
        title={t("loadingTitle")}
      />
    );
  }

  const canvas = (
    <Stack spacing={1.5}>
      <Paper sx={{ border: 1, borderColor: "divider", p: { xs: 1, md: 2 } }}>
        {analysisQuery.isPending ? (
          <ApiStatePanel
            compact
            description={t("analysisLoading")}
            kind="loading"
            title={t("previewLoading")}
          />
        ) : analysisQuery.error ? (
          <ApiStatePanel
            actionLabel={t("retryPreview")}
            compact
            description={
              analysisQuery.error instanceof LabVizApiError
                ? analysisQuery.error.message
                : t("previewFailed")
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
      {preparedExport ? (
        <Alert severity={preparedExport.status === "failed" ? "error" : "success"}>
          {preparedExport.message}
        </Alert>
      ) : null}
      {preparedExport?.status === "ready" && preparedExport.downloadUrl ? (
        <Button
          component="a"
          download
          href={preparedExport.downloadUrl}
          startIcon={<DownloadRoundedIcon />}
          variant="contained"
        >
          {t("download", { format: chartSpec.export.format.toUpperCase() })}
        </Button>
      ) : null}
      {currentExportError ? (
        <Alert severity="error">
          {currentExportError instanceof LabVizApiError
            ? currentExportError.message
            : t("requestFailed")}
        </Alert>
      ) : null}
    </Stack>
  );

  const setFormat = (event: SelectChangeEvent) => {
    updateExport({ format: event.target.value as ChartSpec["export"]["format"] });
  };

  const setDpi = (event: SelectChangeEvent) => {
    updateExport({ dpi: Number(event.target.value) as 300 | 600 });
  };

  const setSize = (event: SelectChangeEvent) => {
    updateExport({
      sizePreset: event.target.value as ChartSpec["export"]["sizePreset"],
    });
  };

  const inspector = (
    <Paper sx={{ border: 1, borderColor: "divider", p: 2.5 }}>
      <Stack spacing={2.25}>
        <Typography component="h2" variant="h3">
          {t("settings")}
        </Typography>
        <FormControl fullWidth size="small">
          <InputLabel id="format-label">{t("format")}</InputLabel>
          <Select label={t("format")} labelId="format-label" onChange={setFormat} value={chartSpec.export.format}>
            <MenuItem value="png">PNG</MenuItem>
            <MenuItem value="svg">SVG</MenuItem>
            <MenuItem value="pdf">PDF</MenuItem>
          </Select>
        </FormControl>
        <FormControl fullWidth size="small">
          <InputLabel id="dpi-label">{t("dpi")}</InputLabel>
          <Select label={t("dpi")} labelId="dpi-label" onChange={setDpi} value={String(chartSpec.export.dpi)}>
            <MenuItem value="300">300 DPI</MenuItem>
            <MenuItem value="600">600 DPI</MenuItem>
          </Select>
        </FormControl>
        <FormControl fullWidth size="small">
          <InputLabel id="size-label">{t("size")}</InputLabel>
          <Select label={t("size")} labelId="size-label" onChange={setSize} value={chartSpec.export.sizePreset}>
            <MenuItem value="single-column">{t("singleColumn")}</MenuItem>
            <MenuItem value="double-column">{t("doubleColumn")}</MenuItem>
            <MenuItem value="a4">{t("a4")}</MenuItem>
            <MenuItem value="custom">{t("custom")}</MenuItem>
          </Select>
        </FormControl>
        {chartSpec.export.sizePreset === "custom" ? (
          <Stack direction="row" spacing={1}>
            <TextField
              fullWidth
              label={t("width")}
              onChange={(event) => {
                const width = Number(event.target.value);
                if (width > 0) updateExport({ width });
              }}
              size="small"
              slotProps={{ htmlInput: { min: 0.1, step: 0.1 } }}
              type="number"
              value={chartSpec.export.width ?? ""}
            />
            <TextField
              fullWidth
              label={t("height")}
              onChange={(event) => {
                const height = Number(event.target.value);
                if (height > 0) updateExport({ height });
              }}
              size="small"
              slotProps={{ htmlInput: { min: 0.1, step: 0.1 } }}
              type="number"
              value={chartSpec.export.height ?? ""}
            />
            <FormControl sx={{ minWidth: 88 }} size="small">
              <InputLabel id="dimension-unit-label">{t("unit")}</InputLabel>
              <Select
                label={t("unit")}
                labelId="dimension-unit-label"
                onChange={(event) =>
                  updateExport({ unit: event.target.value as "mm" | "cm" | "in" })
                }
                value={chartSpec.export.unit}
              >
                <MenuItem value="mm">mm</MenuItem>
                <MenuItem value="cm">cm</MenuItem>
                <MenuItem value="in">in</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        ) : null}
        <FormControl fullWidth size="small">
          <InputLabel id="font-family-label">{t("font")}</InputLabel>
          <Select
            label={t("font")}
            labelId="font-family-label"
            onChange={(event) =>
              updateExport({ fontFamily: event.target.value as "Arial" | "Times New Roman" })
            }
            value={chartSpec.export.fontFamily}
          >
            <MenuItem value="Arial">Arial</MenuItem>
            <MenuItem value="Times New Roman">Times New Roman</MenuItem>
          </Select>
        </FormControl>
        <Stack direction="row" spacing={1}>
          <TextField
            fullWidth
            label={t("fontSize")}
            onChange={(event) => updateExport({ fontSize: Number(event.target.value) })}
            size="small"
            slotProps={{ htmlInput: { min: 6, max: 36, step: 1 } }}
            type="number"
            value={chartSpec.export.fontSize}
          />
          <TextField
            fullWidth
            label={t("lineWidth")}
            onChange={(event) => updateExport({ lineWidth: Number(event.target.value) })}
            size="small"
            slotProps={{ htmlInput: { min: 0.25, max: 10, step: 0.25 } }}
            type="number"
            value={chartSpec.export.lineWidth}
          />
        </Stack>
        <FormControl fullWidth size="small">
          <InputLabel id="legend-position-label">{t("legend")}</InputLabel>
          <Select
            label={t("legend")}
            labelId="legend-position-label"
            onChange={(event) =>
              updateExport({
                legendPosition: event.target.value as ChartSpec["export"]["legendPosition"],
              })
            }
            value={chartSpec.export.legendPosition}
          >
            {(["auto", "top", "bottom", "left", "right", "none"] as const).map(
              (position) => (
                <MenuItem key={position} value={position}>
                  {t(`legendPositions.${position}`)}
                </MenuItem>
              ),
            )}
          </Select>
        </FormControl>
        <FormControlLabel
          control={
            <Switch
              checked={chartSpec.export.grayscalePreview}
              onChange={(event) => updateExport({ grayscalePreview: event.target.checked })}
            />
          }
          label={t("grayscale")}
        />
        <FormControlLabel
          control={
            <Switch
              checked={chartSpec.export.gridVisible}
              onChange={(event) => updateExport({ gridVisible: event.target.checked })}
            />
          }
          label={t("grid")}
        />
        <TextField
          fullWidth
          label={t("backgroundColor")}
          disabled={chartSpec.export.transparentBackground}
          onChange={(event) => updateExport({ backgroundColor: event.target.value })}
          size="small"
          type="color"
          value={chartSpec.export.backgroundColor}
        />
        <FormControlLabel
          control={
            <Switch
              checked={chartSpec.export.transparentBackground}
              onChange={(event) =>
                updateExport({ transparentBackground: event.target.checked })
              }
            />
          }
          label={t("transparent")}
        />
        <Stack direction="row" spacing={1}>
          <Button onClick={onBack} startIcon={<ArrowBackRoundedIcon />} variant="outlined">
            {t("back")}
          </Button>
          <Button
            endIcon={<DownloadRoundedIcon />}
            fullWidth
            loading={exportMutation.isPending}
            onClick={() =>
              exportMutation.mutate({ chart: chartSpec, key: chartSpecKey })
            }
            variant="contained"
          >
            {t("prepare")}
          </Button>
        </Stack>
        <Button
          component="a"
          download
          href={projectId ? labvizApi.cleanedDataUrl(projectId) : undefined}
          startIcon={<DownloadRoundedIcon />}
          variant="outlined"
        >
          {t("downloadCleanedData")}
        </Button>
        <Typography color="text.secondary" variant="caption">
          {t("completeDataNote")}
        </Typography>
        <Divider />
        <CloudProjectActions />
      </Stack>
    </Paper>
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
