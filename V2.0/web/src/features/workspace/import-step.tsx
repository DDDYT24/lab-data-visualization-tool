"use client";

import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
import DescriptionOutlinedIcon from "@mui/icons-material/DescriptionOutlined";
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { DataPreviewTable } from "@/components/common/data-preview-table";
import { ApiStatePanel } from "@/components/common/api-state-panel";
import { formatBytes } from "@/features/import-data/file-policy";

import { StepLayout } from "./step-layout";
import { useWorkspaceStore } from "./workspace-store";

export function ImportStep({
  onContinue,
  onReimport,
}: {
  onContinue: () => void;
  onReimport: (sheetName: string | null, headerRow: number) => void;
}) {
  const t = useTranslations("importData");
  const file = useWorkspaceStore((state) => state.selectedFile);
  const preview = useWorkspaceStore((state) => state.preview);
  const chartSpec = useWorkspaceStore((state) => state.chartSpec);
  const updateChart = useWorkspaceStore((state) => state.updateChart);
  const [sheetName, setSheetName] = useState(file?.sheetName ?? "");
  const [headerRow, setHeaderRow] = useState(file?.headerRow ?? 1);

  if (!preview) {
    return (
      <ApiStatePanel
        description={t("loadingDescription")}
        kind="loading"
        title={t("loadingTitle")}
      />
    );
  }

  const numericColumns = preview.columns.filter(
    (column) => column.kind === "number",
  );

  const canvas = (
    <Paper sx={{ border: 1, borderColor: "divider", overflow: "hidden", p: 2 }}>
      <Stack
        direction="row"
        sx={{ alignItems: "center", justifyContent: "space-between", mb: 2 }}
      >
        <Box>
          <Typography sx={{ fontWeight: 700 }}>{t("rows")}</Typography>
          <Typography color="text.secondary" variant="body2">
            {preview.sampled
              ? t("sampledRows", {
                  count: preview.rows.length.toLocaleString(),
                  total: preview.totalRows.toLocaleString(),
                })
              : t("completeRows", { count: preview.totalRows.toLocaleString() })}
          </Typography>
        </Box>
        <Chip
          color="success"
          label={t("numericColumns", { count: numericColumns.length })}
          size="small"
          variant="outlined"
        />
      </Stack>
      <DataPreviewTable preview={preview} />
    </Paper>
  );

  const inspector = (
    <Stack spacing={2}>
      <Paper sx={{ border: 1, borderColor: "divider", p: 2.5 }}>
        <Stack direction="row" spacing={1.5} sx={{ alignItems: "center" }}>
          <DescriptionOutlinedIcon color="primary" />
          <Box sx={{ minWidth: 0 }}>
            <Typography color="text.secondary" variant="caption">
              {t("file")}
            </Typography>
            <Typography noWrap sx={{ fontWeight: 700 }}>
              {file?.isSample ? t("sampleFile") : file?.name}
            </Typography>
            <Typography color="text.secondary" variant="body2">
              {file ? formatBytes(file.size) : "—"}
            </Typography>
          </Box>
        </Stack>
      </Paper>

      <Paper sx={{ border: 1, borderColor: "divider", p: 2.5 }}>
        <Stack spacing={2.25}>
          <Typography component="h2" variant="h3">
            {t("settings")}
          </Typography>
          <FormControl fullWidth size="small">
            <InputLabel id="sheet-label">{t("sheet")}</InputLabel>
            <Select
              label={t("sheet")}
              labelId="sheet-label"
              disabled={!file?.sourceFile || (file.availableSheets?.length ?? 0) < 2}
              onChange={(event) => setSheetName(event.target.value)}
              value={sheetName}
            >
              {(file?.availableSheets?.length
                ? file.availableSheets
                : [file?.sheetName ?? ""]
              ).map((name) => (
                <MenuItem key={name || "single-table"} value={name}>
                  {name || t("singleTable")}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label={t("header")}
            disabled={!file?.sourceFile}
            onChange={(event) => setHeaderRow(Math.max(1, Number(event.target.value)))}
            size="small"
            slotProps={{ htmlInput: { min: 1, max: 1000, step: 1 } }}
            type="number"
            value={headerRow}
          />
          {file?.sourceFile &&
          (sheetName !== (file.sheetName ?? "") || headerRow !== (file.headerRow ?? 1)) ? (
            <Button
              onClick={() => onReimport(sheetName || null, headerRow)}
              variant="outlined"
            >
              {t("applySettings")}
            </Button>
          ) : null}
          <FormControl fullWidth size="small">
            <InputLabel id="x-field-label">{t("xColumn")}</InputLabel>
            <Select
              label={t("xColumn")}
              labelId="x-field-label"
              onChange={(event) => {
                const column = numericColumns.find(
                  (item) => item.field === event.target.value,
                );
                if (column) {
                  updateChart({
                    xAxis: {
                      field: column.field,
                      title: column.label,
                      unit: column.unit ?? "",
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
          <TextField
            fullWidth
            label={t("xUnit")}
            onChange={(event) =>
              updateChart({
                xAxis: { ...chartSpec.xAxis, unit: event.target.value },
              })
            }
            size="small"
            value={chartSpec.xAxis.unit}
          />
          <FormControl fullWidth size="small">
            <InputLabel id="y-field-label">{t("yColumn")}</InputLabel>
            <Select
              label={t("yColumn")}
              labelId="y-field-label"
              onChange={(event) => {
                const column = numericColumns.find(
                  (item) => item.field === event.target.value,
                );
                if (column) {
                  updateChart({
                    yAxis: {
                      field: column.field,
                      title: column.label,
                      unit: column.unit ?? "",
                    },
                    series: [
                      {
                        ...chartSpec.series[0],
                        field: column.field,
                        label: column.label,
                      },
                    ],
                  });
                }
              }}
              value={chartSpec.series[0].field}
            >
              {numericColumns.map((column) => (
                <MenuItem key={column.field} value={column.field}>
                  {column.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label={t("yUnit")}
            onChange={(event) =>
              updateChart({
                yAxis: { ...chartSpec.yAxis, unit: event.target.value },
              })
            }
            size="small"
            value={chartSpec.yAxis.unit}
          />
          {numericColumns.length === 0 ? (
            <Alert severity="error">{t("noNumericColumns")}</Alert>
          ) : null}
          <Divider />
          <Button
            disabled={numericColumns.length === 0}
            endIcon={<ArrowForwardRoundedIcon />}
            fullWidth
            onClick={onContinue}
            variant="contained"
          >
            {t("review")}
          </Button>
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
