"use client";

import { Box } from "@mui/material";
import { useTranslations } from "next-intl";
import {
  DataGrid,
  type GridCellParams,
  type GridColDef,
} from "@mui/x-data-grid";

import type {
  DataPreview,
  PreviewRow,
  QualityFinding,
} from "@/domain/api-contract";

type DataPreviewTableProps = {
  preview: DataPreview;
  findings?: QualityFinding[];
  emphasizeIssues?: boolean;
};

export function DataPreviewTable({
  preview,
  findings = [],
  emphasizeIssues = false,
}: DataPreviewTableProps) {
  const t = useTranslations("common");
  const columns: GridColDef<PreviewRow>[] = [
    { field: "rowId", headerName: t("row"), width: 72 },
    ...preview.columns.map<GridColDef<PreviewRow>>((column) => ({
      field: column.field,
      headerName: column.unit
        ? `${column.label} (${column.unit})`
        : column.label,
      flex: 1,
      minWidth: column.kind === "number" ? 128 : 150,
      valueFormatter: (value) => value ?? t("missing"),
    })),
  ];

  const cellClassName = (params: GridCellParams<PreviewRow>) => {
    if (!emphasizeIssues) return "";
    const finding = findings.find(
      (item) =>
        item.column === params.field &&
        item.rowIds.some((rowId) => String(rowId) === String(params.id)),
    );
    if (finding?.kind === "missing") return "labviz-missing-cell";
    if (finding) return "labviz-suspicious-cell";
    return "";
  };

  return (
    <Box sx={{ height: 440, width: "100%" }}>
      <DataGrid
        columns={columns}
        disableColumnMenu
        disableRowSelectionOnClick
        getCellClassName={cellClassName}
        getRowId={(row) => row.rowId}
        initialState={{
          pagination: { paginationModel: { page: 0, pageSize: 25 } },
        }}
        pageSizeOptions={[25, 50]}
        rows={preview.rows}
        sx={{
          bgcolor: "background.paper",
          borderColor: "divider",
          "& .MuiDataGrid-columnHeaders": {
            bgcolor: "#F8FAFC",
          },
          "& .labviz-missing-cell": {
            bgcolor: "warning.light",
            color: "warning.dark",
            fontWeight: 700,
          },
          "& .labviz-suspicious-cell": {
            bgcolor: "error.light",
            color: "error.dark",
            fontWeight: 700,
          },
        }}
      />
    </Box>
  );
}
