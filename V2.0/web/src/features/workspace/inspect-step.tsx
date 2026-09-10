"use client";

import ArrowBackRoundedIcon from "@mui/icons-material/ArrowBackRounded";
import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
import ChevronLeftRoundedIcon from "@mui/icons-material/ChevronLeftRounded";
import ChevronRightRoundedIcon from "@mui/icons-material/ChevronRightRounded";
import ErrorOutlineRoundedIcon from "@mui/icons-material/ErrorOutlineRounded";
import RedoRoundedIcon from "@mui/icons-material/RedoRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import UndoRoundedIcon from "@mui/icons-material/UndoRounded";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Select,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { DataPreviewTable } from "@/components/common/data-preview-table";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

import { StepLayout } from "./step-layout";
import { useWorkspaceStore, type IssueAction } from "./workspace-store";

export function InspectStep({ onBack, onContinue }: { onBack: () => void; onContinue: () => void }) {
  const t = useTranslations("inspect");
  const projectId = useWorkspaceStore((state) => state.projectId);
  const preview = useWorkspaceStore((state) => state.preview);
  const quality = useWorkspaceStore((state) => state.quality);
  const issueActions = useWorkspaceStore((state) => state.issueActions);
  const issueActionPast = useWorkspaceStore((state) => state.issueActionPast);
  const issueActionFuture = useWorkspaceStore((state) => state.issueActionFuture);
  const setIssueAction = useWorkspaceStore((state) => state.setIssueAction);
  const setIssueActions = useWorkspaceStore((state) => state.setIssueActions);
  const undoIssueActions = useWorkspaceStore((state) => state.undoIssueActions);
  const redoIssueActions = useWorkspaceStore((state) => state.redoIssueActions);
  const restoreIssueActions = useWorkspaceStore((state) => state.restoreIssueActions);
  const replaceQuality = useWorkspaceStore((state) => state.replaceQuality);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [statusFilter, setStatusFilter] = useState<"all" | "pending" | "decided">(
    "all",
  );
  const [kindFilter, setKindFilter] = useState("all");
  const numericColumns = preview?.columns.filter((column) => column.kind === "number") ?? [];
  const [rangeField, setRangeField] = useState("");
  const [rangeMinimum, setRangeMinimum] = useState("");
  const [rangeMaximum, setRangeMaximum] = useState("");
  const issueKinds = Array.from(
    new Set(quality?.findings.map((finding) => finding.kind) ?? []),
  );
  const visibleFindings =
    quality?.findings.filter((finding) => {
      const decided = Boolean(issueActions[finding.id]);
      const matchesStatus =
        statusFilter === "all" ||
        (statusFilter === "decided" ? decided : !decided);
      const matchesKind = kindFilter === "all" || finding.kind === kindFilter;
      return matchesStatus && matchesKind;
    }) ?? [];
  const visibleIndex = Math.min(
    selectedIndex,
    Math.max(visibleFindings.length - 1, 0),
  );
  const selectedFinding = visibleFindings[visibleIndex];
  const selectedAction = selectedFinding
    ? (issueActions[selectedFinding.id] ?? "")
    : "";
  const decisionMutation = useMutation({
    mutationFn: (actions: Record<string, IssueAction>) => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.saveCleaningDecisions(
        projectId,
        Object.entries(actions).map(([findingId, action]) => ({
          findingId,
          action,
        })),
      );
    },
  });
  const rangeMutation = useMutation({
    mutationFn: () => {
      if (!projectId || !rangeField) throw new Error("Choose a numeric column.");
      return labvizApi.applyQualityRules(projectId, [
        {
          field: rangeField,
          minimum: rangeMinimum === "" ? null : Number(rangeMinimum),
          maximum: rangeMaximum === "" ? null : Number(rangeMaximum),
        },
      ]);
    },
    onSuccess: (nextQuality) => {
      replaceQuality(nextQuality);
      setSelectedIndex(0);
      setStatusFilter("all");
      setKindFilter("all");
    },
  });

  if (!preview || !quality) {
    return (
      <ApiStatePanel
        description={t("loadingDescription")}
        kind="loading"
        title={t("loadingTitle")}
      />
    );
  }

  const findingMessage = (
    finding: (typeof quality.findings)[number],
    field: "summary" | "reason",
  ): string => {
    const code = field === "summary" ? finding.summaryCode : finding.reasonCode;
    const params = field === "summary" ? finding.summaryParams : finding.reasonParams;
    const countParam = params?.count;
    const count =
      typeof countParam === "number"
        ? countParam
        : finding.affectedCount || finding.rowIds.length;
    const value = (name: string, fallback = ""): string | number => {
      const candidate = params?.[name];
      return typeof candidate === "string" || typeof candidate === "number"
        ? candidate
        : fallback;
    };

    switch (code) {
      case "quality.missing.summary":
        return t("findingMessages.missing.summary", { count });
      case "quality.missing.reason":
        return t("findingMessages.missing.reason");
      case "quality.type-conflict.summary":
        return t("findingMessages.typeConflict.summary", { count });
      case "quality.type-conflict.reason":
        return t("findingMessages.typeConflict.reason");
      case "quality.duplicate.summary":
        return t("findingMessages.duplicate.summary", { count });
      case "quality.duplicate.reason":
        return t("findingMessages.duplicate.reason");
      case "quality.outside-range.summary":
        return t("findingMessages.outsideRange.summary", { count });
      case "quality.outside-range.reason.both":
        return t("findingMessages.outsideRange.reasonBoth", {
          maximum: value("maximum"),
          minimum: value("minimum"),
        });
      case "quality.outside-range.reason.minimum":
        return t("findingMessages.outsideRange.reasonMinimum", {
          minimum: value("minimum"),
        });
      case "quality.outside-range.reason.maximum":
        return t("findingMessages.outsideRange.reasonMaximum", {
          maximum: value("maximum"),
        });
      case "quality.extreme-value.summary":
        return t("findingMessages.extremeValue.summary", { count });
      case "quality.extreme-value.reason":
        return t("findingMessages.extremeValue.reason");
      case "quality.sudden-change.summary":
        return t("findingMessages.suddenChange.summary", { count });
      case "quality.sudden-change.summary.grid":
        return t("findingMessages.suddenChange.gridSummary", { count });
      case "quality.sudden-change.reason":
        return t("findingMessages.suddenChange.reason");
      case "quality.sudden-change.reason.grid":
        return t("findingMessages.suddenChange.gridReason");
      case "quality.trend-inconsistent.summary":
        return t("findingMessages.trendInconsistent.summary", { count });
      case "quality.trend-inconsistent.reason":
        return t("findingMessages.trendInconsistent.reason", {
          rSquared: value("rSquared", "?"),
        });
      default:
        return finding[field];
    }
  };

  const decidedCount = quality.findings.filter(
    (finding) => issueActions[finding.id],
  ).length;
  const pendingCount = quality.findings.length - decidedCount;
  const syncCurrentActions = () => {
    decisionMutation.mutate(useWorkspaceStore.getState().issueActions);
  };
  const moveFinding = (offset: number) => {
    setSelectedIndex((index) =>
      Math.min(Math.max(index + offset, 0), visibleFindings.length - 1),
    );
    decisionMutation.reset();
  };

  const canvas = (
    <Paper sx={{ border: 1, borderColor: "divider", overflow: "hidden", p: 2 }}>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={1}
        sx={{
          alignItems: { xs: "flex-start", sm: "center" },
          justifyContent: "space-between",
          mb: 2,
        }}
      >
        <Box>
          <Typography sx={{ fontWeight: 700 }}>{t("summaryTitle")}</Typography>
          <Typography color="text.secondary" variant="body2">
            {quality.findings.length === 0
              ? t("validRowsNoDecisions", {
                  count: quality.validRows.toLocaleString(),
                })
              : t("findingSummary", {
                  count: quality.findings.length,
                  pending: pendingCount,
                })}
          </Typography>
        </Box>
        <Alert severity="warning" sx={{ py: 0 }}>
          {t("sourceUnchanged")}
        </Alert>
      </Stack>
      <DataPreviewTable
        emphasizeIssues
        findings={quality.findings}
        preview={preview}
      />
    </Paper>
  );

  const inspector = (
    <Stack spacing={2}>
      {numericColumns.length > 0 ? (
        <Accordion disableGutters elevation={0} sx={{ border: 1, borderColor: "divider" }}>
          <AccordionSummary>{t("rangeTitle")}</AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1.5}>
              <Typography color="text.secondary" variant="body2">
                {t("rangeDescription")}
              </Typography>
              <FormControl fullWidth size="small">
                <InputLabel id="valid-range-field-label">{t("numericColumn")}</InputLabel>
                <Select
                  label={t("numericColumn")}
                  labelId="valid-range-field-label"
                  onChange={(event) => setRangeField(event.target.value)}
                  value={rangeField}
                >
                  {numericColumns.map((column) => (
                    <MenuItem key={column.field} value={column.field}>
                      {column.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Stack direction="row" spacing={1}>
                <TextField
                  fullWidth
                  label={t("minimum")}
                  onChange={(event) => setRangeMinimum(event.target.value)}
                  size="small"
                  type="number"
                  value={rangeMinimum}
                />
                <TextField
                  fullWidth
                  label={t("maximum")}
                  onChange={(event) => setRangeMaximum(event.target.value)}
                  size="small"
                  type="number"
                  value={rangeMaximum}
                />
              </Stack>
              <Button
                disabled={
                  !rangeField ||
                  (rangeMinimum === "" && rangeMaximum === "") ||
                  rangeMutation.isPending
                }
                onClick={() => rangeMutation.mutate()}
                variant="outlined"
              >
                {rangeMutation.isPending ? t("rechecking") : t("applyRange")}
              </Button>
              {rangeMutation.error ? (
                <Alert severity="error">
                  {rangeMutation.error instanceof LabVizApiError
                    ? rangeMutation.error.message
                    : t("rangeFailed")}
                </Alert>
              ) : null}
            </Stack>
          </AccordionDetails>
        </Accordion>
      ) : null}
      {quality.findings.length > 0 ? (
        <Paper sx={{ border: 1, borderColor: "divider", p: 1.5 }}>
          <Stack spacing={1.25}>
            <Typography sx={{ fontWeight: 700 }} variant="body2">
              {t("filterTitle")}
            </Typography>
            <FormControl fullWidth size="small">
              <InputLabel id="issue-status-filter-label">{t("decisionStatus")}</InputLabel>
              <Select
                label={t("decisionStatus")}
                labelId="issue-status-filter-label"
                onChange={(event) => {
                  setStatusFilter(event.target.value as typeof statusFilter);
                  setSelectedIndex(0);
                }}
                value={statusFilter}
              >
                <MenuItem value="all">{t("allIssues")}</MenuItem>
                <MenuItem value="pending">{t("pendingStatus")}</MenuItem>
                <MenuItem value="decided">{t("decidedStatus")}</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth size="small">
              <InputLabel id="issue-kind-filter-label">{t("issueType")}</InputLabel>
              <Select
                label={t("issueType")}
                labelId="issue-kind-filter-label"
                onChange={(event) => {
                  setKindFilter(event.target.value);
                  setSelectedIndex(0);
                }}
                value={kindFilter}
              >
                <MenuItem value="all">{t("allTypes")}</MenuItem>
                {issueKinds.map((kind) => (
                  <MenuItem key={kind} value={kind}>
                    {t(`issueKinds.${kind}`)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {selectedFinding ? (
              <FormControl fullWidth size="small">
                <InputLabel id="issue-jump-label">{t("jump")}</InputLabel>
                <Select
                  label={t("jump")}
                  labelId="issue-jump-label"
                  onChange={(event) => {
                    const nextIndex = visibleFindings.findIndex(
                      (finding) => finding.id === event.target.value,
                    );
                    if (nextIndex >= 0) setSelectedIndex(nextIndex);
                  }}
                  value={selectedFinding.id}
                >
                  {visibleFindings.map((finding, index) => (
                    <MenuItem key={finding.id} value={finding.id}>
                      {index + 1}. {findingMessage(finding, "summary")}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            ) : null}
          </Stack>
        </Paper>
      ) : null}
      {selectedFinding ? (
        <>
          <Paper sx={{ border: 1, borderColor: "divider", p: 1.5 }}>
            <Stack spacing={1.25}>
              <Stack
                direction="row"
                spacing={1}
                sx={{ alignItems: "center", justifyContent: "space-between" }}
              >
                <Typography sx={{ fontWeight: 700 }} variant="body2">
                  {t("findingPosition", {
                    current: visibleIndex + 1,
                    count: visibleFindings.length,
                  })}
                </Typography>
                <Chip
                  color={pendingCount === 0 ? "success" : "warning"}
                  label={t("pending", { count: pendingCount })}
                  size="small"
                  variant="outlined"
                />
              </Stack>
              <Stack direction="row" spacing={1}>
                <Button
                  aria-label={t("previousFinding")}
                  disabled={visibleIndex === 0}
                  onClick={() => moveFinding(-1)}
                  size="small"
                  startIcon={<ChevronLeftRoundedIcon />}
                >
                  {t("previous")}
                </Button>
                <Button
                  aria-label={t("nextFinding")}
                  disabled={visibleIndex === visibleFindings.length - 1}
                  endIcon={<ChevronRightRoundedIcon />}
                  onClick={() => moveFinding(1)}
                  size="small"
                >
                  {t("next")}
                </Button>
              </Stack>
            </Stack>
          </Paper>
          <Paper
            sx={{
              border: 1,
              borderColor:
                selectedFinding.severity === "error"
                  ? "error.main"
                  : "warning.main",
              p: 2.5,
            }}
          >
            <Stack spacing={1.5}>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                {selectedFinding.severity === "error" ? (
                  <ErrorOutlineRoundedIcon color="error" />
                ) : (
                  <WarningAmberRoundedIcon color="warning" />
                )}
                <Typography component="h2" variant="h3">
                  {findingMessage(selectedFinding, "summary")}
                </Typography>
              </Stack>
              <Typography color="text.secondary" variant="body2">
                {findingMessage(selectedFinding, "reason")}
              </Typography>
              <Typography color="text.secondary" variant="caption">
                {t("affectedRows", {
                  count: selectedFinding.affectedCount || selectedFinding.rowIds.length,
                })}
                {selectedFinding.column
                  ? ` · ${selectedFinding.column}`
                  : ""}
              </Typography>
              {selectedFinding.rowIdsTruncated ? (
                <Alert severity="info" sx={{ py: 0.25 }}>
                  {t("representativeRows", {
                    count: selectedFinding.rowIds.length,
                  })}
                </Alert>
              ) : null}
              <Divider />
              <Typography color="text.secondary" variant="caption">
                {t("decisionHelp")}
              </Typography>
              <RadioGroup
                onChange={(event) => {
                  const action = event.target.value as IssueAction;
                  const nextActions = {
                    ...issueActions,
                    [selectedFinding.id]: action,
                  };
                  setIssueAction(selectedFinding.id, action);
                  decisionMutation.mutate(nextActions);
                }}
                value={selectedAction}
              >
                <FormControlLabel
                  control={<Radio size="small" />}
                  label={t("ignore")}
                  value="ignore"
                />
                <FormControlLabel
                  control={<Radio size="small" />}
                  label={t("exclude")}
                  value="exclude"
                />
                <FormControlLabel
                  control={<Radio size="small" />}
                  label={t("remove")}
                  value="remove"
                />
              </RadioGroup>
            </Stack>
          </Paper>
          {decisionMutation.error ? (
            <Alert severity="error">
              {decisionMutation.error instanceof LabVizApiError
                ? decisionMutation.error.message
                : t("decisionSaveFailed")}
            </Alert>
          ) : null}
          <Paper sx={{ border: 1, borderColor: "divider", p: 1.5 }}>
            <Stack spacing={1}>
              <Button
                disabled={pendingCount === 0 || decisionMutation.isPending}
                onClick={() => {
                  const nextActions = Object.fromEntries(
                    quality.findings.map((finding) => [
                      finding.id,
                      issueActions[finding.id] ?? "ignore",
                    ]),
                  ) as Record<string, IssueAction>;
                  setIssueActions(nextActions);
                  decisionMutation.mutate(nextActions);
                }}
                size="small"
                variant="outlined"
              >
                {t("keepRemaining")}
              </Button>
              <Stack direction="row" spacing={0.5}>
                <Button
                  disabled={issueActionPast.length === 0 || decisionMutation.isPending}
                  onClick={() => {
                    undoIssueActions();
                    syncCurrentActions();
                  }}
                  size="small"
                  startIcon={<UndoRoundedIcon />}
                >
                  {t("undo")}
                </Button>
                <Button
                  disabled={issueActionFuture.length === 0 || decisionMutation.isPending}
                  onClick={() => {
                    redoIssueActions();
                    syncCurrentActions();
                  }}
                  size="small"
                  startIcon={<RedoRoundedIcon />}
                >
                  {t("redo")}
                </Button>
              </Stack>
              <Button
                color="inherit"
                disabled={Object.keys(issueActions).length === 0 || decisionMutation.isPending}
                onClick={() => {
                  restoreIssueActions();
                  syncCurrentActions();
                }}
                size="small"
                startIcon={<RestartAltRoundedIcon />}
              >
                {t("restoreOriginal")}
              </Button>
            </Stack>
          </Paper>
        </>
      ) : (
        <ApiStatePanel
          compact
          actionLabel={quality.findings.length > 0 ? t("clearFilters") : undefined}
          description={
            quality.findings.length > 0
              ? t("noMatchDescription")
              : t("noIssuesDescription")
          }
          kind={quality.findings.length > 0 ? "empty" : "success"}
          onAction={
            quality.findings.length > 0
              ? () => {
                  setStatusFilter("all");
                  setKindFilter("all");
                  setSelectedIndex(0);
                }
              : undefined
          }
          title={quality.findings.length > 0 ? t("noMatchTitle") : t("noIssuesTitle")}
        />
      )}

      <Stack direction="row" spacing={1}>
        <Button onClick={onBack} startIcon={<ArrowBackRoundedIcon />} variant="outlined">
          {t("back")}
        </Button>
        <Button
          disabled={
            pendingCount > 0 ||
            decisionMutation.isPending ||
            Boolean(decisionMutation.error)
          }
          endIcon={<ArrowForwardRoundedIcon />}
          fullWidth
          onClick={onContinue}
          variant="contained"
        >
          {t("create")}
        </Button>
      </Stack>
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
