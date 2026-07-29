"use client";

import ArrowBackRoundedIcon from "@mui/icons-material/ArrowBackRounded";
import {
  Box,
  Button,
  Chip,
  Container,
  LinearProgress,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { useTranslations } from "next-intl";
import Link from "next/link";
import { useEffect, useRef } from "react";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { WorkflowStepper } from "@/components/common/workflow-stepper";

import { ChartStep } from "./chart-step";
import { ExportStep } from "./export-step";
import { ImportStep } from "./import-step";
import { InspectStep } from "./inspect-step";
import { useWorkspaceProject } from "./use-workspace-project";
import { getWorkspaceLoadingCopy } from "./workspace-loading-state";
import { type WorkflowStep, useWorkspaceStore } from "./workspace-store";

export function WorkspaceScreen({
  initialProjectId,
  initialStep,
}: {
  initialProjectId?: string;
  initialStep?: WorkflowStep;
}) {
  const t = useTranslations("workspace");
  const file = useWorkspaceStore((state) => state.selectedFile);
  const currentStep = useWorkspaceStore((state) => state.currentStep);
  const storageMode = useWorkspaceStore((state) => state.storageMode);
  const goToStep = useWorkspaceStore((state) => state.goToStep);
  const { job, loadError, loadStatus, preview, quality, reimport, retry, uploading } =
    useWorkspaceProject(initialProjectId);
  const loadingCopy = getWorkspaceLoadingCopy(uploading, job);
  const initialStepApplied = useRef(false);
  const loadingText = uploading && job?.stage !== "ready"
    ? {
        title: t("loadingUploadTitle"),
        description: t("loadingUploadDescription"),
      }
    : job?.stage === "queued"
      ? {
          title: t("loadingQueueTitle"),
          description: t("loadingQueueDescription"),
        }
      : job?.stage === "parsing"
        ? {
            title: t("loadingParseTitle"),
            description: t("loadingParseDescription"),
          }
        : job?.stage === "profiling"
          ? {
              title: t("loadingProfileTitle"),
              description: t("loadingProfileDescription"),
            }
          : {
              title: t("loadingDefaultTitle"),
              description: t("loadingDefaultDescription"),
            };

  useEffect(() => {
    if (
      initialStep &&
      !initialStepApplied.current &&
      loadStatus === "ready" &&
      preview &&
      quality
    ) {
      initialStepApplied.current = true;
      goToStep(initialStep);
    }
  }, [goToStep, initialStep, loadStatus, preview, quality]);

  if (!file && !initialProjectId) {
    return (
      <Container maxWidth="sm" sx={{ py: 12 }}>
        <Paper sx={{ border: 1, borderColor: "divider", p: 5, textAlign: "center" }}>
          <Stack spacing={2} sx={{ alignItems: "center" }}>
            <Typography component="h1" variant="h2">
              {t("startFileTitle")}
            </Typography>
            <Typography color="text.secondary">
              {t("startFileDescription")}
            </Typography>
            <Button component={Link} href="/" startIcon={<ArrowBackRoundedIcon />} variant="contained">
              {t("returnToUpload")}
            </Button>
          </Stack>
        </Paper>
      </Container>
    );
  }

  const workspaceContent =
    loadStatus === "error" ? (
      <ApiStatePanel
        actionLabel={t("retry")}
        description={
          loadError ??
          t("loadFailed")
        }
        kind="error"
        onAction={retry}
        secondaryAction={
          <Button component={Link} href="/" variant="text">
            {t("chooseAnotherFile")}
          </Button>
        }
        title={t("unavailableTitle")}
      />
    ) : !preview || !quality ? (
      <Stack spacing={2}>
        <ApiStatePanel
          description={loadingText.description}
          kind="loading"
          title={loadingText.title}
        />
        <LinearProgress
          aria-label={t("processingProgress")}
          value={loadingCopy.progress}
          variant="determinate"
        />
      </Stack>
    ) : currentStep === "import" ? (
      <ImportStep
        onContinue={() => goToStep("inspect")}
        onReimport={reimport}
      />
    ) : currentStep === "inspect" ? (
      <InspectStep
        onBack={() => goToStep("import")}
        onContinue={() => goToStep("chart")}
      />
    ) : currentStep === "chart" ? (
      <ChartStep
        onBack={() => goToStep("inspect")}
        onContinue={() => goToStep("export")}
      />
    ) : (
      <ExportStep onBack={() => goToStep("chart")} />
    );

  return (
    <Container disableGutters maxWidth={false}>
      <Paper
        square
        sx={{
          border: 0,
          minHeight: "calc(100vh - 64px)",
          overflow: "hidden",
        }}
      >
        <Stack
          direction={{ xs: "column", sm: "row" }}
          spacing={1.5}
          sx={{
            alignItems: { xs: "flex-start", sm: "center" },
            borderBottom: 1,
            borderColor: "divider",
            justifyContent: "space-between",
            px: { xs: 2, md: 3 },
            py: 2,
          }}
        >
          <Box sx={{ minWidth: 0 }}>
            <Typography color="text.secondary" variant="caption">
              {t("project")}
            </Typography>
            <Typography noWrap sx={{ fontWeight: 720 }}>
              {file?.name ?? initialProjectId}
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
            <Chip
              color="secondary"
              label={
                storageMode === "saved-cloud"
                  ? t("storageSavedCloud")
                  : storageMode === "local"
                    ? t("storageLocal")
                    : t("storageTemporary")
              }
              size="small"
              variant="outlined"
            />
            <Chip
              color={loadStatus === "ready" ? "success" : "default"}
              label={loadStatus === "ready" ? t("ready") : t("processing")}
              size="small"
            />
          </Stack>
        </Stack>

        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: "minmax(0, 1fr)",
              md: "188px minmax(0, 1fr)",
            },
          }}
        >
          <Box
            sx={{
              bgcolor: "background.paper",
              borderRight: { md: 1 },
              borderColor: "divider",
              minWidth: 0,
            }}
          >
            <WorkflowStepper activeStep={currentStep} onChange={goToStep} />
          </Box>

          <Box
            sx={{
              bgcolor: "background.default",
              minWidth: 0,
              p: { xs: 2, md: 3 },
            }}
          >
            {workspaceContent}
          </Box>
        </Box>
      </Paper>
    </Container>
  );
}
