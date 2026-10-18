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
import { useLocale, useTranslations } from "next-intl";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { useInterfaceMode } from "@/components/layout/interface-mode";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { WorkflowStepper } from "@/components/common/workflow-stepper";

import { FigureStatus } from "./figure-status";
import type { FigureWorkspaceState } from "./figure-workspace-state";
import { rememberResearchNavigation, workspaceModeEvent, type ResearchTask } from "./research-navigation";
import { useWorkspaceProject } from "./use-workspace-project";
import { getWorkspaceLoadingCopy } from "./workspace-loading-state";
import { type WorkflowStep, useWorkspaceStore } from "./workspace-store";

const ImportStep = dynamic(() => import("./import-step").then((module) => module.ImportStep));
const InspectStep = dynamic(() => import("./inspect-step").then((module) => module.InspectStep));
const ResearchWorkspace = dynamic(() => import("./research-workspace").then((module) => module.ResearchWorkspace));
const ChartStep = dynamic(() => import("./chart-step").then((module) => module.ChartStep));
const ExportStep = dynamic(() => import("./export-step").then((module) => module.ExportStep));

export function WorkspaceScreen({
  initialProjectId,
  initialStep,
  initialResearchMode = false,
  initialResearchTask = "data",
}: {
  initialProjectId?: string;
  initialStep?: WorkflowStep;
  initialResearchMode?: boolean;
  initialResearchTask?: ResearchTask;
}) {
  const t = useTranslations("workspace");
  const locale = useLocale();
  const overview = useInterfaceMode().mode === "overview";
  const [figureState, setFigureState] = useState<FigureWorkspaceState | null>(null);
  const [researchTask, setResearchTask] = useState(initialResearchTask);
  const [researchMode, setResearchMode] = useState(initialResearchMode);
  const [researchOpened, setResearchOpened] = useState(initialResearchMode);
  const [quickPlotOpened, setQuickPlotOpened] = useState(!initialResearchMode);
  const projectId = useWorkspaceStore((state) => state.projectId);
  const file = useWorkspaceStore((state) => state.selectedFile);
  const currentStep = useWorkspaceStore((state) => state.currentStep);
  const storageMode = useWorkspaceStore((state) => state.storageMode);
  const goToStep = useWorkspaceStore((state) => state.goToStep);
  const { job, loadError, loadStatus, preview, quality, reimport, retry, uploading } =
    useWorkspaceProject(initialProjectId);
  const loadingCopy = getWorkspaceLoadingCopy(uploading, job);
  const initialStepApplied = useRef(false);
  const switchWorkspace = useCallback((next: boolean) => {
    if (next) setResearchOpened(true);
    else setQuickPlotOpened(true);
    setResearchMode(next);
    rememberResearchNavigation(next);
  }, []);
  useEffect(() => {
    const navigate = (event: Event) => {
      if (event instanceof CustomEvent && typeof event.detail === "boolean") switchWorkspace(event.detail);
    };
    window.addEventListener(workspaceModeEvent, navigate);
    return () => window.removeEventListener(workspaceModeEvent, navigate);
  }, [switchWorkspace]);
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
        active={!researchMode}
        onContinue={() => goToStep("inspect")}
        onReimport={reimport}
      />
    ) : currentStep === "inspect" ? (
      <InspectStep
        active={!researchMode}
        onBack={() => goToStep("import")}
        onContinue={() => goToStep("chart")}
      />
    ) : currentStep === "chart" ? (
      <ChartStep
        active={!researchMode}
        onBack={() => goToStep("inspect")}
        onContinue={() => goToStep("export")}
      />
    ) : (
      <ExportStep active={!researchMode} onBack={() => goToStep("chart")} />
    );

  return (
    <Container disableGutters maxWidth={false}>
      <Paper
        square
        sx={{
          border: 0,
          minHeight: "calc(100vh - 64px)",
          overflow: overview ? "clip" : "hidden",
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
            <Typography color="text.secondary" variant="caption" sx={overview ? { display: "none" } : undefined}>
              {t("project")}
            </Typography>
            <Typography component={overview ? "h1" : "p"} noWrap sx={{ fontWeight: 720, fontSize: overview ? 18 : undefined }}>
              {file?.name ?? initialProjectId}
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap", gap: 1, maxWidth: "100%" }}>
            {loadStatus === "ready" && projectId && <Button variant="outlined"
              onClick={() => switchWorkspace(!researchMode)}>
              {researchMode ? (locale.startsWith("zh") ? "返回快速出图" : "Back to quick plot")
                : (locale.startsWith("zh") ? "研究工作台" : "Research workspace")}
            </Button>}
            <Chip
              color="secondary"
              label={
                storageMode === "saved-cloud"
                  ? t("storageSavedCloud")
                  : storageMode === "local" || storageMode === "saved-local"
                    ? t("storageLocal")
                    : t("storageTemporary")
              }
              size="small"
              variant="outlined"
            />
            {overview && researchMode && researchTask === "figures" ? (figureState ? <FigureStatus state={figureState} compact /> : <Chip size="small" label={locale.startsWith("zh") ? "正在加载组合图…" : "Loading figure…"} />) : <Chip
              color={loadStatus === "ready" ? "success" : "default"}
              label={loadStatus === "ready" ? t("ready") : t("processing")}
              size="small"
              variant={loadStatus === "ready" ? "outlined" : "filled"}
            />}
          </Stack>
        </Stack>

        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: researchMode ? "minmax(0, 1fr)" : {
              xs: "minmax(0, 1fr)",
              md: "188px minmax(0, 1fr)",
            },
          }}
        >
          <Box
            hidden={researchMode}
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
              p: { xs: 2, md: overview ? 2 : 3 },
            }}
          >
            {quickPlotOpened && <Box hidden={researchMode}>{workspaceContent}</Box>}
            {researchOpened && projectId && <Box hidden={!researchMode}>
              <ResearchWorkspace onFigureStateChange={setFigureState} onTaskChange={setResearchTask} active={researchMode} initialTask={initialResearchTask} key={projectId} projectId={projectId} />
            </Box>}
          </Box>
        </Box>
      </Paper>
    </Container>
  );
}
