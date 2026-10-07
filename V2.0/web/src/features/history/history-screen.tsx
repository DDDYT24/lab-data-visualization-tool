"use client";

import AddRoundedIcon from "@mui/icons-material/AddRounded";
import AutoGraphRoundedIcon from "@mui/icons-material/AutoGraphRounded";
import ContentCopyRoundedIcon from "@mui/icons-material/ContentCopyRounded";
import DeleteOutlineRoundedIcon from "@mui/icons-material/DeleteOutlineRounded";
import EditRoundedIcon from "@mui/icons-material/EditRounded";
import FileDownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";
import ImageOutlinedIcon from "@mui/icons-material/ImageOutlined";
import MoreHorizRoundedIcon from "@mui/icons-material/MoreHorizRounded";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";
import VisibilityOutlinedIcon from "@mui/icons-material/VisibilityOutlined";
import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  FormControl,
  IconButton,
  InputLabel,
  InputAdornment,
  Menu,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { DataPreviewTable } from "@/components/common/data-preview-table";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function HistoryScreen() {
  const t = useTranslations("history");
  const router = useRouter();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [chartType, setChartType] = useState("all");
  const [updatedWithin, setUpdatedWithin] = useState("all");
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [deleteProjectId, setDeleteProjectId] = useState<string | null>(null);
  const [selectedHistoryProjectId, setSelectedHistoryProjectId] = useState<string | null>(null);
  const [deleteFigureId, setDeleteFigureId] = useState<string | null>(null);
  const [fullPreviewFigureId, setFullPreviewFigureId] = useState<string | null>(null);
  const [filterReferenceTime] = useState(() => Date.now());
  const projectsQuery = useQuery({
    queryKey: ["projects"],
    queryFn: ({ signal }) => labvizApi.listProjects(signal),
  });
  const allProjects = useMemo(
    () => projectsQuery.data?.projects ?? [],
    [projectsQuery.data?.projects],
  );
  const chartTypes = useMemo(
    () => Array.from(new Set(allProjects.map((project) => project.chartType))).sort(),
    [allProjects],
  );
  const projects = useMemo(() => {
    const normalized = search.trim().toLowerCase();
    const maximumAge =
      updatedWithin === "7" ? 7 * 86_400_000 : updatedWithin === "30" ? 30 * 86_400_000 : null;
    return allProjects.filter((project) => {
      const matchesText =
        !normalized ||
        project.title.toLowerCase().includes(normalized) ||
        project.sourceName.toLowerCase().includes(normalized) ||
        project.experiment?.title.toLowerCase().includes(normalized) ||
        project.experiment?.runLabel.toLowerCase().includes(normalized) ||
        project.experiment?.replicateId?.toLowerCase().includes(normalized) ||
        project.experiment?.batchId?.toLowerCase().includes(normalized);
      const matchesType = chartType === "all" || project.chartType === chartType;
      const matchesDate =
        maximumAge === null ||
        filterReferenceTime - new Date(project.updatedAt).getTime() <= maximumAge;
      return matchesText && matchesType && matchesDate;
    });
  }, [allProjects, chartType, filterReferenceTime, search, updatedWithin]);
  const activeProject = allProjects.find((project) => project.id === activeProjectId);
  const deleteProject = allProjects.find((project) => project.id === deleteProjectId);
  const closeMenu = () => {
    setMenuAnchor(null);
    setActiveProjectId(null);
  };
  const duplicateMutation = useMutation({
    mutationFn: (projectId: string) => labvizApi.duplicateProject(projectId),
    onSuccess: async (session) => {
      closeMenu();
      await queryClient.invalidateQueries({ queryKey: ["projects"] });
      router.push(`/workspace/${session.projectId}`);
    },
  });
  const deleteMutation = useMutation({
    mutationFn: (projectId: string) => labvizApi.deleteProject(projectId),
    onSuccess: async () => {
      setDeleteProjectId(null);
      setSearch("");
      setChartType("all");
      setUpdatedWithin("all");
      await queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
  const selectedHistoryProject = allProjects.find(
    (project) => project.id === selectedHistoryProjectId,
  );
  const historyPreviewQuery = useQuery({
    queryKey: ["history-preview", selectedHistoryProjectId],
    queryFn: ({ signal }) => labvizApi.getPreview(selectedHistoryProjectId!, signal),
    enabled: Boolean(selectedHistoryProjectId),
  });
  const figuresQuery = useQuery({
    queryKey: ["figure-snapshots", selectedHistoryProjectId],
    queryFn: ({ signal }) => labvizApi.listFigureSnapshots(selectedHistoryProjectId!, signal),
    enabled: Boolean(selectedHistoryProjectId),
  });
  const deleteFigureMutation = useMutation({
    mutationFn: (snapshotId: string) =>
      labvizApi.deleteFigureSnapshot(selectedHistoryProjectId!, snapshotId),
    onSuccess: async () => {
      setDeleteFigureId(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["figure-snapshots", selectedHistoryProjectId] }),
        queryClient.invalidateQueries({ queryKey: ["projects"] }),
      ]);
    },
  });
  const figureToDelete = figuresQuery.data?.snapshots.find(
    (snapshot) => snapshot.id === deleteFigureId,
  );
  const figureToPreview = figuresQuery.data?.snapshots.find(
    (snapshot) => snapshot.id === fullPreviewFigureId,
  );
  const totalSnapshots = allProjects.reduce(
    (total, project) => total + project.figureSnapshotCount,
    0,
  );
  const totalFigureBytes = allProjects.reduce(
    (total, project) => total + project.figureStorageBytes,
    0,
  );

  return (
    <Stack spacing={3}>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={2}
        sx={{ alignItems: { sm: "center" }, justifyContent: "space-between" }}
      >
        <Box>
          <Typography component="h1" sx={{ fontSize: 28, fontWeight: 750 }}>
            {t("title")}
          </Typography>
          <Typography color="text.secondary" sx={{ mt: 0.5 }} variant="body2">
            {t("description")}
          </Typography>
          {totalSnapshots > 0 ? (
            <Typography color="text.secondary" sx={{ mt: 0.5 }} variant="caption">
              {t("libraryStats", { count: totalSnapshots, size: formatBytes(totalFigureBytes) })}
            </Typography>
          ) : null}
        </Box>
        <Button
          component={Link}
          href="/#import"
          startIcon={<AddRoundedIcon />}
          variant="contained"
        >
          {t("newAnalysis")}
        </Button>
      </Stack>

      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={1.5}
        sx={{ alignItems: { sm: "center" } }}
      >
        <TextField
          onChange={(event) => setSearch(event.target.value)}
          placeholder={t("search")}
          size="small"
          slotProps={{
            input: {
              startAdornment: (
                <InputAdornment position="start">
                  <SearchRoundedIcon fontSize="small" />
                </InputAdornment>
              ),
            },
          }}
          sx={{ maxWidth: 360, width: "100%" }}
          value={search}
        />
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel id="history-chart-type-label">{t("chartType")}</InputLabel>
          <Select
            label={t("chartType")}
            labelId="history-chart-type-label"
            onChange={(event) => setChartType(event.target.value)}
            value={chartType}
          >
            <MenuItem value="all">{t("allChartTypes")}</MenuItem>
            {chartTypes.map((type) => (
              <MenuItem key={type} value={type}>
                {t(`chartTypes.${type}`)}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel id="history-updated-label">{t("updated")}</InputLabel>
          <Select
            label={t("updated")}
            labelId="history-updated-label"
            onChange={(event) => setUpdatedWithin(event.target.value)}
            value={updatedWithin}
          >
            <MenuItem value="all">{t("anyTime")}</MenuItem>
            <MenuItem value="7">{t("last7Days")}</MenuItem>
            <MenuItem value="30">{t("last30Days")}</MenuItem>
          </Select>
        </FormControl>
        {(search || chartType !== "all" || updatedWithin !== "all") && (
          <Chip
            label={t("clearFilters")}
            onDelete={() => {
              setSearch("");
              setChartType("all");
              setUpdatedWithin("all");
            }}
            size="small"
            variant="outlined"
          />
        )}
      </Stack>

      {duplicateMutation.error ? (
        <Alert severity="error" onClose={() => duplicateMutation.reset()}>
          {duplicateMutation.error instanceof LabVizApiError
            ? duplicateMutation.error.message
            : t("duplicateFailed")}
        </Alert>
      ) : null}

      {projectsQuery.isPending ? (
        <ApiStatePanel
          description={t("loadingDescription")}
          kind="loading"
          title={t("loadingTitle")}
        />
      ) : projectsQuery.error ? (
        <ApiStatePanel
          actionLabel={t("retry")}
          description={
            projectsQuery.error instanceof LabVizApiError
              ? projectsQuery.error.message
              : t("loadFailed")
          }
          kind="error"
          onAction={() => void projectsQuery.refetch()}
          title={t("unavailable")}
        />
      ) : projects.length === 0 ? (
        <ApiStatePanel
          description={
            search || chartType !== "all" || updatedWithin !== "all"
              ? t("noMatchDescription")
              : t("emptyDescription")
          }
          kind="empty"
          secondaryAction={
            <Button component={Link} href="/#import" variant="contained">
              {t("startAnalysis")}
            </Button>
          }
          title={
            search || chartType !== "all" || updatedWithin !== "all"
              ? t("noMatchTitle")
              : t("emptyTitle")
          }
        />
      ) : (
        <Box
          sx={{
            display: "grid",
            gap: 2,
            gridTemplateColumns: {
              xs: "1fr",
              sm: "repeat(2, minmax(0, 1fr))",
              xl: "repeat(3, minmax(0, 1fr))",
            },
          }}
        >
          {projects.map((project) => (
            <Paper
              key={project.id}
              sx={{ border: 1, borderColor: "divider", overflow: "hidden" }}
            >
              <Box
                sx={{
                  alignItems: "center",
                  bgcolor: "background.default",
                  borderBottom: 1,
                  borderColor: "divider",
                  display: "flex",
                  height: 168,
                  justifyContent: "center",
                  overflow: "hidden",
                }}
              >
                {project.thumbnailUrl ? (
                  <Box
                    alt={t("thumbnailAlt", { title: project.title })}
                    component="img"
                    src={project.thumbnailUrl}
                    sx={{ height: "100%", objectFit: "cover", width: "100%" }}
                  />
                ) : (
                  <AutoGraphRoundedIcon color="primary" sx={{ fontSize: 44 }} />
                )}
              </Box>
              <Stack spacing={1.5} sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  spacing={1}
                  sx={{ alignItems: "flex-start", justifyContent: "space-between" }}
                >
                  <Box sx={{ minWidth: 0 }}>
                    <Typography noWrap sx={{ fontWeight: 700 }}>
                      {project.title}
                    </Typography>
                    <Typography color="text.secondary" noWrap variant="body2">
                      {project.sourceName}
                    </Typography>
                    {project.experiment ? (
                      <Stack spacing={0.25} sx={{ mt: 1 }}>
                        <Typography color="primary.main" noWrap variant="body2">
                          {project.experiment.title}
                        </Typography>
                        <Typography color="text.secondary" noWrap variant="caption">
                          {t("experimentRun", { run: project.experiment.runLabel })}
                        </Typography>
                      </Stack>
                    ) : null}
                  </Box>
                  <IconButton
                    aria-label={t("actionsFor", { title: project.title })}
                    onClick={(event) => {
                      setActiveProjectId(project.id);
                      setMenuAnchor(event.currentTarget);
                    }}
                    size="small"
                  >
                    <MoreHorizRoundedIcon />
                  </IconButton>
                </Stack>
                <Stack
                  direction="row"
                  spacing={1}
                  sx={{ alignItems: "center", flexWrap: "wrap", rowGap: 1 }}
                >
                  <Chip
                    label={t("local")}
                    size="small"
                    variant="outlined"
                  />
                  <Typography color="text.secondary" variant="caption">
                    {new Intl.DateTimeFormat(undefined, {
                      dateStyle: "medium",
                    }).format(new Date(project.updatedAt))}
                  </Typography>
                  {project.experiment?.replicateId ? (
                    <Chip
                      label={t("replicate", { id: project.experiment.replicateId })}
                      size="small"
                      variant="outlined"
                    />
                  ) : null}
                  {project.experiment?.batchId ? (
                    <Chip
                      label={t("batch", { id: project.experiment.batchId })}
                      size="small"
                      variant="outlined"
                    />
                  ) : null}
                </Stack>
                <Button
                  component={Link}
                  href={`/workspace/${project.id}`}
                  variant="outlined"
                >
                  {t("continueEditing")}
                </Button>
                <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between" }}>
                  <Typography color="text.secondary" variant="caption">
                    {t("figureStats", {
                      count: project.figureSnapshotCount,
                      size: formatBytes(project.figureStorageBytes),
                    })}
                  </Typography>
                  <Button
                    onClick={() => setSelectedHistoryProjectId(project.id)}
                    size="small"
                    startIcon={<VisibilityOutlinedIcon />}
                  >
                    {t("viewDataAndFigures")}
                  </Button>
                </Stack>
              </Stack>
            </Paper>
          ))}
        </Box>
      )}

      <Menu anchorEl={menuAnchor} onClose={closeMenu} open={Boolean(menuAnchor)}>
        {activeProject ? (
          <MenuItem
            component={Link}
            href={`/workspace/${activeProject.id}`}
            onClick={closeMenu}
          >
            <EditRoundedIcon fontSize="small" sx={{ mr: 1 }} />
            {t("continueEditing")}
          </MenuItem>
        ) : null}
        {activeProject ? (
          <MenuItem
            component={Link}
            href={`/workspace/${activeProject.id}?step=export`}
            onClick={closeMenu}
          >
            <FileDownloadOutlinedIcon fontSize="small" sx={{ mr: 1 }} />
            {t("openExport")}
          </MenuItem>
        ) : null}
        <MenuItem
          disabled={!activeProject || duplicateMutation.isPending}
          onClick={() => {
            if (!activeProject) return;
            const projectId = activeProject.id;
            closeMenu();
            duplicateMutation.mutate(projectId);
          }}
        >
          <ContentCopyRoundedIcon fontSize="small" sx={{ mr: 1 }} />
          {t("duplicate")}
        </MenuItem>
        <MenuItem
          disabled={!activeProject}
          onClick={() => {
            if (activeProject) setDeleteProjectId(activeProject.id);
            closeMenu();
          }}
          sx={{ color: "error.main" }}
        >
          <DeleteOutlineRoundedIcon fontSize="small" sx={{ mr: 1 }} />
          {t("delete")}
        </MenuItem>
      </Menu>

      <Dialog
        fullWidth
        maxWidth="xl"
        onClose={() => setSelectedHistoryProjectId(null)}
        open={Boolean(selectedHistoryProject)}
      >
        <DialogTitle>{t("historyDetailTitle", { title: selectedHistoryProject?.title ?? "" })}</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={3}>
            <Box>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ alignItems: { sm: "center" }, justifyContent: "space-between", mb: 1.5 }}>
                <Box>
                  <Typography component="h2" variant="h6">{t("processedData")}</Typography>
                  {historyPreviewQuery.data ? (
                    <Typography color="text.secondary" variant="caption">
                      {t("previewRows", {
                        shown: historyPreviewQuery.data.rows.length,
                        total: historyPreviewQuery.data.totalRows,
                      })}
                    </Typography>
                  ) : null}
                </Box>
                {selectedHistoryProject ? (
                  <Button
                    component="a"
                    href={labvizApi.cleanedDataUrl(selectedHistoryProject.id)}
                    startIcon={<FileDownloadOutlinedIcon />}
                    variant="outlined"
                  >
                    {t("downloadCleanedData")}
                  </Button>
                ) : null}
              </Stack>
              {historyPreviewQuery.isPending ? (
                <ApiStatePanel compact description={t("loadingDataDescription")} kind="loading" title={t("loadingData")} />
              ) : historyPreviewQuery.error ? (
                <Alert severity="error">
                  {historyPreviewQuery.error instanceof LabVizApiError
                    ? historyPreviewQuery.error.message
                    : t("dataLoadFailed")}
                </Alert>
              ) : historyPreviewQuery.data ? (
                <DataPreviewTable preview={historyPreviewQuery.data} />
              ) : null}
            </Box>
            <Divider />
            <Box>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ alignItems: { sm: "center" }, justifyContent: "space-between", mb: 1.5 }}>
                <Box>
                  <Typography component="h2" variant="h6">{t("savedFigures")}</Typography>
                  <Typography color="text.secondary" variant="caption">
                    {figuresQuery.data
                      ? t("figureStorage", {
                          count: figuresQuery.data.snapshots.length,
                          size: formatBytes(figuresQuery.data.totalBytes),
                        })
                      : t("figureRetention")}
                  </Typography>
                </Box>
              </Stack>
              {figuresQuery.isPending ? (
                <ApiStatePanel compact description={t("loadingFiguresDescription")} kind="loading" title={t("loadingFigures")} />
              ) : figuresQuery.error ? (
                <Alert severity="error">
                  {figuresQuery.error instanceof LabVizApiError
                    ? figuresQuery.error.message
                    : t("figuresLoadFailed")}
                </Alert>
              ) : figuresQuery.data?.snapshots.length ? (
                <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))", lg: "repeat(3, minmax(0, 1fr))" } }}>
                  {figuresQuery.data.snapshots.map((snapshot) => (
                    <Paper key={snapshot.id} variant="outlined" sx={{ minWidth: 0, overflow: "hidden" }}>
                      <Box sx={{ alignItems: "center", bgcolor: "background.default", display: "flex", height: 190, justifyContent: "center", overflow: "hidden" }}>
                        {snapshot.previewUrl ? (
                          <Button
                            aria-label={t("openFullFigure", { title: snapshot.title })}
                            onClick={() => setFullPreviewFigureId(snapshot.id)}
                            sx={{ height: "100%", p: 0, width: "100%" }}
                          >
                            <Box alt={snapshot.title} component="img" src={snapshot.thumbnailUrl ?? snapshot.previewUrl} sx={{ height: "100%", objectFit: "contain", width: "100%" }} />
                          </Button>
                        ) : (
                          <Stack spacing={1} sx={{ alignItems: "center" }}>
                            <ImageOutlinedIcon color="primary" sx={{ fontSize: 42 }} />
                            <Typography color="text.secondary" variant="caption">{t("pdfPreviewUnavailable")}</Typography>
                          </Stack>
                        )}
                      </Box>
                      <Stack spacing={1} sx={{ p: 1.5 }}>
                        <Typography noWrap sx={{ fontWeight: 650 }}>{snapshot.title}</Typography>
                        <Typography color="text.secondary" variant="caption">
                          {snapshot.format.toUpperCase()} · {formatBytes(snapshot.byteSize)} · {new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(snapshot.createdAt))}
                        </Typography>
                        <Stack direction="row" spacing={1}>
                          <Button component="a" href={snapshot.downloadUrl} size="small" startIcon={<FileDownloadOutlinedIcon />} variant="outlined">
                            {t("downloadFigure")}
                          </Button>
                          <Button color="error" onClick={() => setDeleteFigureId(snapshot.id)} size="small">
                            {t("deleteFigure")}
                          </Button>
                        </Stack>
                      </Stack>
                    </Paper>
                  ))}
                </Box>
              ) : (
                <ApiStatePanel compact kind="empty" title={t("noFiguresTitle")} description={t("noFiguresDescription")} />
              )}
            </Box>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedHistoryProjectId(null)}>{t("closeHistoryDetail")}</Button>
        </DialogActions>
      </Dialog>

      <Dialog
        fullWidth
        maxWidth="lg"
        onClose={() => setFullPreviewFigureId(null)}
        open={Boolean(figureToPreview?.previewUrl)}
      >
        <DialogTitle>{figureToPreview?.title}</DialogTitle>
        <DialogContent>
          {figureToPreview?.previewUrl ? (
            <Box alt={figureToPreview.title} component="img" src={figureToPreview.previewUrl} sx={{ display: "block", maxHeight: "75vh", maxWidth: "100%", mx: "auto", objectFit: "contain" }} />
          ) : null}
        </DialogContent>
        <DialogActions>
          {figureToPreview ? (
            <Button component="a" href={figureToPreview.downloadUrl} startIcon={<FileDownloadOutlinedIcon />}>
              {t("downloadFigure")}
            </Button>
          ) : null}
          <Button onClick={() => setFullPreviewFigureId(null)}>{t("closeHistoryDetail")}</Button>
        </DialogActions>
      </Dialog>

      <Dialog
        onClose={() => !deleteFigureMutation.isPending && setDeleteFigureId(null)}
        open={Boolean(figureToDelete)}
      >
        <DialogTitle>{t("deleteFigureTitle")}</DialogTitle>
        <DialogContent>
          <DialogContentText>{t("deleteFigureDescription", { title: figureToDelete?.title ?? "" })}</DialogContentText>
          {deleteFigureMutation.error ? (
            <Alert severity="error" sx={{ mt: 2 }}>
              {deleteFigureMutation.error instanceof LabVizApiError
                ? deleteFigureMutation.error.message
                : t("deleteFigureFailed")}
            </Alert>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button disabled={deleteFigureMutation.isPending} onClick={() => setDeleteFigureId(null)}>{t("cancel")}</Button>
          <Button color="error" disabled={!figureToDelete || deleteFigureMutation.isPending} onClick={() => figureToDelete && deleteFigureMutation.mutate(figureToDelete.id)} variant="contained">
            {deleteFigureMutation.isPending ? t("deleting") : t("deleteConfirm")}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        onClose={() => !deleteMutation.isPending && setDeleteProjectId(null)}
        open={Boolean(deleteProject)}
      >
        <DialogTitle>{t("deleteTitle")}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {deleteProject
              ? t("deleteDescription", { title: deleteProject.title })
              : t("deleteFallback")}
          </DialogContentText>
          {deleteMutation.error ? (
            <ApiStatePanel
              compact
              description={
                deleteMutation.error instanceof LabVizApiError
                  ? deleteMutation.error.message
                  : t("deleteFailedDescription")
              }
              kind="error"
              title={t("deleteFailedTitle")}
            />
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button
            disabled={deleteMutation.isPending}
            onClick={() => setDeleteProjectId(null)}
          >
            {t("cancel")}
          </Button>
          <Button
            color="error"
            disabled={!deleteProject || deleteMutation.isPending}
            onClick={() => deleteProject && deleteMutation.mutate(deleteProject.id)}
            variant="contained"
          >
            {deleteMutation.isPending ? t("deleting") : t("deleteConfirm")}
          </Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
