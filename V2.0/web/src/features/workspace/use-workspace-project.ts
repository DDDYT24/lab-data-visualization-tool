"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

import { useWorkspaceStore } from "./workspace-store";

function errorMessage(error: unknown) {
  return error instanceof LabVizApiError
    ? error.message
    : "The project could not be loaded. Your source file was not changed.";
}

export function useWorkspaceProject(initialProjectId?: string) {
  const queryClient = useQueryClient();
  const startedRef = useRef(false);
  const selectedFile = useWorkspaceStore((state) => state.selectedFile);
  const projectId = useWorkspaceStore((state) => state.projectId);
  const job = useWorkspaceStore((state) => state.job);
  const preview = useWorkspaceStore((state) => state.preview);
  const quality = useWorkspaceStore((state) => state.quality);
  const loadStatus = useWorkspaceStore((state) => state.loadStatus);
  const loadError = useWorkspaceStore((state) => state.loadError);
  const openProject = useWorkspaceStore((state) => state.openProject);
  const setSession = useWorkspaceStore((state) => state.setSession);
  const hydrateWorkspace = useWorkspaceStore((state) => state.hydrateWorkspace);
  const setJob = useWorkspaceStore((state) => state.setJob);
  const setWorkspaceData = useWorkspaceStore(
    (state) => state.setWorkspaceData,
  );
  const setLoadError = useWorkspaceStore((state) => state.setLoadError);
  const selectFile = useWorkspaceStore((state) => state.selectFile);

  useEffect(() => {
    if (initialProjectId && projectId !== initialProjectId) {
      openProject(initialProjectId);
    }
  }, [initialProjectId, openProject, projectId]);

  const projectQuery = useQuery({
    queryKey: ["project", initialProjectId],
    queryFn: ({ signal }) =>
      labvizApi.getProjectWorkspace(initialProjectId!, signal),
    enabled: Boolean(initialProjectId),
  });

  useEffect(() => {
    if (projectQuery.data) hydrateWorkspace(projectQuery.data);
  }, [hydrateWorkspace, projectQuery.data]);

  const createMutation = useMutation({
    mutationFn: async () => {
      if (!selectedFile) throw new Error("No file selected");
      if (selectedFile.isSample) return labvizApi.createSampleProject();
      if (!selectedFile.sourceFile) {
        throw new Error("The selected file is no longer available.");
      }
      return labvizApi.createProject(selectedFile.sourceFile, {
        sheetName: selectedFile.sheetName,
        headerRow: selectedFile.headerRow,
      });
    },
    onSuccess: (session) => {
      setSession(session);
      window.history.replaceState(
        window.history.state,
        "",
        `/workspace/${encodeURIComponent(session.projectId)}`,
      );
    },
    onError: (error) => setLoadError(errorMessage(error)),
  });

  useEffect(() => {
    if (
      !initialProjectId &&
      selectedFile &&
      !projectId &&
      !startedRef.current
    ) {
      startedRef.current = true;
      createMutation.mutate();
    }
  }, [createMutation, initialProjectId, projectId, selectedFile]);

  const jobQuery = useQuery({
    queryKey: ["processing-job", job?.id],
    queryFn: ({ signal }) => labvizApi.getJob(job!.id, signal),
    enabled: Boolean(job && !["ready", "failed"].includes(job.stage)),
    refetchInterval: (query) => {
      const stage = query.state.data?.stage;
      return stage && ["ready", "failed"].includes(stage) ? false : 1_000;
    },
  });

  useEffect(() => {
    if (jobQuery.data) setJob(jobQuery.data);
  }, [jobQuery.data, setJob]);

  const dataReady = Boolean(
    projectId && (!job || ["ready"].includes(job.stage)),
  );
  const workspaceDataQuery = useQuery({
    queryKey: ["project-data", projectId],
    queryFn: async ({ signal }) => {
      const [nextPreview, nextQuality] = await Promise.all([
        labvizApi.getPreview(projectId!, signal),
        labvizApi.getQuality(projectId!, signal),
      ]);
      return { preview: nextPreview, quality: nextQuality };
    },
    enabled: dataReady && !initialProjectId,
    retry: (failureCount, error) =>
      error instanceof LabVizApiError && error.code === "processing-not-ready"
        ? failureCount < 8
        : failureCount < 2,
    retryDelay: (attempt) => Math.min(500 * 2 ** attempt, 4_000),
  });

  useEffect(() => {
    if (workspaceDataQuery.data) {
      setWorkspaceData(
        workspaceDataQuery.data.preview,
        workspaceDataQuery.data.quality,
      );
    }
  }, [setWorkspaceData, workspaceDataQuery.data]);

  useEffect(() => {
    const error =
      projectQuery.error ??
      jobQuery.error ??
      workspaceDataQuery.error;
    if (error) setLoadError(errorMessage(error));
  }, [
    jobQuery.error,
    projectQuery.error,
    setLoadError,
    workspaceDataQuery.error,
  ]);

  const reimportMutation = useMutation({
    mutationFn: ({
      file,
      headerRow,
      sheetName,
    }: {
      file: File;
      headerRow: number;
      sheetName: string | null;
    }) => labvizApi.createProject(file, { headerRow, sheetName }),
    onSuccess: (session) => {
      setSession(session);
      window.history.replaceState(
        window.history.state,
        "",
        `/workspace/${encodeURIComponent(session.projectId)}`,
      );
    },
    onError: (error) => setLoadError(errorMessage(error)),
  });

  const reimport = (sheetName: string | null, headerRow: number) => {
    if (!selectedFile?.sourceFile) return;
    const sourceFile = selectedFile.sourceFile;
    selectFile({ ...selectedFile, headerRow, sheetName });
    reimportMutation.mutate({ file: sourceFile, headerRow, sheetName });
  };

  const retry = () => {
    startedRef.current = false;
    void queryClient.invalidateQueries();
    if (initialProjectId) {
      openProject(initialProjectId);
    } else if (selectedFile) {
      selectFile(selectedFile);
    }
  };

  return {
    job,
    loadError,
    loadStatus,
    preview,
    quality,
    reimport,
    retry,
    uploading: createMutation.isPending || reimportMutation.isPending,
  };
}
