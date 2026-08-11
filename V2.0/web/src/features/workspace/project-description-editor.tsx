"use client";

import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import SaveOutlinedIcon from "@mui/icons-material/SaveOutlined";
import { Alert, Button, Stack, TextField, Typography } from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useMemo, useState } from "react";

import { PROJECT_DESCRIPTION_MAX_UTF8_BYTES } from "@/domain/api-contract";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

import { useWorkspaceStore } from "./workspace-store";

function utf8Length(value: string) {
  return new TextEncoder().encode(value).byteLength;
}

export function ProjectDescriptionEditor() {
  const t = useTranslations("projectDescription");
  const projectId = useWorkspaceStore((state) => state.projectId);
  const storageMode = useWorkspaceStore((state) => state.storageMode);
  const description = useWorkspaceStore((state) => state.description);
  const currentRevisionId = useWorkspaceStore(
    (state) => state.currentRevisionId,
  );
  const applyProjectDescription = useWorkspaceStore(
    (state) => state.applyProjectDescription,
  );
  const [draft, setDraft] = useState(description);

  const bytes = useMemo(() => utf8Length(draft), [draft]);
  const tooLarge = bytes > PROJECT_DESCRIPTION_MAX_UTF8_BYTES;
  const savedProject = storageMode === "saved-cloud";
  const canSave = Boolean(
    savedProject &&
      projectId &&
      currentRevisionId &&
      draft !== description &&
      !tooLarge,
  );

  const saveMutation = useMutation({
    mutationFn: () => {
      if (!projectId || !currentRevisionId) {
        throw new Error("Project description metadata is unavailable.");
      }
      return labvizApi.updateProjectDescription(
        projectId,
        draft,
        currentRevisionId,
      );
    },
    onSuccess: (updated) => {
      applyProjectDescription(updated.description, updated.revisionId);
      setDraft(updated.description);
    },
  });
  const refreshMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.getProject(projectId);
    },
    onSuccess: (session) => {
      if (!session.currentRevisionId) return;
      applyProjectDescription(session.description, session.currentRevisionId);
      setDraft(session.description);
      saveMutation.reset();
    },
  });

  const error = refreshMutation.error ?? saveMutation.error;
  const apiError = error instanceof LabVizApiError ? error : null;
  const conflict = apiError?.code === "project-revision-conflict";
  const unauthorized = apiError?.status === 401 || apiError?.status === 403;
  const deleted = apiError?.status === 404;

  return (
    <Stack spacing={1.25}>
      <Typography sx={{ fontWeight: 700 }}>{t("title")}</Typography>
      <Typography color="text.secondary" variant="body2">
        {t("body")}
      </Typography>
      <TextField
        disabled={
          !savedProject || currentRevisionId === null || saveMutation.isPending
        }
        error={tooLarge}
        fullWidth
        helperText={
          tooLarge
            ? t("tooLong", { bytes, limit: PROJECT_DESCRIPTION_MAX_UTF8_BYTES })
            : t("byteCount", { bytes, limit: PROJECT_DESCRIPTION_MAX_UTF8_BYTES })
        }
        label={t("label")}
        minRows={4}
        multiline
        onChange={(event) => {
          setDraft(event.target.value);
          saveMutation.reset();
        }}
        placeholder={t("placeholder")}
        value={draft}
      />
      {!savedProject ? (
        <Alert severity="info">{t("saveFirst")}</Alert>
      ) : currentRevisionId === null ? (
        <Alert severity="info">{t("loading")}</Alert>
      ) : description.length === 0 && draft.length === 0 ? (
        <Typography color="text.secondary" variant="caption">
          {t("empty")}
        </Typography>
      ) : null}
      {saveMutation.isSuccess ? (
        <Alert aria-live="polite" severity="success">
          {t("saved")}
        </Alert>
      ) : null}
      {conflict ? (
        <Alert
          action={
            <Button
              color="inherit"
              loading={refreshMutation.isPending}
              onClick={() => refreshMutation.mutate()}
              size="small"
              startIcon={<RefreshRoundedIcon />}
            >
              {t("reload")}
            </Button>
          }
          severity="warning"
        >
          {t("conflict")}
        </Alert>
      ) : unauthorized ? (
        <Alert severity="error">{t("unauthorized")}</Alert>
      ) : deleted ? (
        <Alert severity="error">{t("deleted")}</Alert>
      ) : error ? (
        <Alert severity="error">
          {apiError?.message ?? t("failed")}
        </Alert>
      ) : null}
      <Button
        disabled={!canSave}
        loading={saveMutation.isPending}
        onClick={() => saveMutation.mutate()}
        startIcon={<SaveOutlinedIcon />}
        variant="outlined"
      >
        {t("save")}
      </Button>
    </Stack>
  );
}
