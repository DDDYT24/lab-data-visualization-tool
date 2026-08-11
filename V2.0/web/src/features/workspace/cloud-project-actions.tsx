"use client";

import CheckCircleOutlineRoundedIcon from "@mui/icons-material/CheckCircleOutlineRounded";
import CloudUploadOutlinedIcon from "@mui/icons-material/CloudUploadOutlined";
import ContentCopyRoundedIcon from "@mui/icons-material/ContentCopyRounded";
import ShareOutlinedIcon from "@mui/icons-material/ShareOutlined";
import {
  Alert,
  Button,
  Divider,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { EmailCodeDialog } from "@/features/auth/email-code-dialog";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

import { ProjectDescriptionEditor } from "./project-description-editor";
import { useWorkspaceStore } from "./workspace-store";

export function CloudProjectActions() {
  const t = useTranslations("cloudActions");
  const queryClient = useQueryClient();
  const projectId = useWorkspaceStore((state) => state.projectId);
  const storageMode = useWorkspaceStore((state) => state.storageMode);
  const shares = useWorkspaceStore((state) => state.shares);
  const markProjectSaved = useWorkspaceStore((state) => state.markProjectSaved);
  const addShare = useWorkspaceStore((state) => state.addShare);
  const removeShare = useWorkspaceStore((state) => state.removeShare);
  const [authOpen, setAuthOpen] = useState(false);
  const [shareOpen, setShareOpen] = useState(false);
  const [downloadsEnabled, setDownloadsEnabled] = useState(false);
  const [copied, setCopied] = useState(false);
  const authQuery = useQuery({
    queryKey: ["auth-state"],
    queryFn: ({ signal }) => labvizApi.getAuthState(signal),
  });
  const saveMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.saveProject(projectId);
    },
    onSuccess: (session) => {
      markProjectSaved(session);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
    onError: (error) => {
      if (error instanceof LabVizApiError && error.status === 401) {
        setAuthOpen(true);
      }
    },
  });
  const shareMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project is open.");
      return labvizApi.createShareLink(projectId, downloadsEnabled);
    },
    onSuccess: (share) => addShare(share),
  });
  const updateShareMutation = useMutation({
    mutationFn: () => {
      if (!projectId || !latestShare) throw new Error("No share link is open.");
      return labvizApi.updateShareLink(
        projectId,
        latestShare.token,
        downloadsEnabled,
      );
    },
    onSuccess: (share) => addShare(share),
  });
  const revokeShareMutation = useMutation({
    mutationFn: () => {
      if (!projectId || !latestShare) throw new Error("No share link is open.");
      return labvizApi.revokeShareLink(projectId, latestShare.token);
    },
    onSuccess: () => {
      if (latestShare) removeShare(latestShare.token);
      setCopied(false);
      setShareOpen(false);
    },
  });
  const latestShare = shares[0] ?? null;
  const authenticated = Boolean(authQuery.data?.authenticated);

  const save = () => {
    if (!authenticated) {
      setAuthOpen(true);
      return;
    }
    saveMutation.mutate();
  };

  return (
    <>
      <Stack spacing={1.25}>
        <Typography sx={{ fontWeight: 700 }}>{t("title")}</Typography>
        {storageMode === "saved-cloud" ? (
          <Alert icon={<CheckCircleOutlineRoundedIcon />} severity="success">
            {t("saved")}
          </Alert>
        ) : (
          <Button
            fullWidth
            loading={saveMutation.isPending}
            onClick={save}
            startIcon={<CloudUploadOutlinedIcon />}
            variant="outlined"
          >
            {t("save")}
          </Button>
        )}
        <Button
          disabled={storageMode !== "saved-cloud"}
          fullWidth
          onClick={() => {
            setDownloadsEnabled(latestShare?.downloadsEnabled ?? false);
            setCopied(false);
            setShareOpen(true);
          }}
          startIcon={<ShareOutlinedIcon />}
          variant="outlined"
        >
          {t("share")}
        </Button>
        {storageMode !== "saved-cloud" ? (
          <Typography color="text.secondary" variant="caption">
            {t("saveBeforeShare")}
          </Typography>
        ) : null}
        {saveMutation.error && !(saveMutation.error instanceof LabVizApiError && saveMutation.error.status === 401) ? (
          <Alert severity="error">
            {saveMutation.error instanceof LabVizApiError
              ? saveMutation.error.message
              : t("saveFailed")}
          </Alert>
        ) : null}
        <Divider />
        <ProjectDescriptionEditor />
      </Stack>

      <EmailCodeDialog
        onAuthenticated={() => saveMutation.mutate()}
        onClose={() => setAuthOpen(false)}
        open={authOpen}
      />
      <Dialog
        fullWidth
        maxWidth="xs"
        onClose={() => setShareOpen(false)}
        open={shareOpen}
      >
        <DialogTitle>{t("dialogTitle")}</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ pt: 1 }}>
            <Typography color="text.secondary" variant="body2">
              {t("dialogBody")}
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={downloadsEnabled}
                  onChange={(event) => setDownloadsEnabled(event.target.checked)}
                />
              }
              label={t("allowDownloads")}
            />
            {latestShare ? (
              <>
                <TextField
                  fullWidth
                  label={t("shareLink")}
                  slotProps={{ htmlInput: { readOnly: true } }}
                  value={latestShare.url}
                />
                {copied ? <Alert severity="success">{t("copied")}</Alert> : null}
              </>
            ) : null}
            {shareMutation.error ? (
              <Alert severity="error">
                {shareMutation.error instanceof LabVizApiError
                  ? shareMutation.error.message
                  : t("shareFailed")}
              </Alert>
            ) : null}
            {updateShareMutation.isSuccess ? (
              <Alert severity="success">{t("updated")}</Alert>
            ) : null}
            {updateShareMutation.error ? (
              <Alert severity="error">
                {updateShareMutation.error instanceof LabVizApiError
                  ? updateShareMutation.error.message
                  : t("updateFailed")}
              </Alert>
            ) : null}
            {revokeShareMutation.error ? (
              <Alert severity="error">
                {revokeShareMutation.error instanceof LabVizApiError
                  ? revokeShareMutation.error.message
                  : t("revokeFailed")}
              </Alert>
            ) : null}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShareOpen(false)}>{t("close")}</Button>
          {latestShare ? (
            <>
              <Button
                color="error"
                loading={revokeShareMutation.isPending}
                onClick={() => revokeShareMutation.mutate()}
              >
                {t("revoke")}
              </Button>
              <Button
                loading={updateShareMutation.isPending}
                onClick={() => updateShareMutation.mutate()}
                variant="outlined"
              >
                {t("update")}
              </Button>
              <Button
                onClick={() => {
                  void navigator.clipboard.writeText(latestShare.url).then(() => setCopied(true));
                }}
                startIcon={<ContentCopyRoundedIcon />}
                variant="contained"
              >
                {t("copy")}
              </Button>
            </>
          ) : (
            <Button
              loading={shareMutation.isPending}
              onClick={() => shareMutation.mutate()}
              variant="contained"
            >
              {t("create")}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </>
  );
}
