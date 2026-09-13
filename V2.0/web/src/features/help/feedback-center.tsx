"use client";

import BugReportOutlinedIcon from "@mui/icons-material/BugReportOutlined";
import ContentCopyRoundedIcon from "@mui/icons-material/ContentCopyRounded";
import OpenInNewRoundedIcon from "@mui/icons-material/OpenInNewRounded";
import SaveAltRoundedIcon from "@mui/icons-material/SaveAltRounded";
import {
  Alert,
  Button,
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
import { useEffect, useMemo, useState } from "react";

import {
  serializeFeedbackDiagnostic,
  type FeedbackCategory,
} from "./feedback-diagnostic";

async function copyText(value: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(value);
      return true;
    } catch {
      // Some local HTTP/browser contexts expose the API but reject clipboard writes.
      // Fall through to the older document command so offline feedback still works.
    }
  }
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  return copied;
}

export function FeedbackCenter() {
  const t = useTranslations("help");
  const [category, setCategory] = useState<FeedbackCategory>("bug");
  const [description, setDescription] = useState("");
  const [context, setContext] = useState({ pathname: "", userAgent: "" });
  const [notice, setNotice] = useState<"copied" | "saved" | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setContext({ pathname: window.location.pathname, userAgent: navigator.userAgent });
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const diagnostic = useMemo(
    () =>
      serializeFeedbackDiagnostic({
        category,
        description,
        pathname: context.pathname,
        userAgent: context.userAgent,
      }),
    [category, context.pathname, context.userAgent, description],
  );

  const saveDiagnostic = () => {
    const blob = new Blob([diagnostic], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "labviz-feedback-diagnostic.json";
    anchor.click();
    URL.revokeObjectURL(url);
    setNotice("saved");
  };

  const openIssue = () => {
    if (!window.confirm(t("externalWarning"))) return;
    const params = new URLSearchParams({
      title: t("feedbackIssueTitle"),
      body: diagnostic,
    });
    window.open(
      `https://github.com/DDDYT24/lab-data-visualization-tool/issues/new?${params.toString()}`,
      "_blank",
      "noopener,noreferrer",
    );
  };

  return (
    <Paper sx={{ border: 1, borderColor: "divider", p: { xs: 2, md: 3 } }}>
      <Stack spacing={2}>
        <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
          <BugReportOutlinedIcon color="secondary" />
          <Typography component="h2" sx={{ fontWeight: 750 }}>
            {t("feedbackTitle")}
          </Typography>
        </Stack>
        <Typography color="text.secondary" variant="body2">
          {t("feedbackDescription")}
        </Typography>
        <Alert severity="info">{t("feedbackPrivacy")}</Alert>
        <FormControl fullWidth size="small">
          <InputLabel id="feedback-category-label">{t("feedbackCategory")}</InputLabel>
          <Select
            label={t("feedbackCategory")}
            labelId="feedback-category-label"
            onChange={(event) => setCategory(event.target.value as FeedbackCategory)}
            value={category}
          >
            <MenuItem value="bug">{t("feedbackCategories.bug")}</MenuItem>
            <MenuItem value="usability">{t("feedbackCategories.usability")}</MenuItem>
            <MenuItem value="scientific">{t("feedbackCategories.scientific")}</MenuItem>
            <MenuItem value="feature">{t("feedbackCategories.feature")}</MenuItem>
          </Select>
        </FormControl>
        <TextField
          fullWidth
          label={t("feedbackDescriptionLabel")}
          multiline
          minRows={3}
          onChange={(event) => setDescription(event.target.value)}
          placeholder={t("feedbackDescriptionPlaceholder")}
          value={description}
        />
        <BoxPreview label={t("feedbackPreview")} value={diagnostic} />
        <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
          <Button
            onClick={async () => {
              if (await copyText(diagnostic)) setNotice("copied");
            }}
            startIcon={<ContentCopyRoundedIcon />}
            variant="outlined"
          >
            {t("copyFeedback")}
          </Button>
          <Button onClick={saveDiagnostic} startIcon={<SaveAltRoundedIcon />} variant="outlined">
            {t("saveFeedback")}
          </Button>
          <Button onClick={openIssue} startIcon={<OpenInNewRoundedIcon />} variant="contained">
            {t("openIssue")}
          </Button>
        </Stack>
        {notice === "copied" ? <Alert severity="success">{t("copiedFeedback")}</Alert> : null}
        {notice === "saved" ? <Alert severity="success">{t("savedFeedback")}</Alert> : null}
      </Stack>
    </Paper>
  );
}

function BoxPreview({ label, value }: { label: string; value: string }) {
  return (
    <Paper
      aria-label={label}
      component="pre"
      sx={{
        bgcolor: "action.hover",
        fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
        fontSize: 12,
        m: 0,
        maxWidth: "100%",
        overflowX: "auto",
        p: 1.5,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
      }}
    >
      {value}
    </Paper>
  );
}
