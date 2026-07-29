"use client";

import AutoGraphOutlinedIcon from "@mui/icons-material/AutoGraphOutlined";
import FactCheckOutlinedIcon from "@mui/icons-material/FactCheckOutlined";
import FileDownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import UploadFileOutlinedIcon from "@mui/icons-material/UploadFileOutlined";
import {
  Alert,
  Box,
  Button,
  Container,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { useTranslations } from "next-intl";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { validateWebFile } from "@/features/import-data/file-policy";
import { useWorkspaceStore } from "@/features/workspace/workspace-store";

type HomeError = "invalid-type" | "too-large" | "read-error" | null;

export function HomeScreen() {
  const t = useTranslations("home");
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const selectFile = useWorkspaceStore((state) => state.selectFile);
  const [error, setError] = useState<HomeError>(null);
  const [dragActive, setDragActive] = useState(false);

  const openWorkspace = (file: File) => {
    const result = validateWebFile(file);
    if (!result.ok) {
      setError(result.reason === "too-large" ? "too-large" : "invalid-type");
      return;
    }

    setError(null);
    selectFile({
      name: file.name,
      size: file.size,
      type: file.type,
      isSample: false,
      sourceFile: file,
    });
    router.push("/workspace/new");
  };

  const handleFiles = (files: FileList | null) => {
    const file = files?.item(0);
    if (!file) return;
    try {
      openWorkspace(file);
    } catch {
      setError("read-error");
    }
  };

  const useSample = () => {
    selectFile({
      name: "labviz-sample.csv",
      size: 18_432,
      type: "text/csv",
      isSample: true,
    });
    router.push("/workspace/new");
  };

  const errorMessage =
    error === "too-large"
      ? t("tooLarge")
      : error === "invalid-type"
        ? t("invalidType")
        : error === "read-error"
          ? t("readError")
          : null;

  const steps = [
    {
      icon: <UploadFileOutlinedIcon />,
      title: t("stepImport"),
      body: t("stepImportBody"),
    },
    {
      icon: <FactCheckOutlinedIcon />,
      title: t("stepInspect"),
      body: t("stepInspectBody"),
    },
    {
      icon: <AutoGraphOutlinedIcon />,
      title: t("stepChart"),
      body: t("stepChartBody"),
    },
    {
      icon: <FileDownloadOutlinedIcon />,
      title: t("stepExport"),
      body: t("stepExportBody"),
    },
  ];

  return (
    <>
      <Box
        sx={{
          background:
            "radial-gradient(circle at 76% 10%, rgba(37,99,235,0.09), transparent 30%), radial-gradient(circle at 18% 38%, rgba(15,118,110,0.06), transparent 26%)",
          borderBottom: 1,
          borderColor: "divider",
          py: { xs: 7, md: 10 },
        }}
      >
        <Container maxWidth="lg">
          <Box
            sx={{
              alignItems: "center",
              display: "grid",
              gap: { xs: 5, md: 8 },
              gridTemplateColumns: { xs: "1fr", md: "minmax(0, 0.92fr) minmax(420px, 1.08fr)" },
            }}
          >
            <Stack spacing={3}>
              <Typography color="secondary.main" sx={{ fontWeight: 720 }} variant="overline">
                {t("eyebrow")}
              </Typography>
              <Typography component="h1" variant="h1">
                {t("title")}
              </Typography>
              <Typography
                color="text.secondary"
                sx={{ fontSize: { xs: 17, md: 19 }, maxWidth: 610 }}
              >
                {t("subtitle")}
              </Typography>
              <Stack
                direction="row"
                spacing={1}
                sx={{ alignItems: "center", color: "text.secondary" }}
              >
                <LockOutlinedIcon color="secondary" fontSize="small" />
                <Typography variant="body2">{t("privacyTitle")}</Typography>
              </Stack>
            </Stack>

            <Stack spacing={2}>
              <Paper
                onDragEnter={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragActive(false);
                  handleFiles(event.dataTransfer.files);
                }}
                sx={{
                  bgcolor: dragActive ? "primary.light" : "background.paper",
                  border: "2px dashed",
                  borderColor: dragActive ? "primary.main" : "divider",
                  boxShadow: 2,
                  p: { xs: 4, md: 6 },
                  textAlign: "center",
                  transition: "background-color 160ms ease, border-color 160ms ease",
                }}
              >
                <Stack spacing={2} sx={{ alignItems: "center" }}>
                  <Box
                    sx={{
                      alignItems: "center",
                      bgcolor: "primary.light",
                      borderRadius: "50%",
                      color: "primary.main",
                      display: "flex",
                      height: 64,
                      justifyContent: "center",
                      width: 64,
                    }}
                  >
                    <UploadFileOutlinedIcon fontSize="large" />
                  </Box>
                  <Box>
                    <Typography component="h2" variant="h3">
                      {t("dropTitle")}
                    </Typography>
                    <Typography color="text.secondary" sx={{ mt: 0.75 }} variant="body2">
                      {t("dropBody")}
                    </Typography>
                  </Box>
                  <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                    <Button onClick={() => inputRef.current?.click()} variant="contained">
                      {t("browse")}
                    </Button>
                    <Button onClick={useSample} variant="outlined">
                      {t("sample")}
                    </Button>
                  </Stack>
                  <input
                    ref={inputRef}
                    accept=".xlsx,.csv,.tsv,.txt,.json"
                    hidden
                    onChange={(event) => handleFiles(event.target.files)}
                    type="file"
                  />
                </Stack>
              </Paper>
              {errorMessage ? <Alert severity="error">{errorMessage}</Alert> : null}
            </Stack>
          </Box>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ py: { xs: 7, md: 9 } }}>
        <Typography component="h2" sx={{ textAlign: "center" }} variant="h2">
          {t("workflowTitle")}
        </Typography>
        <Box
          sx={{
            display: "grid",
            gap: 3,
            gridTemplateColumns: {
              xs: "1fr",
              sm: "repeat(2, minmax(0, 1fr))",
              lg: "repeat(4, minmax(0, 1fr))",
            },
            mt: 5,
          }}
        >
          {steps.map((step) => (
            <Paper key={step.title} sx={{ border: 1, borderColor: "divider", p: 3.5 }}>
              <Stack spacing={2}>
                <Box sx={{ color: "primary.main" }}>{step.icon}</Box>
                <Typography component="h3" variant="h3">
                  {step.title}
                </Typography>
                <Typography color="text.secondary" variant="body2">
                  {step.body}
                </Typography>
              </Stack>
            </Paper>
          ))}
        </Box>
        <Alert icon={<LockOutlinedIcon />} severity="info" sx={{ mt: 4 }}>
          <Typography sx={{ fontWeight: 650 }} variant="body2">
            {t("privacyTitle")}
          </Typography>
          <Typography variant="body2">{t("privacyBody")}</Typography>
        </Alert>
      </Container>
    </>
  );
}
