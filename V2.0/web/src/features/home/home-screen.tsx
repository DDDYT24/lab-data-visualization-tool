"use client";

import AutoGraphOutlinedIcon from "@mui/icons-material/AutoGraphOutlined";
import ExpandMoreRoundedIcon from "@mui/icons-material/ExpandMoreRounded";
import FactCheckOutlinedIcon from "@mui/icons-material/FactCheckOutlined";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import UploadFileOutlinedIcon from "@mui/icons-material/UploadFileOutlined";
import {
  Alert,
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
  Container,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { alpha, useTheme } from "@mui/material/styles";
import { useTranslations } from "next-intl";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import type { SampleExample } from "@/domain/api-contract";
import { validateWebFile } from "@/features/import-data/file-policy";
import { useWorkspaceStore } from "@/features/workspace/workspace-store";

import { ExampleGallery } from "./example-gallery";

type HomeError =
  | "invalid-type"
  | "too-large"
  | "read-error"
  | "experiment-title-required"
  | null;

export function HomeScreen() {
  const t = useTranslations("home");
  const theme = useTheme();
  const router = useRouter();
  const searchParams = useSearchParams();
  const inputRef = useRef<HTMLInputElement>(null);
  const browseButtonRef = useRef<HTMLButtonElement>(null);
  const selectFile = useWorkspaceStore((state) => state.selectFile);
  const [error, setError] = useState<HomeError>(null);
  const [dragActive, setDragActive] = useState(false);
  const [experimentTitle, setExperimentTitle] = useState("");
  const [runLabel, setRunLabel] = useState("");
  const [replicateId, setReplicateId] = useState("");
  const [batchId, setBatchId] = useState("");
  const [exampleGalleryOpen, setExampleGalleryOpen] = useState(
    () => searchParams.get("examples") === "1",
  );

  useEffect(() => {
    if (window.location.hash === "#import") browseButtonRef.current?.focus();
  }, []);

  const openWorkspace = (file: File) => {
    if (!experimentTitle.trim() && (runLabel.trim() || replicateId.trim() || batchId.trim())) {
      setError("experiment-title-required");
      return;
    }
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
      experimentTitle: experimentTitle.trim() || null,
      runLabel: runLabel.trim() || null,
      replicateId: replicateId.trim() || null,
      batchId: batchId.trim() || null,
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

  const useSample = (sample: SampleExample) => {
    selectFile({
      name: sample.filename,
      size: sample.byteSize,
      type: sample.mediaType,
      isSample: true,
      sampleSlug: sample.slug,
      recommendedChart: sample.recommendedChart,
    });
    setExampleGalleryOpen(false);
    router.push("/workspace/new");
  };

  const errorMessage =
    error === "too-large"
      ? t("tooLarge")
      : error === "invalid-type"
        ? t("invalidType")
        : error === "read-error"
          ? t("readError")
          : error === "experiment-title-required"
            ? t("experimentTitleRequired")
            : null;

  const features = [
    {
      icon: <AutoGraphOutlinedIcon />,
      title: t("featureCharts"),
    },
    {
      icon: <FactCheckOutlinedIcon />,
      title: t("featureReproducible"),
    },
    {
      icon: <LockOutlinedIcon />,
      title: t("featureLocal"),
    },
  ];

  return (
    <>
      <Box
        sx={{
          bgcolor: "background.default",
          backgroundImage: `radial-gradient(ellipse at 50% 48%, ${alpha(
            theme.palette.mode === "dark" ? theme.palette.common.white : theme.palette.primary.main,
            theme.palette.mode === "dark" ? 0.025 : 0.045,
          )} 0%, transparent 58%)`,
          display: "flex",
          minHeight: { md: "calc(100svh - 72px)" },
        }}
      >
        <Container
          maxWidth="lg"
          sx={{
            display: "flex",
            flex: 1,
            flexDirection: "column",
            py: { xs: 5, sm: 6, md: 3.5 },
          }}
        >
          <Stack
            spacing={{ xs: 1.5, md: 2 }}
            sx={{
              alignItems: "center",
              mb: { xs: 4, md: 4.5 },
              textAlign: "center",
            }}
          >
            <Typography
              component="h1"
              variant="h1"
              sx={{
                color: "text.primary",
                fontSize: { xs: "2.25rem", sm: "3rem", md: "3.75rem" },
                maxWidth: 980,
                textWrap: "balance",
              }}
            >
              {t("title")}
            </Typography>
            <Typography
              color="text.secondary"
              sx={{ fontSize: { xs: 16, md: 19 }, maxWidth: 680 }}
            >
              {t("subtitle")}
            </Typography>
          </Stack>

          <Paper
            aria-labelledby="home-drop-title"
            component="section"
            data-testid="home-dropzone"
            id="import"
            onDragEnter={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              const nextTarget = event.relatedTarget;
              if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
                return;
              }
              setDragActive(false);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
            }}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              handleFiles(event.dataTransfer.files);
            }}
            sx={{
              alignSelf: "center",
              bgcolor: "background.paper",
              border: 1,
              borderColor: dragActive ? "primary.main" : "divider",
              borderRadius: 4,
              boxShadow: `0 20px 52px ${alpha(
                theme.palette.mode === "dark" ? theme.palette.common.black : theme.palette.primary.main,
                theme.palette.mode === "dark" ? 0.28 : 0.07,
              )}`,
              maxWidth: 760,
              p: { xs: 1.25, sm: 2 },
              transition: "border-color 160ms ease, box-shadow 160ms ease",
              width: "100%",
            }}
          >
            <Stack
              spacing={{ xs: 1.75, sm: 2 }}
              sx={{
                alignItems: "center",
                bgcolor: dragActive ? "primary.light" : "transparent",
                border: "1.5px dashed",
                borderColor: dragActive ? "primary.main" : "divider",
                borderRadius: 3,
                px: { xs: 1.5, sm: 3 },
                py: { xs: 3.5, sm: 4.5 },
                textAlign: "center",
                transition: "background-color 160ms ease, border-color 160ms ease",
              }}
            >
              <Box
                aria-hidden="true"
                sx={{
                  alignItems: "center",
                  bgcolor: "primary.light",
                  borderRadius: 3,
                  color: "primary.main",
                  display: "flex",
                  height: 72,
                  justifyContent: "center",
                  width: 72,
                }}
              >
                <UploadFileOutlinedIcon sx={{ fontSize: 38 }} />
              </Box>
              <Box>
                <Typography component="h2" id="home-drop-title" variant="h3">
                  {t("dropTitle")}
                </Typography>
                <Typography color="text.secondary" sx={{ mt: 0.75 }} variant="body2">
                  {t("dropBody")}
                </Typography>
              </Box>
              <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={1}
                sx={{ alignItems: "center", pt: 0.5 }}
              >
                <Button
                  ref={browseButtonRef}
                  onClick={() => inputRef.current?.click()}
                  size="large"
                  sx={{ borderRadius: 2.5, minWidth: 184, py: 1.25 }}
                  variant="contained"
                >
                  {t("browse")}
                </Button>
                <Button
                  onClick={() => setExampleGalleryOpen(true)}
                  size="large"
                  sx={{ borderRadius: 2.5, px: 2 }}
                  variant="text"
                >
                  {t("sample")}
                </Button>
              </Stack>
              <input
                ref={inputRef}
                accept=".xlsx,.csv,.tsv,.txt,.json"
                hidden
                onChange={(event) => {
                  handleFiles(event.currentTarget.files);
                  event.currentTarget.value = "";
                }}
                type="file"
              />
              <Accordion
                disableGutters
                elevation={0}
                sx={{
                  "&::before": { display: "none" },
                  bgcolor: "transparent",
                  borderTop: 1,
                  borderColor: "divider",
                  maxWidth: 500,
                  textAlign: "left",
                  width: "100%",
                }}
              >
                <AccordionSummary expandIcon={<ExpandMoreRoundedIcon />}>
                  <Stack
                    direction="row"
                    spacing={1}
                    sx={{ alignItems: "center", justifyContent: "space-between", width: "100%" }}
                  >
                    <Typography sx={{ fontWeight: 600 }} variant="body2">
                      {t("experimentMetadata")}
                    </Typography>
                    <Typography color="text.secondary" variant="caption">
                      {t("optional")}
                    </Typography>
                  </Stack>
                </AccordionSummary>
                <AccordionDetails>
                  <Box
                    sx={{
                      display: "grid",
                      gap: 1.5,
                      gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))" },
                    }}
                  >
                    <TextField
                      label={t("experimentTitle")}
                      onChange={(event) => setExperimentTitle(event.target.value)}
                      size="small"
                      value={experimentTitle}
                    />
                    <TextField
                      helperText={t("runLabelHelp")}
                      label={t("runLabel")}
                      onChange={(event) => setRunLabel(event.target.value)}
                      size="small"
                      value={runLabel}
                    />
                    <TextField
                      label={t("replicateId")}
                      onChange={(event) => setReplicateId(event.target.value)}
                      size="small"
                      value={replicateId}
                    />
                    <TextField
                      label={t("batchId")}
                      onChange={(event) => setBatchId(event.target.value)}
                      size="small"
                      value={batchId}
                    />
                  </Box>
                </AccordionDetails>
              </Accordion>
            </Stack>
          </Paper>

          {errorMessage ? (
            <Alert
              severity="error"
              sx={{ alignSelf: "center", mt: 2, width: "100%", maxWidth: 760 }}
            >
              {errorMessage}
            </Alert>
          ) : null}

          <Box
            id="about"
            sx={{
              mt: { xs: 5, md: "auto" },
              pt: { xs: 4, md: 3 },
            }}
          >
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", sm: "repeat(3, minmax(0, 1fr))" },
                maxWidth: 960,
                mx: "auto",
              }}
            >
              {features.map((feature, index) => (
                <Stack
                  key={feature.title}
                  direction="row"
                  spacing={1.5}
                  sx={{
                    alignItems: "center",
                    borderColor: "divider",
                    borderRight: { sm: index < features.length - 1 ? 1 : 0 },
                    justifyContent: "center",
                    minHeight: { xs: 56, sm: 64 },
                  }}
                >
                  <Box
                    aria-hidden="true"
                    sx={{
                      alignItems: "center",
                      bgcolor: index === 1 ? "secondary.light" : "primary.light",
                      borderRadius: 2.5,
                      color: index === 1 ? "secondary.main" : "primary.main",
                      display: "flex",
                      height: 44,
                      justifyContent: "center",
                      width: 44,
                    }}
                  >
                    {feature.icon}
                  </Box>
                  <Typography color="text.secondary" sx={{ fontWeight: 600 }} variant="body2">
                    {feature.title}
                  </Typography>
                </Stack>
              ))}
            </Box>
            <Stack
              direction={{ xs: "column", sm: "row" }}
              spacing={0.75}
              sx={{
                alignItems: "center",
                color: "text.secondary",
                justifyContent: "center",
                mt: 1.5,
                textAlign: "center",
              }}
            >
              <LockOutlinedIcon fontSize="small" />
              <Typography variant="caption">
                {t("privacyTitle")} · {t("privacyBody")}
              </Typography>
            </Stack>
          </Box>
        </Container>
      </Box>
      <ExampleGallery
        onClose={() => setExampleGalleryOpen(false)}
        onSelect={useSample}
        open={exampleGalleryOpen}
      />
    </>
  );
}
