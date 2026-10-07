"use client";

import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import OpenInNewRoundedIcon from "@mui/icons-material/OpenInNewRounded";
import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  Paper,
  Skeleton,
  Stack,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";

import type { SampleExample } from "@/domain/api-contract";
import { labvizApi } from "@/lib/api/labviz-api";

function localized(
  value: { en: string; zh: string },
  locale: string,
): string {
  return locale === "zh" ? value.zh : value.en;
}

export function ExampleGallery({
  onClose,
  onSelect,
  open,
}: {
  onClose: () => void;
  onSelect: (example: SampleExample) => void;
  open: boolean;
}) {
  const t = useTranslations("home");
  const locale = useLocale();
  const samplesQuery = useQuery({
    enabled: open,
    queryKey: ["sample-catalog"],
    queryFn: ({ signal }) => labvizApi.listSamples(signal),
    staleTime: 5 * 60_000,
  });

  return (
    <Dialog fullWidth maxWidth="md" onClose={onClose} open={open}>
      <DialogTitle component="div" sx={{ pr: 7 }}>
        <Typography component="h2" sx={{ fontWeight: 750 }} variant="h3">
          {t("sampleChooserTitle")}
        </Typography>
        <Typography color="text.secondary" sx={{ mt: 0.5 }} variant="body2">
          {t("sampleChooserDescription")}
        </Typography>
        <IconButton
          aria-label={t("closeSampleChooser")}
          onClick={onClose}
          sx={{ position: "absolute", right: 12, top: 12 }}
        >
          <CloseRoundedIcon />
        </IconButton>
      </DialogTitle>
      <Divider />
      <DialogContent dividers>
        <Alert severity="info" sx={{ mb: 2 }}>
          {t("sampleSyntheticNotice")}
        </Alert>
        {samplesQuery.isPending ? (
          <Stack spacing={2}>
            {Array.from({ length: 3 }, (_, index) => (
              <Skeleton height={150} key={index} variant="rounded" />
            ))}
          </Stack>
        ) : samplesQuery.isError ? (
          <Alert severity="error">{t("sampleCatalogError")}</Alert>
        ) : (
          <Stack spacing={2}>
            {samplesQuery.data.examples.map((example) => (
              <Paper
                component="article"
                data-example-slug={example.slug}
                key={example.slug}
                sx={{ border: 1, borderColor: "divider", p: 2.5 }}
              >
                <Stack spacing={1.5}>
                  <Stack
                    direction={{ xs: "column", sm: "row" }}
                    spacing={1}
                    sx={{
                      alignItems: { xs: "flex-start", sm: "center" },
                      justifyContent: "space-between",
                    }}
                  >
                    <Box>
                      <Typography component="h3" sx={{ fontWeight: 750 }} variant="h3">
                        {localized(example.title, locale)}
                      </Typography>
                      <Typography color="text.secondary" sx={{ mt: 0.35 }} variant="caption">
                        {example.filename} · {example.rowCount} {t("sampleRows")}
                      </Typography>
                    </Box>
                    <Stack direction="row" sx={{ flexWrap: "wrap", gap: 0.75 }}>
                      <Chip label={example.format.toUpperCase()} size="small" variant="outlined" />
                      <Chip
                        color="secondary"
                        label={t(`sampleDifficulty.${example.difficulty}`)}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        color="primary"
                        label={t(`sampleChartTypes.${example.recommendedChart}`)}
                        size="small"
                      />
                    </Stack>
                  </Stack>
                  <Typography variant="body2">{localized(example.purpose, locale)}</Typography>
                  <Typography color="text.secondary" variant="body2">
                    {t("sampleFields")}: {example.fields.map((field) => field.name).join(", ")}
                  </Typography>
                  <Typography color="text.secondary" variant="body2">
                    {t("sampleLearningGoal")}: {localized(example.learningGoal, locale)}
                  </Typography>
                  {example.qualityIssues.length > 0 ? (
                    <Typography color="warning.main" variant="body2">
                      {t("sampleQualityLesson")}: {example.qualityIssues.map((item) => localized(item, locale)).join(" ")}
                    </Typography>
                  ) : null}
                  <Button
                    onClick={() => onSelect(example)}
                    startIcon={<OpenInNewRoundedIcon />}
                    sx={{ alignSelf: { xs: "stretch", sm: "flex-start" } }}
                    variant="contained"
                  >
                    {t("openSample")}
                  </Button>
                </Stack>
              </Paper>
            ))}
          </Stack>
        )}
      </DialogContent>
    </Dialog>
  );
}
