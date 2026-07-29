"use client";

import DownloadRoundedIcon from "@mui/icons-material/DownloadRounded";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import ShareOutlinedIcon from "@mui/icons-material/ShareOutlined";
import {
  Box,
  Button,
  Chip,
  Container,
  Divider,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { ChartCanvas } from "@/components/common/chart-canvas";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

export function SharedChartScreen({ token }: { token: string }) {
  const t = useTranslations("shared");
  const sharedQuery = useQuery({
    queryKey: ["shared-chart", token],
    queryFn: ({ signal }) => labvizApi.getSharedChart(token, signal),
  });

  if (sharedQuery.isPending) {
    return (
      <Container maxWidth="lg" sx={{ py: { xs: 3, md: 6 } }}>
        <ApiStatePanel
          description={t("loadingDescription")}
          kind="loading"
          title={t("loadingTitle")}
        />
      </Container>
    );
  }

  if (sharedQuery.error) {
    const expired =
      sharedQuery.error instanceof LabVizApiError &&
      (sharedQuery.error.status === 410 ||
        sharedQuery.error.code === "share-expired");
    return (
      <Container maxWidth="md" sx={{ py: { xs: 3, md: 8 } }}>
        <ApiStatePanel
          actionLabel={expired ? undefined : t("retry")}
          description={
            expired
              ? t("expiredDescription")
              : sharedQuery.error instanceof LabVizApiError
                ? sharedQuery.error.message
                : t("loadFailed")
          }
          kind={expired ? "expired" : "error"}
          onAction={
            expired ? undefined : () => void sharedQuery.refetch()
          }
          title={expired ? t("expiredTitle") : t("unavailableTitle")}
        />
      </Container>
    );
  }

  const shared = sharedQuery.data;
  const downloads = Object.entries(shared.downloads).filter(
    (entry): entry is [string, string] => Boolean(entry[1]),
  );
  const downloadBaseName =
    shared.title
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "") || "labviz-chart";

  return (
    <Box sx={{ bgcolor: "background.default", minHeight: "calc(100vh - 64px)" }}>
      <Container maxWidth="xl" sx={{ py: { xs: 2, md: 4 } }}>
        <Stack spacing={2.5}>
          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={1.5}
            sx={{ alignItems: { sm: "center" }, justifyContent: "space-between" }}
          >
            <Box>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <Chip color="secondary" label={t("readOnly")} size="small" />
                <Typography color="text.secondary" variant="caption">
                  {t("updated")}{" "}
                  {new Intl.DateTimeFormat(undefined, {
                    dateStyle: "medium",
                  }).format(new Date(shared.updatedAt))}
                </Typography>
              </Stack>
              <Typography
                component="h1"
                sx={{ fontSize: { xs: 24, md: 30 }, fontWeight: 750, mt: 1 }}
              >
                {shared.title}
              </Typography>
            </Box>
            <Button
              onClick={() =>
                void navigator.clipboard?.writeText(window.location.href)
              }
              startIcon={<ShareOutlinedIcon />}
              variant="outlined"
            >
              {t("copyLink")}
            </Button>
          </Stack>

          <Box
            sx={{
              alignItems: "start",
              display: "grid",
              gap: 2,
              gridTemplateColumns: {
                xs: "1fr",
                lg: "minmax(0, 1fr) 280px",
              },
            }}
          >
            <Paper sx={{ border: 1, borderColor: "divider", p: { xs: 1, md: 3 } }}>
              <ChartCanvas
                analysis={shared.analysis}
                preview={shared.preview}
                spec={shared.chart}
              />
            </Paper>
            <Paper sx={{ border: 1, borderColor: "divider", p: 2.5 }}>
              <Stack spacing={2}>
                <Box>
                  <Typography sx={{ fontWeight: 750 }}>{t("description")}</Typography>
                  <Typography color="text.secondary" sx={{ mt: 1 }} variant="body2">
                    {shared.description || t("noDescription")}
                  </Typography>
                </Box>
                <Divider />
                <Box>
                  <Typography sx={{ fontWeight: 750 }}>{t("downloads")}</Typography>
                  {downloads.length === 0 ? (
                    <Stack
                      direction="row"
                      spacing={1}
                      sx={{ alignItems: "center", mt: 1 }}
                    >
                      <LockOutlinedIcon color="action" fontSize="small" />
                      <Typography color="text.secondary" variant="body2">
                        {t("disabled")}
                      </Typography>
                    </Stack>
                  ) : (
                    <Stack spacing={1} sx={{ mt: 1.5 }}>
                      {downloads.map(([format, href]) => (
                        <Button
                          component="a"
                          download={`${downloadBaseName}.${format}`}
                          href={href}
                          key={format}
                          startIcon={<DownloadRoundedIcon />}
                          variant="outlined"
                        >
                          {t("download", { format: format.toUpperCase() })}
                        </Button>
                      ))}
                    </Stack>
                  )}
                </Box>
                <Divider />
                <Typography color="text.secondary" variant="caption">
                  {t("sourceNeverShared")}
                </Typography>
              </Stack>
            </Paper>
          </Box>
        </Stack>
      </Container>
    </Box>
  );
}
