import CheckCircleOutlineRoundedIcon from "@mui/icons-material/CheckCircleOutlineRounded";
import CloudOffRoundedIcon from "@mui/icons-material/CloudOffRounded";
import HourglassEmptyRoundedIcon from "@mui/icons-material/HourglassEmptyRounded";
import InboxOutlinedIcon from "@mui/icons-material/InboxOutlined";
import {
  Box,
  Button,
  CircularProgress,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import type { ReactNode } from "react";

type ApiStatePanelProps = {
  kind: "loading" | "empty" | "error" | "expired" | "success";
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  secondaryAction?: ReactNode;
  compact?: boolean;
};

const stateIcons = {
  empty: InboxOutlinedIcon,
  error: CloudOffRoundedIcon,
  expired: HourglassEmptyRoundedIcon,
  loading: CircularProgress,
  success: CheckCircleOutlineRoundedIcon,
};

export function ApiStatePanel({
  kind,
  title,
  description,
  actionLabel,
  onAction,
  secondaryAction,
  compact = false,
}: ApiStatePanelProps) {
  const Icon = stateIcons[kind];
  const color =
    kind === "success"
      ? "success.main"
      : kind === "error" || kind === "expired"
        ? "warning.main"
        : "primary.main";

  return (
    <Paper
      sx={{
        border: 1,
        borderColor: "divider",
        minHeight: compact ? 220 : 360,
        px: { xs: 3, md: 6 },
        py: { xs: 5, md: 7 },
      }}
    >
      <Stack
        spacing={2}
        sx={{
          alignItems: "center",
          height: "100%",
          justifyContent: "center",
          textAlign: "center",
        }}
      >
        <Box
          sx={{
            alignItems: "center",
            bgcolor:
              kind === "success"
                ? "success.light"
                : kind === "error" || kind === "expired"
                  ? "warning.light"
                  : "primary.light",
            borderRadius: "50%",
            color,
            display: "flex",
            height: 52,
            justifyContent: "center",
            width: 52,
          }}
        >
          {kind === "loading" ? (
            <CircularProgress aria-label={title} size={24} />
          ) : (
            <Icon sx={{ fontSize: 26 }} />
          )}
        </Box>
        <Box>
          <Typography component="h2" variant="h3">
            {title}
          </Typography>
          <Typography
            color="text.secondary"
            sx={{ maxWidth: 520, mt: 1 }}
            variant="body2"
          >
            {description}
          </Typography>
        </Box>
        {actionLabel && onAction ? (
          <Button onClick={onAction} variant="contained">
            {actionLabel}
          </Button>
        ) : null}
        {secondaryAction}
      </Stack>
    </Paper>
  );
}
