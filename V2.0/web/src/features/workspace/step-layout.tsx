import { Box, Stack, Typography } from "@mui/material";
import type { ReactNode } from "react";

type StepLayoutProps = {
  title: string;
  description: string;
  canvas: ReactNode;
  inspector: ReactNode;
};

export function StepLayout({ title, description, canvas, inspector }: StepLayoutProps) {
  return (
    <Stack spacing={2.5}>
      <Box>
        <Typography
          component="h1"
          sx={{ fontSize: { xs: 23, md: 26 }, fontWeight: 750 }}
        >
          {title}
        </Typography>
        <Typography
          color="text.secondary"
          sx={{ maxWidth: 760, mt: 0.75 }}
          variant="body2"
        >
          {description}
        </Typography>
      </Box>
      <Box
        sx={{
          alignItems: "start",
          display: "grid",
          gap: 2,
          gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 1fr) 288px" },
        }}
      >
        <Box sx={{ minWidth: 0 }}>{canvas}</Box>
        <Box sx={{ minWidth: 0 }}>{inspector}</Box>
      </Box>
    </Stack>
  );
}
