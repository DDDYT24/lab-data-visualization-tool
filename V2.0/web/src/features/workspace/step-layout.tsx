import { Box, Stack, Typography } from "@mui/material";
import type { ReactNode } from "react";

type StepLayoutProps = {
  title: string;
  canvas: ReactNode;
  inspector: ReactNode;
};

export function StepLayout({ title, canvas, inspector }: StepLayoutProps) {
  return (
    <Stack spacing={2.5}>
      <Typography
        component="h1"
        sx={{ fontSize: { xs: 23, md: 26 }, fontWeight: 750 }}
      >
        {title}
      </Typography>
      <Box
        sx={{
          alignItems: "start",
          display: "grid",
          gap: 2,
          gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 1fr) 320px" },
        }}
      >
        <Box sx={{ minWidth: 0 }}>{canvas}</Box>
        <Box sx={{ minWidth: 0 }}>{inspector}</Box>
      </Box>
    </Stack>
  );
}
