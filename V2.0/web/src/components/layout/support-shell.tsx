"use client";

import { Box } from "@mui/material";
import type { ReactNode } from "react";

import { useInterfaceMode } from "./interface-mode";
import { PrimarySidebar } from "./primary-sidebar";

export function SupportShell({ children }: { children: ReactNode }) {
  const overview = useInterfaceMode().mode === "overview";
  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: overview ? "minmax(0, 1fr)" : { xs: "1fr", md: "208px minmax(0, 1fr)" },
        minHeight: "calc(100vh - 64px)",
      }}
    >
      <Box hidden={overview} sx={{ borderRight: { md: 1 }, borderColor: "divider" }}>
        <PrimarySidebar />
      </Box>
      <Box
        sx={{
          bgcolor: "background.default",
          width: "100%",
          maxWidth: overview ? 1440 : undefined,
          mx: "auto",
          minWidth: 0,
          px: { xs: 2, md: 4 },
          py: { xs: 3, md: 4 },
        }}
      >
        {children}
      </Box>
    </Box>
  );
}
