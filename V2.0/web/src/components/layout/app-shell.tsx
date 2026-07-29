import { Box } from "@mui/material";
import type { ReactNode } from "react";

import { AppHeader } from "./app-header";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <Box sx={{ minHeight: "100vh" }}>
      <AppHeader />
      <Box component="main">{children}</Box>
    </Box>
  );
}
