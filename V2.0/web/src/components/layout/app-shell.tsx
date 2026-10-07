import { Box } from "@mui/material";
import type { ReactNode } from "react";

import { AppHeader } from "./app-header";

export function AppShell({
  children,
  landing = false,
}: {
  children: ReactNode;
  landing?: boolean;
}) {
  return (
    <Box sx={{ minHeight: "100vh" }}>
      <AppHeader landing={landing} />
      <Box component="main">{children}</Box>
    </Box>
  );
}
