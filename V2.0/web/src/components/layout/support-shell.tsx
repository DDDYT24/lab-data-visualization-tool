import { Box } from "@mui/material";
import type { ReactNode } from "react";

import { PrimarySidebar } from "./primary-sidebar";

export function SupportShell({ children }: { children: ReactNode }) {
  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: { xs: "1fr", md: "208px minmax(0, 1fr)" },
        minHeight: "calc(100vh - 64px)",
      }}
    >
      <Box sx={{ borderRight: { md: 1 }, borderColor: "divider" }}>
        <PrimarySidebar />
      </Box>
      <Box
        sx={{
          bgcolor: "background.default",
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
