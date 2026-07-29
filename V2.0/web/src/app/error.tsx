"use client";

import { Container } from "@mui/material";
import { useEffect } from "react";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { AppShell } from "@/components/layout/app-shell";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <AppShell>
      <Container maxWidth="md" sx={{ py: 8 }}>
        <ApiStatePanel
          actionLabel="Try again"
          description="The current view could not be completed. Any uploaded source file remains unchanged."
          kind="error"
          onAction={reset}
          title="Something went wrong"
        />
      </Container>
    </AppShell>
  );
}
