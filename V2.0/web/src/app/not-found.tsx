import { Button, Container } from "@mui/material";
import Link from "next/link";

import { ApiStatePanel } from "@/components/common/api-state-panel";
import { AppShell } from "@/components/layout/app-shell";

export default function NotFound() {
  return (
    <AppShell>
      <Container maxWidth="md" sx={{ py: 8 }}>
        <ApiStatePanel
          description="The requested project, page, or shared link could not be found."
          kind="empty"
          secondaryAction={
            <Button component={Link} href="/" variant="contained">
              Return home
            </Button>
          }
          title="Page not found"
        />
      </Container>
    </AppShell>
  );
}
