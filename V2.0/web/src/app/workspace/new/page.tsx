import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { WorkspaceScreen } from "@/features/workspace/workspace-screen";

export const metadata: Metadata = {
  title: "Workspace",
};

export default function NewWorkspacePage() {
  return (
    <AppShell>
      <WorkspaceScreen />
    </AppShell>
  );
}
