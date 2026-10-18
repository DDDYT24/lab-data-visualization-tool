import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { WorkspaceScreen } from "@/features/workspace/workspace-screen";

export const metadata: Metadata = {
  title: "Workspace",
};

export default async function NewWorkspacePage({ searchParams }: { searchParams: Promise<{ workspace?: string }> }) {
  const { workspace } = await searchParams;
  return (
    <AppShell>
      <WorkspaceScreen initialResearchMode={workspace === "research"} />
    </AppShell>
  );
}
