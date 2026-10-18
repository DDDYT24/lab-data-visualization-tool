import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { researchTaskFromQuery } from "@/features/workspace/research-navigation";
import { WorkspaceScreen } from "@/features/workspace/workspace-screen";

export const metadata: Metadata = {
  title: "Workspace",
};

export default async function SavedWorkspacePage({
  params,
  searchParams,
}: {
  params: Promise<{ projectId: string }>;
  searchParams: Promise<{ step?: string; workspace?: string; researchTask?: string }>;
}) {
  const { projectId } = await params;
  const { step, workspace, researchTask } = await searchParams;
  const initialStep =
    step === "import" || step === "inspect" || step === "chart" || step === "export"
      ? step
      : undefined;

  return (
    <AppShell>
      <WorkspaceScreen initialProjectId={projectId} initialStep={initialStep}
        initialResearchMode={workspace === "research"} initialResearchTask={researchTaskFromQuery(researchTask)} />
    </AppShell>
  );
}
