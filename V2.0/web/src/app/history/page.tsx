import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { SupportShell } from "@/components/layout/support-shell";
import { HistoryScreen } from "@/features/history/history-screen";

export const metadata: Metadata = {
  title: "Project history",
};

export default function HistoryPage() {
  return (
    <AppShell>
      <SupportShell>
        <HistoryScreen />
      </SupportShell>
    </AppShell>
  );
}
