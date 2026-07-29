import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { SharedChartScreen } from "@/features/share/shared-chart-screen";

export const metadata: Metadata = {
  title: "Shared chart",
};

export default async function SharedChartPage({
  params,
}: {
  params: Promise<{ token: string }>;
}) {
  const { token } = await params;

  return (
    <AppShell>
      <SharedChartScreen token={token} />
    </AppShell>
  );
}
