import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { SupportShell } from "@/components/layout/support-shell";
import { HelpScreen } from "@/features/help/help-screen";

export const metadata: Metadata = {
  title: "Help",
};

export default function HelpPage() {
  return (
    <AppShell>
      <SupportShell>
        <HelpScreen />
      </SupportShell>
    </AppShell>
  );
}
