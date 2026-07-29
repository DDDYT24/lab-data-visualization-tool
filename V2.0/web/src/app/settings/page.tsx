import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { SupportShell } from "@/components/layout/support-shell";
import { SettingsScreen } from "@/features/settings/settings-screen";

export const metadata: Metadata = {
  title: "Settings",
};

export default function SettingsPage() {
  return (
    <AppShell>
      <SupportShell>
        <SettingsScreen />
      </SupportShell>
    </AppShell>
  );
}
