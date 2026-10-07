import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { SupportShell } from "@/components/layout/support-shell";
import { AboutScreen } from "@/features/about/about-screen";

export const metadata: Metadata = {
  title: "About",
};

export default function AboutPage() {
  return (
    <AppShell>
      <SupportShell>
        <AboutScreen />
      </SupportShell>
    </AppShell>
  );
}
