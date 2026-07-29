import { AppShell } from "@/components/layout/app-shell";
import { HomeScreen } from "@/features/home/home-screen";

export default function HomePage() {
  return (
    <AppShell>
      <HomeScreen />
    </AppShell>
  );
}
