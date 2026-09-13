"use client";

import { AppRouterCacheProvider } from "@mui/material-nextjs/v16-appRouter";
import { CssBaseline, ThemeProvider } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NextIntlClientProvider } from "next-intl";
import { type ReactNode, useEffect, useMemo, useState } from "react";

import { loadUserPreferences } from "@/features/settings/user-preferences";
import { createLabvizTheme } from "@/theme/theme";

type AppProvidersProps = {
  children: ReactNode;
  locale: string;
  messages: Record<string, unknown>;
};

export function AppProviders({ children, locale, messages }: AppProvidersProps) {
  const [resolvedMode, setResolvedMode] = useState<"light" | "dark">("light");
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            retry: 1,
            staleTime: 30_000,
          },
        },
      }),
  );
  const appTheme = useMemo(() => createLabvizTheme(resolvedMode), [resolvedMode]);

  useEffect(() => {
    const syncAppearance = () => {
      const next = loadUserPreferences().appearance;
      const systemMode = window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
      const resolved = next === "system" ? systemMode : next;
      setResolvedMode(resolved);
      document.documentElement.dataset.labvizTheme = resolved;
    };
    const timer = window.setTimeout(syncAppearance, 0);
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handleSystemChange = () => {
      if (loadUserPreferences().appearance === "system") syncAppearance();
    };
    window.addEventListener("labviz:preferences-changed", syncAppearance);
    media.addEventListener("change", handleSystemChange);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("labviz:preferences-changed", syncAppearance);
      media.removeEventListener("change", handleSystemChange);
    };
  }, []);

  return (
    <AppRouterCacheProvider options={{ enableCssLayer: true }}>
      <NextIntlClientProvider locale={locale} messages={messages} timeZone="UTC">
        <ThemeProvider theme={appTheme}>
          <CssBaseline />
          <QueryClientProvider client={queryClient}>
            {children}
          </QueryClientProvider>
        </ThemeProvider>
      </NextIntlClientProvider>
    </AppRouterCacheProvider>
  );
}
