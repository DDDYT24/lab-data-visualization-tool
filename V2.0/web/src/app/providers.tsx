"use client";

import { AppRouterCacheProvider } from "@mui/material-nextjs/v16-appRouter";
import { CssBaseline, ThemeProvider } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NextIntlClientProvider } from "next-intl";
import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react";

import { InterfaceModeContext } from "@/components/layout/interface-mode";
import { interfaceModeEvent, interfaceModeKey, loadInterfaceMode, parseInterfaceMode, saveInterfaceMode, type InterfaceMode } from "@/features/settings/interface-mode";

import { loadUserPreferences } from "@/features/settings/user-preferences";
import { createLabvizTheme } from "@/theme/theme";

type AppProvidersProps = {
  children: ReactNode;
  locale: string;
  messages: Record<string, unknown>;
};

export function AppProviders({ children, locale, messages }: AppProvidersProps) {
  const [resolvedMode, setResolvedMode] = useState<"light" | "dark">("light");
  const [interfaceMode, setInterfaceMode] = useState<InterfaceMode>("classic");
  const [interfaceReady, setInterfaceReady] = useState(false);
  const [interfacePersistent, setInterfacePersistent] = useState(true);
  const changeInterface = useCallback((next: InterfaceMode) => {
    setInterfaceMode(next);
    document.documentElement.dataset.labvizInterface = next;
    setInterfacePersistent(saveInterfaceMode(next));
    const url = new URL(window.location.href);
    if (url.searchParams.has("ui")) {
      url.searchParams.set("ui", next);
      window.history.replaceState(window.history.state, "", url);
    }
  }, []);
  const interfaceContext = useMemo(() => ({ mode: interfaceMode, setMode: changeInterface, persistent: interfacePersistent, ready: interfaceReady }), [interfaceMode, changeInterface, interfacePersistent, interfaceReady]);
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
  const appTheme = useMemo(() => createLabvizTheme(resolvedMode, interfaceMode), [resolvedMode, interfaceMode]);

  useEffect(() => {
    const apply = (next: InterfaceMode) => {
      setInterfaceMode(next);
      document.documentElement.dataset.labvizInterface = next;
    };
    const timer = window.setTimeout(() => {
      const explicit = parseInterfaceMode(new URL(window.location.href).searchParams.get("ui"));
      const next = explicit ?? loadInterfaceMode();
      apply(next);
      setInterfaceReady(true);
      if (explicit) setInterfacePersistent(saveInterfaceMode(explicit));
    }, 0);
    const sync = () => apply(loadInterfaceMode());
    const storage = (event: StorageEvent) => { if (event.key === interfaceModeKey) sync(); };
    window.addEventListener(interfaceModeEvent, sync);
    window.addEventListener("storage", storage);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener(interfaceModeEvent, sync);
      window.removeEventListener("storage", storage);
    };
  }, []);

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
        <InterfaceModeContext.Provider value={interfaceContext}>
        <ThemeProvider theme={appTheme}>
          <CssBaseline />
          <QueryClientProvider client={queryClient}>
            {children}
          </QueryClientProvider>
        </ThemeProvider>
        </InterfaceModeContext.Provider>
      </NextIntlClientProvider>
    </AppRouterCacheProvider>
  );
}
