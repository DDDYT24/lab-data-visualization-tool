"use client";

import { AppRouterCacheProvider } from "@mui/material-nextjs/v16-appRouter";
import { CssBaseline, ThemeProvider } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NextIntlClientProvider } from "next-intl";
import { type ReactNode, useState } from "react";

import { theme } from "@/theme/theme";

type AppProvidersProps = {
  children: ReactNode;
  locale: string;
  messages: Record<string, unknown>;
};

export function AppProviders({ children, locale, messages }: AppProvidersProps) {
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

  return (
    <AppRouterCacheProvider options={{ enableCssLayer: true }}>
      <NextIntlClientProvider locale={locale} messages={messages}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <QueryClientProvider client={queryClient}>
            {children}
          </QueryClientProvider>
        </ThemeProvider>
      </NextIntlClientProvider>
    </AppRouterCacheProvider>
  );
}
