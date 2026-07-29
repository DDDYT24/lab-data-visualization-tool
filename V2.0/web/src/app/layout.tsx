import type { Metadata } from "next";
import { cookies } from "next/headers";
import type { ReactNode } from "react";

import enMessages from "@/messages/en.json";
import zhMessages from "@/messages/zh.json";

import "./globals.css";
import { AppProviders } from "./providers";

export const metadata: Metadata = {
  title: {
    default: "LabViz — Experiment data to publication figures",
    template: "%s · LabViz",
  },
  description:
    "A guided scientific data visualization workflow for experimenters who do not use Python.",
};

export default async function RootLayout({ children }: { children: ReactNode }) {
  const cookieStore = await cookies();
  const locale = cookieStore.get("labviz-locale")?.value === "zh" ? "zh" : "en";
  const messages = locale === "zh" ? zhMessages : enMessages;

  return (
    <html lang={locale}>
      <body>
        <AppProviders locale={locale} messages={messages}>
          {children}
        </AppProviders>
      </body>
    </html>
  );
}
