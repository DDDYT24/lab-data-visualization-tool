// @vitest-environment jsdom

import { createElement, type ReactNode } from "react";
import { act } from "react";
import { hydrateRoot, type Root } from "react-dom/client";
import { renderToString } from "react-dom/server";
import { afterEach, expect, it, vi } from "vitest";

import { createTheme, ThemeProvider } from "@mui/material/styles";

import { SettingsScreen } from "./settings-screen";
import { defaultUserPreferences } from "./user-preferences";

vi.mock("next-intl", () => ({
  useLocale: () => "en",
  useTranslations: () => (key: string) => key,
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh: vi.fn() }) }));

vi.mock("next/link", () => ({
  default: ({ href, children, ...props }: { href: string; children?: ReactNode }) =>
    createElement("a", { ...props, href }, children),
}));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

let mountedRoot: Root | null = null;
afterEach(async () => {
  if (mountedRoot) {
    await act(async () => mountedRoot?.unmount());
    mountedRoot = null;
  }
  window.localStorage.clear();
});

it("hydrates Settings from defaults before applying stored browser preferences", async () => {
  window.localStorage.clear();
  const theme = createTheme();
  const tree = createElement(
    ThemeProvider,
    { theme },
    createElement(SettingsScreen),
  );
  const serverMarkup = renderToString(tree);
  window.localStorage.setItem(
    "labviz:user-preferences:v1",
    JSON.stringify({
      ...defaultUserPreferences,
      appearance: "dark",
      figureLanguage: "zh",
      fontFamily: "Times New Roman",
      sizePreset: "custom",
      unit: "cm",
      dpi: 600,
      grayscalePreview: true,
    }),
  );

  const host = document.createElement("div");
  host.innerHTML = serverMarkup;
  const hydrationErrors: string[] = [];
  await act(async () => {
    mountedRoot = hydrateRoot(host, tree, {
      onRecoverableError: (error) =>
        hydrationErrors.push(error instanceof Error ? error.message : String(error)),
    });
  });
  await act(async () => new Promise((resolve) => window.setTimeout(resolve, 10)));

  expect(hydrationErrors).toEqual([]);
  expect(host.textContent).toContain("Times New Roman");
  expect(host.textContent).toContain("custom");
  expect(host.textContent).toContain("cm");
  expect(host.textContent).toContain("600");
});
