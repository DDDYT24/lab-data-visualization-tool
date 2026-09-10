import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";

import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

const approvedBreakpoints = {
  desktop: { width: 1_440, height: 1_024 },
  tablet: { width: 1_024, height: 1_024 },
  mobile: { width: 390, height: 844 },
} as const;

async function settleVisualPage(page: Page) {
  await page.addStyleTag({
    content: `
      *, *::before, *::after {
        animation-delay: 0s !important;
        animation-duration: 0s !important;
        caret-color: transparent !important;
        transition-delay: 0s !important;
        transition-duration: 0s !important;
      }
      nextjs-portal { display: none !important; }
    `,
  });
  await expect(page.locator("main")).toBeVisible();
  await expect.poll(async () => page.evaluate(() => document.fonts.status)).toBe("loaded");
}

async function expectRouteReady(page: Page, route: string) {
  const heading = route === "/"
    ? "Turn experiment data into a clear figure."
    : route.includes("step=import")
      ? "Confirm your data"
      : route.includes("step=inspect")
        ? "Review data quality"
        : route.includes("step=chart")
          ? "Create your chart"
          : route.includes("step=export")
            ? "Prepare publication export"
            : route === "/history"
              ? "Project history"
              : route === "/settings"
                ? "Settings"
                : route === "/help"
                  ? "Help and scientific guidance"
                  : "Thermal response";
  await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
  if (
    route.includes("step=chart") ||
    route.includes("step=export") ||
    route.startsWith("/share/")
  ) {
    await expect(page.locator('main [role="img"]')).toBeVisible();
    await expect(
      page.locator('main .echarts-for-react[_echarts_instance_]'),
    ).toBeVisible();
  }
}

async function expectNoHorizontalPageOverflow(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

async function expectNoSeriousAccessibilityViolations(page: Page, route: string) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  const serious = results.violations.filter(({ impact }) =>
    impact === "serious" || impact === "critical",
  );
  expect(
    serious,
    `${route}\n${serious
      .map(
        ({ help, id, nodes }) =>
          `${id}: ${help}\n${nodes.map(({ target }) => `  ${target.join(" ")}`).join("\n")}`,
      )
      .join("\n\n")}`,
  ).toEqual([]);
}

test("matches the approved Home breakpoints", async ({ page }) => {
  await installMockApi(page);

  for (const [name, viewport] of Object.entries(approvedBreakpoints)) {
    await page.setViewportSize(viewport);
    await page.goto("/");
    await expectRouteReady(page, "/");
    await settleVisualPage(page);
    await expectNoHorizontalPageOverflow(page);
    await expect(page).toHaveScreenshot(`home-${name}.png`, {
      animations: "disabled",
      fullPage: true,
    });
  }
});

test("keeps the complete workspace readable on desktop", async ({ page }) => {
  await installMockApi(page);
  await page.setViewportSize(approvedBreakpoints.desktop);

  for (const step of ["import", "inspect", "chart", "export"] as const) {
    const route = `/workspace/project-e2e?step=${step}`;
    await page.goto(route);
    await expectRouteReady(page, route);
    await settleVisualPage(page);
    await expectNoHorizontalPageOverflow(page);
    await expect(page).toHaveScreenshot(`workspace-${step}-desktop.png`, {
      animations: "disabled",
    });
  }
});

test("keeps chart editing readable at the mobile breakpoint", async ({ page }) => {
  await installMockApi(page);
  await page.setViewportSize(approvedBreakpoints.mobile);
  const route = "/workspace/project-e2e?step=chart";
  await page.goto(route);
  await expectRouteReady(page, route);
  await settleVisualPage(page);

  await expect(page.getByRole("heading", { name: "Create your chart" })).toBeVisible();
  await expectNoHorizontalPageOverflow(page);
  const figureSubtitle = page.getByLabel("Figure subtitle");
  const subtitleControl = figureSubtitle.locator(
    "xpath=ancestor::*[contains(@class, 'MuiFormControl-root')]",
  );
  const [inputBox, controlBox, inputBoxSizing] = await Promise.all([
    figureSubtitle.boundingBox(),
    subtitleControl.boundingBox(),
    figureSubtitle.evaluate((input) => getComputedStyle(input).boxSizing),
  ]);
  expect(inputBoxSizing).toBe("content-box");
  expect(inputBox?.height).toBeGreaterThanOrEqual(38);
  expect(controlBox?.height).toBeGreaterThanOrEqual(38);
  const contentWidth = await page.getByRole("heading", { name: "Create your chart" }).evaluate(
    (heading) => heading.getBoundingClientRect().width,
  );
  const documentWidth = await page.evaluate(
    () => document.documentElement.clientWidth,
  );
  expect(contentWidth).toBeLessThanOrEqual(documentWidth - 16);
  await expect(page).toHaveScreenshot("workspace-chart-mobile.png", {
    animations: "disabled",
    fullPage: true,
  });
});

test("covers supporting pages and the mobile shared-chart flow", async ({ page }) => {
  await installMockApi(page, { history: "saved" });
  await page.setViewportSize(approvedBreakpoints.desktop);

  for (const route of ["history", "settings", "help"] as const) {
    await page.goto(`/${route}`);
    await expectRouteReady(page, `/${route}`);
    await settleVisualPage(page);
    await expectNoHorizontalPageOverflow(page);
    await expect(page).toHaveScreenshot(`${route}-desktop.png`, {
      animations: "disabled",
    });
  }

  await page.setViewportSize(approvedBreakpoints.mobile);
  await page.goto("/share/share-e2e");
  await expectRouteReady(page, "/share/share-e2e");
  await settleVisualPage(page);
  await expect(page.getByRole("heading", { name: "Thermal response" })).toBeVisible();
  await expectNoHorizontalPageOverflow(page);
  await expect(page).toHaveScreenshot("shared-chart-mobile.png", {
    animations: "disabled",
    fullPage: true,
  });
});

test("has no serious WCAG violations in principal product states", async ({ page }) => {
  await installMockApi(page, { history: "saved" });
  await page.setViewportSize(approvedBreakpoints.desktop);

  for (const route of [
    "/",
    "/workspace/project-e2e?step=import",
    "/workspace/project-e2e?step=inspect",
    "/workspace/project-e2e?step=chart",
    "/workspace/project-e2e?step=export",
    "/history",
    "/settings",
    "/help",
    "/share/share-e2e",
  ]) {
    await page.goto(route);
    await expectRouteReady(page, route);
    await settleVisualPage(page);
    await expectNoSeriousAccessibilityViolations(page, route);
  }
});
