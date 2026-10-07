import { expect, test } from "./fixtures";
import type { Page } from "@playwright/test";

import { installMockApi } from "./support/mock-api";

async function selectSurfaceFields(page: Page) {
  await page.getByRole("combobox", { name: "Chart type" }).click();
  await page.getByRole("option", { name: "3D surface" }).click();
  await page.getByRole("combobox", { name: "X field" }).click();
  await page.getByRole("option", { name: "X", exact: true }).click();
  await page.getByRole("combobox", { name: "Surface Y field" }).click();
  await page.getByRole("option", { name: "Y", exact: true }).click();
  await page.getByRole("combobox", { name: "Surface Z field" }).click();
  await page.getByRole("option", { name: "Z", exact: true }).click();
}

test("keeps the primary upload experience usable at a phone viewport", async ({
  page,
}) => {
  await installMockApi(page);
  await page.goto("/");

  await expect(page.locator("main")).toBeVisible();
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "Make experiment data clear and insightful.",
    }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Choose a file" })).toBeVisible();
  await expect(page.getByRole("button", { name: "View examples" })).toBeVisible();

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);

  const unnamedButtons = await page.getByRole("button").evaluateAll((buttons) =>
    buttons.filter((button) => !(button.textContent?.trim() || button.getAttribute("aria-label")))
      .length,
  );
  expect(unnamedButtons).toBe(0);
});

test("keeps the supporting history state readable on mobile", async ({ page }) => {
  await installMockApi(page, { history: "empty" });
  await page.goto("/history");

  await expect(
    page.getByRole("heading", { level: 1, name: "Project history" }),
  ).toBeVisible();
  await expect(page.getByText("No saved projects yet")).toBeVisible();
  await expect(page.getByRole("link", { name: "Start a new analysis" })).toBeVisible();

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

test("opens every bundled example without horizontal overflow on mobile", async ({ page }) => {
  const observations = await installMockApi(page);
  const slugs = [
    "time-series",
    "repeated-runs",
    "scatter-fit",
    "categorical-comparison",
    "distribution",
    "correlation-heatmap",
    "surface-3d",
  ];

  for (const slug of slugs) {
    await page.goto("/");
    await page.getByRole("button", { name: "View examples" }).tap();
    const dialog = page.getByRole("dialog");
    const overflow = await dialog.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
    await dialog
      .locator(`[data-example-slug="${slug}"]`)
      .getByRole("button", { name: "Open this example" })
      .tap();
    await expect(page).toHaveURL(/\/workspace\/project-e2e$/);
    await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  }

  expect(observations.sampleSlugs).toEqual(slugs);
});

test("keeps the regular surface controls usable on mobile", async ({ page }) => {
  const observations = await installMockApi(page, { dataset: "surface", lowPerformance: true });
  await page.goto("/workspace/project-e2e?step=chart");

  await selectSurfaceFields(page);

  const surface = page.locator('main [role="img"][data-surface-support="ready"]');
  await expect(surface).toBeVisible();
  await expect(surface).toHaveAttribute("data-surface-point-count", "441");
  await expect(surface).toHaveAttribute("data-surface-fallback", "low-cost");
  await expect(surface).toHaveAttribute("data-touch-action", "pan-y");
  await expect(page.locator("[data-surface-gesture-hint]")).toContainText(
    "one finger to rotate",
  );

  const controls = page.getByLabel("3D surface controls");
  await expect(controls.getByRole("button", { name: "Reset view" })).toBeVisible();
  const targetSizes = await controls.getByRole("button").evaluateAll((buttons) =>
    buttons.map((button) => {
      const bounds = button.getBoundingClientRect();
      return { height: bounds.height, width: bounds.width };
    }),
  );
  expect(targetSizes.every(({ height, width }) => height >= 44 && width >= 44)).toBe(true);
  await controls.getByRole("button", { name: "Rotate right" }).tap();
  await expect(surface).toHaveAttribute("data-surface-view", "25:60:200");

  const bounds = await surface.boundingBox();
  expect(bounds).not.toBeNull();
  const centerX = bounds!.x + bounds!.width / 2;
  const centerY = bounds!.y + bounds!.height / 2;
  const session = await page.context().newCDPSession(page);
  await session.send("Input.dispatchTouchEvent", {
    type: "touchStart",
    touchPoints: [
      { x: centerX - 20, y: centerY },
      { x: centerX + 20, y: centerY },
    ],
  });
  await session.send("Input.dispatchTouchEvent", {
    type: "touchMove",
    touchPoints: [
      { x: centerX - 35, y: centerY + 5 },
      { x: centerX + 35, y: centerY + 5 },
    ],
  });
  await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await expect(surface).toBeVisible();

  await page.setViewportSize({ height: 390, width: 844 });
  await expect(surface).toBeVisible();
  await expect.poll(() => observations.analysisCharts.findLast((chart) => chart.type === "surface3d"))
    .toMatchObject({
      type: "surface3d",
      xAxis: { field: "x" },
      series: [{ field: "y" }, { field: "z" }],
    });

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

for (const surfaceCase of ["duplicate", "missing"] as const) {
  test(`blocks the ${surfaceCase} surface with an actionable mobile diagnostic`, async ({ page }) => {
    await installMockApi(page, { dataset: "surface", surfaceCase });
    await page.goto("/workspace/project-e2e?step=chart");
    await selectSurfaceFields(page);

    const message =
      surfaceCase === "duplicate"
        ? /contain 2 duplicate row\(s\)/
        : /missing 1 grid cell\(s\)/;
    await expect(page.getByRole("alert").filter({ hasText: message })).toBeVisible();
    await expect(page.getByRole("button", { name: "Customize export" })).toBeDisabled();
  });
}

test("uses the disclosed low-cost fallback for a 101 x 101 mobile surface", async ({ page }) => {
  await installMockApi(page, {
    dataset: "surface",
    lowPerformance: true,
    surfaceCase: "large",
  });
  await page.goto("/workspace/project-e2e?step=chart");
  await selectSurfaceFields(page);

  const surface = page.locator('main [role="img"][data-surface-support="ready"]');
  await expect(surface).toBeVisible();
  const renderedPointCount = Number(await surface.getAttribute("data-surface-point-count"));
  expect(renderedPointCount).toBeGreaterThan(0);
  expect(renderedPointCount).toBeLessThanOrEqual(800);
  await expect(surface).toHaveAttribute("data-surface-fallback", "low-cost");
  await expect(page.getByText(/lower-cost view is active/i)).toBeVisible();
  await expect(
    page.getByText("Preview uses 200 evenly distributed rows from 10201 total rows."),
  ).toBeVisible();
});
