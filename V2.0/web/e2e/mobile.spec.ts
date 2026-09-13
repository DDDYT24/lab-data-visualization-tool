import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("keeps the primary upload experience usable at a phone viewport", async ({
  page,
}) => {
  await installMockApi(page);
  await page.goto("/");

  await expect(page.locator("main")).toBeVisible();
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "Turn experiment data into a clear figure.",
    }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Choose a file" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Try sample data" })).toBeVisible();

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

test("keeps the regular surface controls usable on mobile", async ({ page }) => {
  await installMockApi(page, { dataset: "surface", lowPerformance: true });
  await page.goto("/workspace/project-e2e?step=chart");

  await page.getByRole("combobox", { name: "Chart type" }).click();
  await page.getByRole("option", { name: "3D surface" }).click();
  await page.getByRole("combobox", { name: "X field" }).click();
  await page.getByRole("option", { name: "X", exact: true }).click();
  await page.getByRole("combobox", { name: "Surface Y field" }).click();
  await page.getByRole("option", { name: "Y", exact: true }).click();
  await page.getByRole("combobox", { name: "Surface Z field" }).click();
  await page.getByRole("option", { name: "Z", exact: true }).click();

  const surface = page.locator('main [role="img"][data-surface-support="ready"]');
  await expect(surface).toBeVisible();
  await expect(surface).toHaveAttribute("data-surface-point-count", "441");
  await expect(surface).toHaveAttribute("data-surface-fallback", "low-cost");

  const controls = page.getByLabel("3D surface controls");
  await expect(controls.getByRole("button", { name: "Reset view" })).toBeVisible();
  await controls.getByRole("button", { name: "Rotate right" }).tap();
  await expect(surface).toHaveAttribute("data-surface-view", "25:60:100");

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});
