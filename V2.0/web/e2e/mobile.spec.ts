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
