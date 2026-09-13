import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("shows the versioned example catalog and opens a selected example", async ({ page }) => {
  const observations = await installMockApi(page);

  await page.goto("/");
  await page.getByRole("button", { name: "Try sample data" }).click();

  const dialog = page.getByRole("dialog");
  await expect(
    dialog.getByRole("heading", { name: "Choose an example dataset", exact: true }),
  ).toBeVisible();
  await expect(dialog.locator("[data-example-slug]")).toHaveCount(7);
  const surface = dialog.locator('[data-example-slug="surface-3d"]');
  await expect(surface).toContainText("Regular X/Y/Z surface");
  await expect(surface).toContainText("3D surface");
  await expect(surface.getByText(/441 rows/)).toBeVisible();

  await surface.getByRole("button", { name: "Open this example" }).focus();
  await surface.getByRole("button", { name: "Open this example" }).press("Enter");
  await expect(page).toHaveURL(/\/workspace\/project-e2e$/);
  expect(observations.sampleSlugs).toEqual(["surface-3d"]);
});

test("keeps the example chooser bilingual", async ({ page }) => {
  await installMockApi(page);

  await page.goto("/");
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文" }).click();
  await page.getByRole("button", { name: "使用示例数据" }).click();

  const dialog = page.getByRole("dialog");
  await expect(
    dialog.getByRole("heading", { name: "选择示例数据", exact: true }),
  ).toBeVisible();
  await expect(dialog.locator('[data-example-slug="time-series"]')).toContainText("时间响应曲线");
  await expect(dialog.getByRole("button", { name: "打开此示例" }).first()).toBeVisible();
});
