import { expect, test } from "./fixtures";
import path from "node:path";
import manifest from "../../api/samples/v22/manifest.json";

import { createCsvNearSize } from "./support/mock-api";

test.skip(
  process.env.LABVIZ_E2E_LIVE !== "1",
  "Set LABVIZ_E2E_LIVE=1 and run FastAPI on the configured proxy target.",
);

test.beforeEach(async ({ page }) => {
  const localKey = process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY;
  if (!localKey) return;
  const unlocked = page.waitForResponse((response) =>
    response.url().endsWith("/api/v1/local/session") &&
    response.request().method() === "POST",
  );
  await page.goto(`/#labviz-access=${encodeURIComponent(localKey)}`);
  expect((await unlocked).ok()).toBeTruthy();
});

test("processes and exports a generated 1.87 MB CSV through the live FastAPI service", async ({
  page,
}) => {
  test.setTimeout(240_000);
  const csv = createCsvNearSize();

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    buffer: csv,
    mimeType: "text/csv",
    name: "live-1.87mb.csv",
  });

  await expect(page).toHaveURL(/\/workspace\/(?:new|[a-z0-9]+)$/);
  await expect(
    page.getByRole("heading", { name: "Confirm your data" }),
  ).toBeVisible({ timeout: 90_000 });

  await page.getByRole("button", { name: "Review data quality" }).click();
  await expect(
    page.getByRole("heading", { name: "Review data quality" }),
  ).toBeVisible();

  const keepActions = page.getByRole("radio", { name: "Keep as is" });
  for (let index = 0; index < (await keepActions.count()); index += 1) {
    await keepActions.nth(index).check();
  }

  await page.getByRole("button", { name: "Create chart" }).click();
  await expect(
    page.getByRole("heading", { name: "Create your chart" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Customize export" }).click();

  await expect(
    page.getByRole("heading", { name: "Prepare publication export" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Prepare export" }).click();
  const downloadLink = page.getByRole("link", { name: "Download PNG" });
  await expect(downloadLink).toBeVisible({ timeout: 90_000 });

  const downloadPromise = page.waitForEvent("download");
  await downloadLink.click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.png$/i);
  expect(await download.failure()).toBeNull();
});

test("processes the configured real workbook through the live FastAPI service", async ({
  page,
}) => {
  const workbookPath = process.env.LABVIZ_E2E_FILE ?? path.resolve(
    "../api/samples/v22", manifest.examples.find((entry) => entry.format === "xlsx")!.filename,
  );
  test.setTimeout(240_000);

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(workbookPath!);

  await expect(page).toHaveURL(/\/workspace\/(?:new|[a-z0-9]+)$/);
  await expect(
    page.getByRole("heading", { name: "Confirm your data" }),
  ).toBeVisible({ timeout: 120_000 });
  await expect(page.getByRole("button", { name: "Review data quality" })).toBeEnabled();
});
