import { expect, test } from "./fixtures";
import { readFile } from "node:fs/promises";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Run with a real local-mode API and test key.");

test("real local history is account-free and hides cloud actions after launcher bootstrap", async ({ page, browser }) => {
  const key = process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY!;
  const unlocked = page.waitForResponse((response) =>
    response.url().endsWith("/api/v1/local/session") &&
    response.request().method() === "POST",
  );
  await page.goto(`/#labviz-access=${key}`);
  expect((await unlocked).ok()).toBeTruthy();
  await page.locator('input[type="file"]').setInputFiles({
    name: "local-history-e2e.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("time,response\n0,1\n1,2\n2,4\n"),
  });
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({ timeout: 60_000 });

  await page.getByRole("button", { name: "Review data quality" }).click();
  await expect(page.getByRole("heading", { name: "Review data quality" })).toBeVisible();
  const keepActions = page.getByRole("radio", { name: "Keep as is" });
  for (let index = 0; index < (await keepActions.count()); index += 1) {
    await keepActions.nth(index).check();
  }
  await page.getByRole("button", { name: "Create chart" }).click();
  await expect(page.getByRole("heading", { name: "Create your chart" })).toBeVisible();
  await page.getByRole("button", { name: "Customize export" }).click();
  await expect(page.getByRole("heading", { name: "Prepare publication export" })).toBeVisible();

  await page.getByRole("button", { name: "Prepare export" }).click();
  const initialPngLink = page.getByRole("link", { name: "Download PNG", exact: true });
  await expect(initialPngLink).toBeVisible({ timeout: 60_000 });
  const initialPngPromise = page.waitForEvent("download");
  await initialPngLink.click();
  const initialPng = await initialPngPromise;
  expect(await initialPng.failure()).toBeNull();
  expect((await readFile((await initialPng.path())!)).subarray(0, 8).toString("hex"))
    .toBe("89504e470d0a1a0a");

  await page.goto("/history");
  await expect(page.getByText("local-history-e2e.csv").first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Sign in" })).toHaveCount(0);

  // This project is newest, so choose its action if a repeated test run left older fixtures behind.
  await page.getByRole("button", { name: "Data & figures" }).first().click();
  await expect(page.getByRole("heading", { name: "Processed data preview", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Saved figure snapshots", exact: true })).toBeVisible();
  const fullFigureButton = page.getByRole("button", { name: /Open full-size figure/ });
  await expect(fullFigureButton).toHaveCount(1);
  await fullFigureButton.click();
  await expect(page.getByRole("dialog").last().locator("img")).toBeVisible();
  await page.getByRole("dialog").last().getByRole("button", { name: "Close" }).click();

  const cleanedDataPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download cleaned CSV" }).click();
  const cleanedData = await cleanedDataPromise;
  expect(await cleanedData.failure()).toBeNull();
  expect(await readFile((await cleanedData.path())!, "utf8")).toContain("time,response");

  const galleryPngPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download", exact: true }).click();
  const galleryPng = await galleryPngPromise;
  expect(await galleryPng.failure()).toBeNull();
  expect((await readFile((await galleryPng.path())!)).subarray(0, 8).toString("hex"))
    .toBe("89504e470d0a1a0a");

  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Delete this saved figure?" })).toBeVisible();
  await page.getByRole("dialog").last().getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByRole("heading", { name: "No saved figures yet" })).toBeVisible();
  await page.getByRole("dialog").last().getByRole("button", { name: "Close" }).click();

  const workspacePath = await page.getByRole("link", { name: "Continue editing" }).first().getAttribute("href");
  expect(workspacePath).toMatch(/^\/workspace\//);
  await page.goto(`${workspacePath}?step=export`);
  await expect(page.getByRole("button", { name: "Prepare export" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("button", { name: "Sign in" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Create share link" })).toHaveCount(0);
  await expect(page.getByText("Save and share", { exact: true })).toHaveCount(0);

  const other = await browser.newContext();
  try {
    const otherPage = await other.newPage();
    await otherPage.goto("/history");
    await expect(otherPage.getByText("local-history-e2e.csv")).toHaveCount(0);
    const otherUnlocked = otherPage.waitForResponse((response) =>
      response.url().endsWith("/api/v1/local/session") &&
      response.request().method() === "POST",
    );
    await otherPage.goto(`/#labviz-access=${key}`);
    expect((await otherUnlocked).ok()).toBeTruthy();
    await otherPage.goto("/history");
    await expect(otherPage.getByText("local-history-e2e.csv").first()).toBeVisible();
  } finally {
    await other.close();
  }
});
