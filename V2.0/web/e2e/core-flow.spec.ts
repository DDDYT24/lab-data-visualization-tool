import { expect, test } from "./fixtures";

import { createCsvNearSize, installMockApi } from "./support/mock-api";

test("uploads a 1.87 MB CSV and completes inspect, chart, export, and download", async ({
  page,
}) => {
  const observations = await installMockApi(page, { dataDelayMs: 500 });
  const csv = createCsvNearSize();

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    buffer: csv,
    mimeType: "text/csv",
    name: "experiment-1.87mb.csv",
  });

  await expect(page).toHaveURL(/\/workspace\/project-e2e$/);
  await expect(
    page.getByRole("heading", { name: "Confirm your data" }),
  ).toBeVisible();
  expect(observations.uploadedBytes[0]).toBeGreaterThanOrEqual(csv.byteLength);

  await page.getByRole("button", { name: "Review data quality" }).click();
  await expect(
    page.getByRole("heading", { name: "Review data quality" }),
  ).toBeVisible();
  await expect(page.getByText("2 findings · 2 decisions pending")).toBeVisible();
  await expect(page.getByText("Finding 1 of 2")).toBeVisible();
  await page.getByRole("radio", { name: "Keep as is" }).check();
  await page.getByRole("button", { name: "Next quality finding" }).click();
  await expect(page.getByText("Finding 2 of 2")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Missing measurement" }),
  ).toBeVisible();
  await page.getByRole("radio", { name: "Exclude from chart" }).check();
  await expect.poll(() => observations.cleaningActions).toEqual([
    "ignore",
    "exclude",
  ]);

  await page.getByRole("button", { name: "Create chart" }).click();
  await expect(
    page.getByRole("heading", { name: "Create your chart" }),
  ).toBeVisible();
  await page.getByLabel("Figure title").fill("Publication figure");
  await page.getByRole("button", { name: "Fitting and uncertainty" }).click();
  await page.getByLabel("Fit model").click();
  await page.getByRole("option", { name: "Linear" }).click();
  await page.getByRole("switch", { name: "Show confidence band" }).check();
  await page.getByLabel("Confidence level").click();
  await page.getByRole("option", { name: "99%" }).click();
  await expect.poll(() => observations.analysisModels).toContain("linear");
  await page.getByRole("button", { name: "Customize export" }).click();

  await expect(
    page.getByRole("heading", { name: "Prepare publication export" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Prepare export" }).click();
  await expect(page.getByText("PNG export is ready.")).toBeVisible();
  await expect(page.getByRole("link", { name: "Download PNG" })).toBeVisible();

  await page.getByLabel("Format").click();
  await page.getByRole("option", { name: "SVG" }).click();
  await expect(page.getByRole("link", { name: "Download PNG" })).toHaveCount(0);
  await page.getByRole("button", { name: "Prepare export" }).click();
  await expect(page.getByText("SVG export is ready.")).toBeVisible();
  expect(observations.exportRequests).toBe(2);
  expect(observations.exportFormats).toEqual(["png", "svg"]);

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download SVG" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.svg$/i);
  expect(await download.failure()).toBeNull();
});

test("explains console OTP delivery and completes the verification-code dialog", async ({
  page,
}) => {
  const observations = await installMockApi(page, { deliveryMode: "console" });

  await page.goto("/");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByLabel("Email address").fill("researcher@example.com");
  await page.getByRole("button", { name: "Send verification code" }).click();

  await expect(page.getByLabel("Six-digit code")).toBeVisible();
  await expect(
    page.getByText(/This local development server is in console mode/i),
  ).toBeVisible();
  await page.getByLabel("Six-digit code").fill("123456");
  await page.getByRole("button", { name: "Verify and sign in" }).click();

  await expect(page.getByRole("heading", { name: "Signed in" })).toBeVisible();
  expect(observations.requestedEmail).toBe("researcher@example.com");
  expect(observations.verifiedCode).toBe("123456");
});

test("uses normal inbox copy when SMTP email delivery is enabled", async ({ page }) => {
  await installMockApi(page, { deliveryMode: "email" });

  await page.goto("/");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByLabel("Email address").fill("researcher@example.com");
  await page.getByRole("button", { name: "Send verification code" }).click();

  await expect(
    page.getByText("Enter the six-digit code sent to researcher@example.com."),
  ).toBeVisible();
  await expect(
    page.getByText(/This local development server is in console mode/i),
  ).toHaveCount(0);
});

test("switches the principal interface to Simplified Chinese", async ({ page }) => {
  await installMockApi(page);

  await page.goto("/help");
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文" }).click();

  await expect(
    page.getByRole("heading", { level: 1, name: "帮助与科研说明" }),
  ).toBeVisible();
  await expect(page.getByPlaceholder("搜索帮助")).toBeVisible();
});

test("renders an unknown route without browser runtime errors", async ({
  page,
  expectConsoleError,
}) => {
  expectConsoleError(/Failed to load resource:.*status of 404 \(Not Found\)/);
  await installMockApi(page);

  await page.goto("/this-route-does-not-exist");
  await expect(page.getByRole("heading", { name: "Page not found" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Return home" })).toHaveAttribute(
    "href",
    "/",
  );
});
