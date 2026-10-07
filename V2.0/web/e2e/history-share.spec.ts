import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("shows saved history and its filtered empty state", async ({ page }) => {
  const observations = await installMockApi(page, { history: "saved" });

  await page.goto("/history");
  await expect(
    page.getByRole("heading", { name: "Project history", exact: true }),
  ).toBeVisible();
  await expect.poll(() => observations.requests).toContain("GET /projects");
  await expect(page.getByText("Thermal response")).toBeVisible();
  await expect(page.getByText("Dose response study")).toBeVisible();
  await expect(page.getByText("Acquisition: Acquisition 2")).toBeVisible();
  await expect(page.getByText("Replicate R2")).toBeVisible();
  await expect(page.getByText("Batch B-2026-09")).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Continue editing" }),
  ).toHaveAttribute("href", "/workspace/project-e2e");

  await page.getByPlaceholder("Search projects").fill("R2");
  await expect(page.getByText("Thermal response")).toBeVisible();
  await page.getByPlaceholder("Search projects").fill("does not exist");
  await expect(
    page.getByText("No saved project matches the selected filters."),
  ).toBeVisible();
});

test("filters and deletes a saved project with explicit confirmation", async ({
  page,
}) => {
  const observations = await installMockApi(page, { history: "saved" });
  await page.goto("/history");

  await page.getByLabel("Chart type").click();
  await page.getByRole("option", { name: "line" }).click();
  await expect(page.getByText("Thermal response")).toBeVisible();

  await page.getByRole("button", { name: "Actions for Thermal response" }).click();
  await expect(page.getByRole("menuitem", { name: "Open export" })).toHaveAttribute(
    "href",
    "/workspace/project-e2e?step=export",
  );
  await expect(page.getByRole("menuitem", { name: "Open save and share" })).toHaveCount(0);
  await page.getByRole("menuitem", { name: "Delete project" }).click();
  await expect(
    page.getByRole("heading", { name: "Delete this saved project?" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Delete", exact: true }).click();

  await expect(page.getByText("No saved projects yet")).toBeVisible();
  expect(observations.deletedProjects).toEqual(["project-e2e"]);
});

test("views historical processed data and figure snapshots, downloads and deletes a figure", async ({
  page,
}) => {
  const observations = await installMockApi(page, { history: "saved" });
  await page.goto("/history");
  await page.getByRole("button", { name: "Data & figures" }).click();

  await expect(page.getByRole("heading", { name: "Processed data preview" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Download cleaned CSV" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Saved figure snapshots" })).toBeVisible();
  await expect(page.getByText("Thermal response figure", { exact: true })).toBeVisible();
  await expect(page.getByText("Thermal response paper figure", { exact: true })).toBeVisible();

  const dataDownload = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download cleaned CSV" }).click();
  expect((await dataDownload).suggestedFilename()).toMatch(/\.csv$/i);

  await page.getByRole("button", { name: "Open full-size figure: Thermal response figure" }).click();
  await expect(page.getByRole("dialog").last().getByRole("heading", { name: "Thermal response figure" })).toBeVisible();
  await page.getByRole("dialog").last().getByRole("button", { name: "Close" }).click();

  const figureDownload = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download", exact: true }).first().click();
  expect((await figureDownload).suggestedFilename()).toMatch(/\.png$/i);

  await page.getByRole("button", { name: "Delete", exact: true }).first().click();
  await expect(page.getByRole("heading", { name: "Delete this saved figure?" })).toBeVisible();
  await page.getByRole("dialog").last().getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByText("Thermal response figure", { exact: true })).toHaveCount(0);
  expect(observations.deletedFigureSnapshots).toEqual(["figure-png-e2e"]);
});

test("shows a recoverable error when duplicating local history fails", async ({ page, expectConsoleError }) => {
  expectConsoleError(/Failed to load resource:.*status of 503/);
  await installMockApi(page, { history: "saved" });
  await page.route("**/api/v1/projects/*/duplicate", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ code: "database-unavailable", message: "Local data is unavailable." }),
    });
  });
  await page.goto("/history");
  await page.getByRole("button", { name: "Actions for Thermal response" }).click();
  await page.getByRole("menuitem", { name: "Duplicate project" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Local data is unavailable." })).toBeVisible();
  await expect(page.getByText("Thermal response")).toBeVisible();
});

test("renders a read-only shared chart and creator-enabled download", async ({
  page,
}) => {
  await installMockApi(page);

  await page.goto("/share/share-e2e");
  await expect(
    page.getByRole("heading", { name: "Thermal response" }),
  ).toBeVisible();
  await expect(page.getByText("Read-only", { exact: true })).toBeVisible();

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download PNG" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.png$/i);
  expect(await download.failure()).toBeNull();
});

test("explains an expired share without exposing project data", async ({
  page,
  expectConsoleError,
}) => {
  expectConsoleError(/Failed to load resource:.*status of 410 \(Gone\)/);
  expectConsoleError(/Failed to load resource:.*status of 410 \(Gone\)/);

  await installMockApi(page, { shareExpired: true });

  await page.goto("/share/share-e2e");
  await expect(
    page.getByRole("heading", {
      name: "This share link has expired",
    }),
  ).toBeVisible();
  await expect(page.getByText(/No original experiment file was exposed/i)).toBeVisible();
  await expect(page.getByText("Thermal response")).toHaveCount(0);
});
