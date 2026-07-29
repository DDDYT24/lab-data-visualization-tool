import { expect, test } from "@playwright/test";

import { installMockApi } from "./support/mock-api";

test("shows saved history and its filtered empty state", async ({ page }) => {
  const observations = await installMockApi(page, { history: "saved" });

  await page.goto("/history");
  await expect(
    page.getByRole("heading", { name: "Project history", exact: true }),
  ).toBeVisible();
  await expect.poll(() => observations.requests).toContain("GET /projects");
  await expect(page.getByText("Thermal response")).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Continue editing" }),
  ).toHaveAttribute("href", "/workspace/project-e2e");

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
  await expect(
    page.getByRole("menuitem", { name: "Open save and share" }),
  ).toHaveAttribute("href", "/workspace/project-e2e?step=export");
  await page.getByRole("menuitem", { name: "Delete project" }).click();
  await expect(
    page.getByRole("heading", { name: "Delete this saved project?" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Delete", exact: true }).click();

  await expect(page.getByText("No saved projects yet")).toBeVisible();
  expect(observations.deletedProjects).toEqual(["project-e2e"]);
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

test("explains an expired share without exposing project data", async ({ page }) => {
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
