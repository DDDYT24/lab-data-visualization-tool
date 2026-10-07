import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("explains the local processing path and supports offline feedback", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/help");

  await expect(page.getByRole("heading", { name: "How an upload becomes a chart" })).toBeVisible();
  await expect(page.getByText("FastAPI receives it")).toBeVisible();
  await expect(
    page.getByText(/PNG, SVG, and PDF are generated independently by the Python backend/),
  ).toBeVisible();
  await expect(page.getByText("What is a prediction interval?", { exact: true })).toBeVisible();
  await expect(page.getByText("What is a simultaneous confidence band?", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Local feedback center" })).toBeVisible();

  await page.getByLabel("What happened or what would help?").fill("The empty state could explain the next step.");
  await page.getByRole("button", { name: "Copy diagnostic" }).click();
  await expect(page.getByText("Diagnostic copied to the clipboard.")).toBeVisible();
  const diagnostic = await page.getByLabel("Diagnostic preview").textContent();
  expect(diagnostic).toContain('"screen": "help"');
  expect(diagnostic).not.toContain("project-e2e");
  expect(diagnostic).not.toContain("filename");
});

test("lets users dismiss and recover a local context tip", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/help");

  const tip = page.getByText(/Feedback is prepared locally/);
  await expect(tip).toBeVisible();
  await page.getByRole("button", { name: "Dismiss tip" }).click();
  await expect(tip).toHaveCount(0);
  await page.getByRole("button", { name: "Show dismissed tips again" }).click();
  await expect(tip).toBeVisible();
});

test("saves feedback locally and requires consent before preparing an external issue", async ({ page }) => {
  await installMockApi(page);
  await page.addInitScript(() => {
    window.confirm = () => document.documentElement.dataset.allowExternalIssue === "true";
    window.open = (url) => {
      document.documentElement.dataset.externalIssueUrl = String(url);
      return null;
    };
  });
  await page.goto("/help");
  await page.getByLabel("What happened or what would help?").fill("Improve the empty-state guidance.");

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save diagnostic" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("labviz-feedback-diagnostic.json");
  expect(await download.failure()).toBeNull();
  await expect(page.getByText("Diagnostic saved as a local text file.")).toBeVisible();

  await page.getByRole("button", { name: "Open GitHub issue" }).click();
  expect(await page.locator("html").getAttribute("data-external-issue-url")).toBeNull();
  await page.evaluate(() => {
    document.documentElement.dataset.allowExternalIssue = "true";
  });
  await page.getByRole("button", { name: "Open GitHub issue" }).click();

  const openedUrl = await page.locator("html").getAttribute("data-external-issue-url");
  expect(openedUrl).not.toBeNull();
  const issueUrl = new URL(openedUrl!);
  expect(issueUrl.origin).toBe("https://github.com");
  expect(issueUrl.pathname).toBe("/DDDYT24/lab-data-visualization-tool/issues/new");
  expect(issueUrl.searchParams.get("body")).toContain("Improve the empty-state guidance.");
});
