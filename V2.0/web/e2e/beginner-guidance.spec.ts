import AxeBuilder from "@axe-core/playwright";

import { expect, test } from "./fixtures";

import { createSurfaceCsv, installMockApi } from "./support/mock-api";

test("guides a first-time user from a surface CSV to export without developer terminology", async ({
  page,
}) => {
  await installMockApi(page, { dataset: "surface" });
  const csv = createSurfaceCsv();

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    buffer: csv,
    mimeType: "text/csv",
    name: "07_surface3d.csv",
  });
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  await page.getByRole("combobox", { name: "X field" }).click();
  await page.getByRole("option", { name: "X", exact: true }).click();
  await page.getByRole("combobox", { name: "Y field" }).click();
  await page.getByRole("option", { name: "Y", exact: true }).click();
  await page.getByRole("button", { name: "Review data quality" }).click();

  await expect(page.getByRole("heading", { name: "No quality issues found" })).toBeVisible();
  await page.getByRole("button", { name: "Create chart" }).click();
  const recommendation = page.getByLabel("Recommended chart: 3D surface");
  await expect(recommendation).toContainText("complete 21 × 21 grid");
  const useRecommendation = recommendation.getByRole("button", {
    name: "Use 3D surface",
  });
  await useRecommendation.focus();
  await page.keyboard.press("Enter");

  const surface = page.locator('main [role="img"][data-surface-support="ready"]');
  await expect(surface).toHaveAttribute("data-surface-point-count", "441");
  await expect(page.getByText(/X and Y locate each point on the grid/)).toBeVisible();
  await expect(page.getByText(/Grouping draws separate series/)).toBeVisible();
  await page.getByRole("button", { name: "Fitting and uncertainty" }).click();
  await expect(page.getByText(/Fitting estimates a trend/)).toBeVisible();

  const accessibility = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(
    accessibility.violations.filter(({ impact }) =>
      impact === "serious" || impact === "critical",
    ),
  ).toEqual([]);

  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文" }).click();
  await expect(page.getByText(/X 和 Y 用于定位网格点/)).toBeVisible();
  await expect(page.getByText(/拟合用于估计所选数据点的趋势/)).toBeVisible();
  await page.getByLabel("语言").click();
  await page.getByRole("option", { name: "English" }).click();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByText(/Continue on a desktop for fitting/)).toBeVisible();
  await page.setViewportSize({ width: 1_440, height: 1_024 });
  await page.getByRole("button", { name: "Customize export" }).click();
  await expect(
    page.getByRole("heading", { name: "Prepare publication export" }),
  ).toBeVisible();
});

test("explains each reversible cleaning choice beside the decision controls", async ({
  page,
}) => {
  await installMockApi(page);
  await page.goto("/workspace/project-e2e?step=inspect");

  await expect(
    page.getByText(/Exclude hides it only from analysis and charts/),
  ).toBeVisible();
  await expect(page.getByRole("radio", { name: "Keep as is" })).toBeVisible();
  await expect(
    page.getByRole("radio", { name: "Exclude from chart" }),
  ).toBeVisible();
  await expect(
    page.getByRole("radio", { name: "Remove from cleaned copy" }),
  ).toBeVisible();
});
