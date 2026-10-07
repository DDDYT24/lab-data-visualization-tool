import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";
import budgets from "../../api/samples/v22/performance_budgets.json";

const BROWSER_BUDGETS = {
  sampleImportToReadyMs: budgets.browser.sample_import_to_ready_ms,
  firstChartMs: budgets.browser.first_chart_ms,
  chartInteractionMs: budgets.browser.chart_interaction_ms,
} as const;

test("meets the local browser responsiveness budgets", async ({ page }) => {
  const observations = await installMockApi(page);
  await page.goto("/");

  const importStarted = performance.now();
  await page.getByRole("button", { name: "View examples" }).click();
  await page
    .locator('[data-example-slug="time-series"]')
    .getByRole("button", { name: "Open this example" })
    .click();
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  const sampleImportToReadyMs = Math.round(performance.now() - importStarted);

  const chartStarted = performance.now();
  await page.getByRole("button", { name: "Review data quality" }).click();
  await expect(page.getByRole("heading", { name: "Review data quality" })).toBeVisible();
  await page.getByRole("radio", { name: "Keep as is" }).check();
  const nextFinding = page.getByRole("button", { name: "Next quality finding" });
  if (await nextFinding.isVisible()) {
    await nextFinding.click();
    await page.getByRole("radio", { name: "Keep as is" }).check();
  }
  await page.getByRole("button", { name: "Create chart" }).click();
  await expect(page.getByRole("heading", { name: "Create your chart" })).toBeVisible();
  await expect(page.locator('main [role="img"]').first()).toBeVisible();
  const firstChartMs = Math.round(performance.now() - chartStarted);

  const previousAnalysisCount = observations.analysisCharts.length;
  const interactionStarted = performance.now();
  await page.getByRole("combobox", { name: "Chart type" }).click();
  await page.getByRole("option", { name: "Scatter plot" }).click();
  await expect
    .poll(() => observations.analysisCharts.length)
    .toBeGreaterThan(previousAnalysisCount);
  const chartInteractionMs = Math.round(performance.now() - interactionStarted);

  const metrics = { sampleImportToReadyMs, firstChartMs, chartInteractionMs };
  test.info().annotations.push({ type: "performance", description: JSON.stringify(metrics) });
  expect(sampleImportToReadyMs).toBeLessThanOrEqual(BROWSER_BUDGETS.sampleImportToReadyMs);
  expect(firstChartMs).toBeLessThanOrEqual(BROWSER_BUDGETS.firstChartMs);
  expect(chartInteractionMs).toBeLessThanOrEqual(BROWSER_BUDGETS.chartInteractionMs);
});
