import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("keeps advanced interval and robust-fit choices explicit in the chart request", async ({
  page,
}) => {
  const observations = await installMockApi(page);
  await page.goto("/workspace/project-e2e?step=chart");

  await page.getByRole("button", { name: "Fitting and uncertainty" }).click();
  await page.getByRole("combobox", { name: "Fit model" }).click();
  await page.getByRole("option", { name: "Linear", exact: true }).click();

  await page.getByRole("combobox", { name: "Fit method" }).click();
  await page.getByRole("option", { name: "Huber robust linear fit" }).click();
  await expect(page.getByText(/Huber robust fitting is limited to linear models/)).toBeVisible();
  await expect(page.getByRole("switch", { name: "Show confidence band" })).toBeDisabled();

  await page.getByRole("combobox", { name: "Fit method" }).click();
  await page.getByRole("option", { name: "Ordinary least squares" }).click();
  await page.getByRole("switch", { name: "Show confidence band" }).click();
  await page.getByRole("combobox", { name: "Interval meaning" }).click();
  await page.getByRole("option", { name: "Prediction interval for one response" }).click();

  await expect(page.getByRole("combobox", { name: "Confidence-band method" })).toBeDisabled();
  await expect.poll(() => observations.analysisCharts.at(-1)?.fitting).toMatchObject({
    confidenceBand: true,
    confidenceMethod: "student-t",
    fitMethod: "ordinary-least-squares",
    intervalKind: "prediction",
    model: "linear",
  });
  await expect(page.getByText(/Sample size n=4/)).toBeVisible();
  await expect(
    page.getByText("Displayed interval: Prediction interval for one response.", { exact: true }),
  ).toBeVisible();
});
