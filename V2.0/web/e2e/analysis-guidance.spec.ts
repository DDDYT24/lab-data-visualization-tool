import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("exposes analysis method, evidence, and deferred-method disclosures", async ({
  page,
}) => {
  const observations = await installMockApi(page);
  await page.goto("/workspace/project-e2e?step=chart");

  await page.getByRole("button", { name: "Fitting and uncertainty" }).click();
  await page.getByRole("combobox", { name: "Fit model" }).click();
  await page.getByRole("option", { name: "Linear", exact: true }).click();
  await page.getByRole("switch", { name: "Show confidence band" }).click();
  await page.getByRole("combobox", { name: "Confidence-band method" }).click();
  await page
    .getByRole("option", { name: "Residual-bootstrap pointwise mean interval" })
    .click();

  await expect(page.getByText(/Sample size n=4/)).toBeVisible();
  await expect(page.getByText(/Residual diagnostic: RMSE=0.1500/)).toBeVisible();
  await expect(
    page.getByText(/Deferred in this release: prediction intervals/),
  ).toBeVisible();

  await expect.poll(() => {
    const latest = observations.analysisCharts.at(-1) as
      | (typeof observations.analysisCharts)[number] & {
          fitting?: { confidenceMethod?: string; model?: string };
        }
      | undefined;
    return latest?.fitting;
  }).toMatchObject({ confidenceMethod: "bootstrap", model: "linear" });

  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文" }).click();
  await expect(page.getByText(/样本量 n=4/)).toBeVisible();
  await expect(page.getByText(/所选模型形式适合当前科研问题/)).toBeVisible();
});
