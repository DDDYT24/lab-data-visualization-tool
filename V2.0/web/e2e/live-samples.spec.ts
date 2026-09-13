import { expect, test } from "./fixtures";

const sampleSlugs = [
  "time-series",
  "repeated-runs",
  "scatter-fit",
  "categorical-comparison",
  "distribution",
  "correlation-heatmap",
  "surface-3d",
] as const;

const recommendedChartLabels = {
  "time-series": "Line chart",
  "repeated-runs": "Line chart",
  "scatter-fit": "Scatter plot",
  "categorical-comparison": "Box plot",
  distribution: "Histogram",
  "correlation-heatmap": "Heatmap",
  "surface-3d": "3D surface",
} as const;

test.skip(
  process.env.LABVIZ_E2E_LIVE !== "1",
  "Set LABVIZ_E2E_LIVE=1, start FastAPI, and run the real sample workflow.",
);

test("runs every bundled example through the real API and PNG export", async ({ page }) => {
  test.setTimeout(600_000);

  for (const slug of sampleSlugs) {
    await page.goto("/");
    await page.getByRole("button", { name: "Try sample data" }).click();
    const card = page.locator(`[data-example-slug="${slug}"]`);
    await expect(card).toBeVisible();
    await card.getByRole("button", { name: "Open this example" }).click();

    await expect(page).toHaveURL(/\/workspace\/[a-z0-9]+$/);
    await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({
      timeout: 120_000,
    });
    await page.getByRole("button", { name: "Review data quality" }).click();
    await expect(page.getByRole("heading", { name: "Review data quality" })).toBeVisible();

    for (let index = 0; index < 100; index += 1) {
      const keep = page.getByRole("radio", { name: "Keep as is" }).first();
      if (!(await keep.isVisible())) {
        break;
      }
      await keep.check();
      const next = page.getByRole("button", { name: "Next quality finding" });
      if (!(await next.isEnabled())) {
        break;
      }
      await next.click();
    }

    const createChart = page.getByRole("button", { name: "Create chart" });
    await expect(createChart).toBeEnabled({ timeout: 120_000 });
    await createChart.click();
    await expect(page.getByRole("heading", { name: "Create your chart" })).toBeVisible({
      timeout: 120_000,
    });
    await expect(page.getByRole("combobox", { name: "Chart type" })).toHaveText(
      recommendedChartLabels[slug],
    );
    await expect(page.locator('main [role="img"]').first()).toBeVisible({ timeout: 120_000 });
    await page.getByRole("button", { name: "Customize export" }).click();
    await expect(
      page.getByRole("heading", { name: "Prepare publication export" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Prepare export" }).click();
    const downloadPromise = page.waitForEvent("download");
    const downloadLink = page.getByRole("link", { name: "Download PNG" });
    await expect(downloadLink).toBeVisible({ timeout: 120_000 });
    await downloadLink.click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/\.png$/i);
    expect(await download.failure()).toBeNull();
  }
});
