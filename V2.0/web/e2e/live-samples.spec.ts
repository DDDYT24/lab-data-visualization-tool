import { expect, test } from "./fixtures";
import { readFile } from "node:fs/promises";
import manifest from "../../api/samples/v22/manifest.json";
import budgets from "../../api/samples/v22/performance_budgets.json";

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

test("runs every bundled example through the real API and all exports", async ({ page }) => {
  test.setTimeout(600_000);
  let lastWorkspaceUrl = "";

  for (const slug of sampleSlugs) {
    await page.goto("/");
    await page.getByRole("button", { name: "Try sample data" }).click();
    const card = page.locator(`[data-example-slug="${slug}"]`);
    await expect(card).toBeVisible();
    await card.getByRole("button", { name: "Open this example" }).focus();
    const importStarted = performance.now();
    await card.getByRole("button", { name: "Open this example" }).press("Enter");

    await expect(page).toHaveURL(/\/workspace\/[a-z0-9]+$/);
    await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({
      timeout: 120_000,
    });
    const importMs = performance.now() - importStarted;
    expect(importMs).toBeLessThanOrEqual(budgets.browser.sample_import_to_ready_ms);
    await page.getByRole("button", { name: "Review data quality" }).click();
    await expect(page.getByRole("heading", { name: "Review data quality" })).toBeVisible();

    const keepRemaining = page.getByRole("button", { name: "Keep all remaining as is" });
    if (await keepRemaining.isVisible()) {
      const saved = page.waitForResponse((response) =>
        response.url().endsWith("/cleaning-decisions") &&
        response.request().method() === "PATCH",
      );
      await keepRemaining.click();
      expect((await saved).ok()).toBeTruthy();
    }

    const createChart = page.getByRole("button", { name: "Create chart" });
    await expect(createChart).toBeEnabled({ timeout: 120_000 });
    const chartStarted = performance.now();
    await createChart.click();
    await expect(page.getByRole("heading", { name: "Create your chart" })).toBeVisible({
      timeout: 120_000,
    });
    await expect(page.getByRole("combobox", { name: "Chart type" })).toHaveText(
      recommendedChartLabels[slug],
    );
    await expect(page.locator('main [role="img"]').first()).toBeVisible({ timeout: 120_000 });
    const chartMs = performance.now() - chartStarted;
    expect(chartMs).toBeLessThanOrEqual(budgets.browser.first_chart_ms);
    const interactionStarted = performance.now();
    // A title edit is presentation-only: it deliberately does not rerun analysis.
    await page.getByRole("textbox", { name: "Figure title" }).fill(`Verified ${slug}`);
    await expect(page.locator('main [role="img"]').first()).toHaveAttribute(
      "aria-label", new RegExp(`Verified ${slug}`),
    );
    const interactionMs = performance.now() - interactionStarted;
    expect(interactionMs).toBeLessThanOrEqual(budgets.browser.chart_interaction_ms);
    test.info().annotations.push({type: "live-performance", description: JSON.stringify({slug, importMs, chartMs, interactionMs})});
    await page.getByRole("button", { name: "Customize export" }).click();
    await expect(
      page.getByRole("heading", { name: "Prepare publication export" }),
    ).toBeVisible();
    for (const format of ["PNG", "SVG", "PDF"]) {
      await page.getByRole("combobox", { name: "Format", exact: true }).click();
      await page.getByRole("option", { name: format, exact: true }).click();
      await page.getByRole("button", { name: "Prepare export" }).click();
      const downloadLink = page.getByRole("link", { name: `Download ${format}`, exact: true });
      await expect(downloadLink).toBeVisible({ timeout: 120_000 });
      const downloadPromise = page.waitForEvent("download");
      await downloadLink.click();
      const download = await downloadPromise;
      expect(await download.failure()).toBeNull();
      const bytes = await readFile((await download.path())!);
      expect(bytes.length).toBeGreaterThan(100);
      if (format === "PNG") expect(bytes.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
      if (format === "PDF") expect(bytes.subarray(0, 4).toString()).toBe("%PDF");
      if (format === "SVG") expect(bytes.toString()).toContain("<svg");
    }
    const cleanedPromise = page.waitForEvent("download");
    await page.getByRole("link", { name: "Download complete cleaned data (CSV)" }).click();
    const cleaned = await cleanedPromise;
    expect(await cleaned.failure()).toBeNull();
    const csv = await readFile((await cleaned.path())!, "utf8");
    const example = manifest.examples.find((entry) => entry.slug === slug)!;
    expect(csv.trim().split(/\r?\n/).length - 1).toBe(example.rowCount);
    expect(example.fields).toBeDefined();
    for (const field of example.fields!) expect(csv.split(/\r?\n/)[0]).toContain(field.name);
    lastWorkspaceUrl = page.url();
  }
  await page.reload({ waitUntil: "networkidle" });
  await expect(page).toHaveURL(lastWorkspaceUrl);
  await page.goto("/", { waitUntil: "networkidle" });
  await page.goBack({ waitUntil: "networkidle" });
  await expect(page).toHaveURL(lastWorkspaceUrl);
  await page.goto("/");
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文" }).click();
  await page.getByRole("button", { name: "使用示例数据" }).click();
  for (const example of manifest.examples.filter((entry) => entry.visibility === "public")) {
    await expect(page.locator(`[data-example-slug="${example.slug}"]`)).toContainText(example.title!.zh);
  }
});
