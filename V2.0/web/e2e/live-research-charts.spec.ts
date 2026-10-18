import { readFile } from "node:fs/promises";
import { expect, test } from "./fixtures";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires isolated local API.");

test("three scientific charts persist and export the same saved data", async ({ page }, testInfo) => {
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/api/v1/local/session"));
  await page.goto(`/#labviz-access=${process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY}`);
  expect((await unlocked).ok()).toBeTruthy();
  const rows = ["sample,value,unit,time,pair,group"];
  for (let s = 1; s <= 6; s++) {
    rows.push(`s${s},${s},mV,before,p${s},A`, `s${s},${s + 2},mV,after,p${s},A`);
  }
  const source = { name: "paired-chart.csv", mimeType: "text/csv", buffer: Buffer.from(rows.join("\n") + "\n") };
  await page.locator('input[type="file"]').setInputFiles(source);
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  const research = page.getByTestId("research-workspace");
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  await research.locator('input[type="file"]').first().setInputFiles(source);
  await expect(research.getByText(/12 source rows/)).toBeVisible();
  await research.getByRole("button", { name: "More metadata, limits and experimental records" }).click();
  for (const [label, field] of [
    ["Measurement value", "value"], ["Sample ID", "sample"],
    ["Independent experimental unit", "sample"], ["Unit column", "unit"],
    ["Group", "group"], ["Time point", "time"], ["Pair key", "pair"],
  ]) {
    await research.getByRole("combobox", { name: label, exact: true }).fill(field);
    await research.getByRole("combobox", { name: label, exact: true }).press("Tab");
  }
  await research.getByRole("button", { name: "Preview preparation", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Observations: 12");
  await research.getByRole("button", { name: "Apply preview and save revision" }).click();
  await page.getByRole("tab", { name: "Analysis", exact: true }).click();
  const charts = page.getByTestId("research-charts");
  await expect(charts.getByRole("button", { name: "Generate and save chart" })).toBeEnabled();
  const titles = ["Raw saved", "Paired saved", "Violin saved"];
  for (const [index, family] of ["Grouped raw points", "Paired change", "Violin distribution"].entries()) {
    await charts.getByRole("combobox", { name: "Chart family", exact: true }).click();
    await page.getByRole("option", { name: family, exact: true }).click();
    await charts.getByRole("textbox", { name: "Chart title", exact: true }).fill(titles[index]);
    await charts.getByRole("button", { name: "Generate and save chart" }).click();
    const img = charts.getByRole("img", { name: titles[index], exact: true });
    await expect(img).toBeVisible();
    await expect.poll(() => img.evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
    await expect(charts.getByTestId("chart-counts")).toContainText("Observations: 12");
    if (index === 1) {
      await expect(charts.getByRole("table", { name: "Paired changes" }).getByRole("row")).toHaveCount(7);
      await expect(charts.getByRole("table", { name: "Paired changes" }).getByRole("row").nth(1)).toContainText("2");
    }
    const jsonUrl = await charts.getByRole("link", { name: "Download JSON", exact: true }).getAttribute("href");
    const response = await page.request.get(jsonUrl!);
    expect(response.ok()).toBeTruthy();
    const frozen = await response.json();
    expect(frozen.result.points).toHaveLength(12);
    if (index === 1) expect(frozen.result.pairs.map((p: { delta: number }) => p.delta)).toEqual([2, 2, 2, 2, 2, 2]);
    if (index === 2) expect(frozen.result.densities[0].grid.length).toBeGreaterThanOrEqual(129);
    for (const format of ["PNG", "SVG", "PDF"]) {
      const downloadEvent = page.waitForEvent("download");
      await charts.getByRole("link", { name: `Download ${format}`, exact: true }).click();
      const downloaded = await downloadEvent;
      const output = testInfo.outputPath(`${index}-${format.toLowerCase()}.${format.toLowerCase()}`);
      await downloaded.saveAs(output);
      const bytes = await readFile(output);
      expect(bytes.length).toBeGreaterThan(1000);
      if (format === "PNG") expect(bytes.subarray(1, 4).toString()).toBe("PNG");
      if (format === "PDF") expect(bytes.subarray(0, 4).toString()).toBe("%PDF");
      if (format === "SVG") expect(bytes.toString()).toContain("<svg");
    }
    await charts.screenshot({ path: testInfo.outputPath(`chart-${index}.png`) });
  }
  await page.reload();
  await expect(page.getByRole("button", { name: "Back to quick plot", exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Analysis", exact: true }).click();
  await expect(charts.getByRole("img", { name: "Violin saved", exact: true })).toBeVisible();
  await charts.getByRole("combobox", { name: "Saved charts", exact: true }).click();
  await page.getByRole("option", { name: /Raw saved/ }).click();
  await expect(charts.getByRole("img", { name: "Raw saved", exact: true })).toBeVisible();
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文", exact: true }).click();
  await expect(charts.getByRole("heading", { name: "科学图表", exact: true })).toBeVisible();
  await charts.getByRole("textbox", { name: "图表标题", exact: true }).fill("研究原始点图");
  await charts.getByRole("button", { name: "生成并保存图表", exact: true }).click();
  const chinese = charts.getByRole("img", { name: "研究原始点图", exact: true });
  await expect.poll(() => chinese.evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  await charts.screenshot({ path: testInfo.outputPath("chart-mobile-zh.png") });
});
