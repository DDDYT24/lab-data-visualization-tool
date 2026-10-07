import { readFile } from "node:fs/promises";
import { expect, test } from "./fixtures";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires packaged local API and launcher bootstrap.");

test("real API scientific controls persist into the downloaded figure", async ({ page }) => {
  test.setTimeout(240_000);
  const observed: { fitting: Record<string, unknown>; body: { series: { fit: unknown }[] } }[] = [];
  page.on("response", async (response) => {
    if (response.url().endsWith("/chart-analysis") && response.ok()) {
      observed.push({ fitting: response.request().postDataJSON().chart.fitting, body: await response.json() });
    }
  });
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/local/session") && r.request().method() === "POST");
  await page.goto(`/#labviz-access=${encodeURIComponent(process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY!)}`);
  expect((await unlocked).ok()).toBeTruthy();
  await page.locator('input[type="file"]').setInputFiles({
    name: "live-controls.csv", mimeType: "text/csv",
    buffer: Buffer.from("time,response\n0,1.1\n1,2.8\n2,5.2\n3,6.9\n4,9.1\n5,10.8\n6,13.2\n7,15.1\n"),
  });
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Review data quality" }).click();
  const keep = page.getByRole("button", { name: "Keep all remaining as is" });
  if (await keep.isVisible()) await keep.click();
  await page.getByRole("button", { name: "Create chart", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Create your chart" })).toBeVisible();
  await page.getByRole("button", { name: "Fitting and uncertainty", exact: true }).click();
  const choose = async (label: string, option: string) => {
    await page.getByRole("combobox", { name: label, exact: true }).click();
    await page.getByRole("option", { name: option, exact: true }).click();
    await expect(page.getByRole("combobox", { name: label, exact: true })).toHaveText(option);
  };
  const analyze = async (action: () => Promise<void>, expected: Record<string, unknown>) => {
    await action();
    // React Query correctly reuses an earlier API answer when returning to the same fit.
    const find = () => observed.findLast((r) => Object.entries(expected).every(([key, value]) => r.fitting[key] === value));
    await expect.poll(() => Boolean(find())).toBeTruthy();
    const body = find()!.body;
    expect(body.series).toHaveLength(1);
    expect(body.series[0].fit).toMatchObject(expected);
    await expect(page.locator('main [role="img"] svg').first()).toBeVisible();
  };
  await analyze(() => choose("Fit model", "Linear"), { model: "linear" });
  await analyze(() => choose("Fit method", "Huber robust linear fit"), { fitMethod: "robust-huber" });
  await expect(page.getByRole("switch", { name: "Show confidence band" })).toBeDisabled();
  await analyze(() => choose("Fit method", "Ordinary least squares"), { fitMethod: "ordinary-least-squares" });
  await page.getByRole("switch", { name: "Show confidence band" }).click();
  await analyze(() => choose("Interval meaning", "Prediction interval for one response"), { intervalKind: "prediction" });
  await analyze(() => choose("Interval meaning", "Working-Hotelling simultaneous mean band"), { intervalKind: "simultaneous" });
  await expect(page.getByText("Displayed interval: Working-Hotelling simultaneous mean band.", { exact: true })).toBeVisible();
  await page.getByRole("textbox", { name: "Figure title", exact: true }).fill("Live controls verified");
  await page.getByRole("button", { name: "Series appearance", exact: true }).click();
  await page.locator('input[type="color"]').first().fill("#dc2626");
  await page.getByRole("button", { name: "Customize export", exact: true }).click();
  await choose("Format", "SVG");
  await choose("Resolution", "600 DPI");
  await choose("Figure size", "Double column");
  await page.getByRole("switch", { name: "Show chart grid", exact: true }).click();
  await page.getByRole("button", { name: "Prepare export", exact: true }).click();
  const link = page.getByRole("link", { name: "Download SVG", exact: true });
  await expect(link).toBeVisible({ timeout: 60_000 });
  const event = page.waitForEvent("download");
  await link.click();
  const download = await event;
  expect(await download.failure()).toBeNull();
  const svg = await readFile((await download.path())!, "utf8");
  expect(svg).toContain("Live controls verified");
  expect(svg).toContain("#dc2626");
  expect(svg).toContain("<svg");
  await download.saveAs(test.info().outputPath("controls.svg"));
  const workspace = new URL(page.url()).pathname;
  await page.goto(`${workspace}?step=chart`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Fitting and uncertainty", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Interval meaning" })).toHaveText("Working-Hotelling simultaneous mean band");
  await expect(page.getByRole("textbox", { name: "Figure title", exact: true })).toHaveValue("Live controls verified");
});

test("real API import failure explains the error and permits recovery", async ({ page }) => {
  test.setTimeout(120_000);
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/local/session") && r.request().method() === "POST");
  await page.goto(`/#labviz-access=${encodeURIComponent(process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY!)}`);
  expect((await unlocked).ok()).toBeTruthy();
  const failedJob = page.waitForResponse(async (r) => {
    if (!/\/jobs\/[a-z0-9]+$/.test(r.url()) || !r.ok()) return false;
    return (await r.json()).stage === "failed";
  });
  await page.locator('input[type="file"]').setInputFiles({
    name: "unreadable.xlsx", mimeType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    buffer: Buffer.from("This is deliberately not a ZIP workbook."),
  });
  const failed = await (await failedJob).json();
  expect(JSON.stringify(failed)).toContain("unreadable");
  await expect(page.getByText("The workbook is unreadable or password protected.", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Choose another file", exact: true }).click();
  await page.locator('input[type="file"]').setInputFiles({
    name: "recovered.csv", mimeType: "text/csv", buffer: Buffer.from("time,response\n0,1\n1,2\n2,4\n"),
  });
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("button", { name: "Review data quality", exact: true })).toBeEnabled();
  await page.goto("/history", { waitUntil: "networkidle" });
  await expect(page.getByText("recovered.csv").first()).toBeVisible();
});
