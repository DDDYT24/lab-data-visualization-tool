import path from "node:path";

import { expect, test } from "./fixtures";
import { gridColor } from "../src/domain/chart-render-contract";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires the real local API and launcher bootstrap.");

const SAMPLE = path.resolve(__dirname, "../../api/samples/v22/05_distribution.json");

async function openChartStep(page: import("@playwright/test").Page) {
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/api/v1/local/session") && r.request().method() === "POST");
  await page.goto(`/#labviz-access=${encodeURIComponent(process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY!)}`);
  expect((await unlocked).ok()).toBeTruthy();
  await page.locator('input[type="file"]').setInputFiles(SAMPLE);
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Review data quality" }).click();
  const keep = page.getByRole("button", { name: "Keep all remaining as is" });
  if (await keep.isVisible()) await keep.click();
  await page.getByRole("button", { name: "Create chart" }).click();
  await expect(page.getByRole("combobox", { name: "Chart type" })).toHaveText("Histogram");
  const chart = page.locator('main [role="img"]').first();
  await expect(chart.locator("svg")).toBeVisible();
  return chart;
}

test("planned previews follow appearance edits and the grid setting", async ({ page }) => {
  test.setTimeout(120_000);
  const chart = await openChartStep(page);
  await page.getByRole("button", { name: "Series appearance", exact: true }).click();
  await chart.locator("svg").evaluate((svg) => { (svg as SVGElement & { marker?: boolean }).marker = true; });
  await page.locator('input[type="color"]').first().fill("#dc2626");
  await expect.poll(() => chart.locator('path[fill="#dc2626"]').count()).toBeGreaterThan(0);
  // The chart stays on screen while its new plan loads.
  expect(await chart.locator("svg").evaluate((svg) => (svg as SVGElement & { marker?: boolean }).marker)).toBe(true);

  await page.getByRole("button", { name: "Customize export" }).click();
  const shown = page.locator('main [role="img"]').first();
  await expect.poll(() => shown.locator(`path[stroke="${gridColor}"]`).count()).toBeGreaterThan(0);
  await page.getByLabel("Show chart grid").uncheck();
  await expect.poll(() => shown.locator(`path[stroke="${gridColor}"]`).count()).toBe(0);
  await expect(shown.locator('path[fill="#dc2626"]').first()).toBeAttached();
});

test("a box preview has no legend and no grid", async ({ page }) => {
  test.setTimeout(120_000);
  const chart = await openChartStep(page);
  await page.getByRole("combobox", { name: "Chart type" }).click();
  await page.getByRole("option", { name: "Box plot", exact: true }).click();
  await expect.poll(() => chart.locator('svg path[stroke="#ff7f0e"]').count()).toBeGreaterThan(0);
  const projectId = new URL(page.url()).pathname.split("/").at(-1)!;
  const workspace = await (await page.request.get(`/api/v1/projects/${projectId}/workspace`)).json();
  const series: string = workspace.chart.series[0].label;
  // The category tick and the y title carry this name; a legend would be a third.
  await expect(chart.locator("svg text", { hasText: series })).toHaveCount(2);
  await expect(chart.locator(`svg path[stroke="${gridColor}"]`)).toHaveCount(0);
  await chart.screenshot({ path: test.info().outputPath("box-preview.png") });
});

test("captured box panels share an axis and open as interactive panels", async ({ page }) => {
  test.setTimeout(180_000);
  await openChartStep(page);
  const projectId = new URL(page.url()).pathname.split("/").at(-1)!;
  const project = `/api/v1/projects/${projectId}`;
  const { chart } = await (await page.request.get(`${project}/workspace`)).json();
  const capture = async (color: string) => {
    const box = { ...chart, type: "box", series: [{ ...chart.series[0], color }] };
    expect((await page.request.put(`${project}/chart`, { data: { chart: box } })).ok()).toBeTruthy();
    const captured = await page.request.post(`${project}/research/figure-sources?panel=1`);
    expect(captured.ok()).toBeTruthy();
    return (await captured.json()).id as string;
  };
  const boards = `${project}/research/figure-boards`;
  const created = await page.request.post(boards, {
    data: { title: "Captured boxes", panelChartIds: [await capture("#1768ac"), await capture("#0f766e")], axisMode: "shared_y" },
  });
  expect(created.ok()).toBeTruthy();
  const board = await created.json();
  const scene = await (await page.request.get(`${boards}/${board.id}/interaction`)).json();
  const planHashes = scene.panels.map((panel: { planHash: string | null }) => panel.planHash);
  expect(planHashes.every(Boolean)).toBe(true);
  const plan = await (await page.request.get(`${boards}/${board.id}/plans/A`)).json();
  const y = plan.panels[0].axes.y;
  expect(y.ticks.at(-1).value).toBeLessThanOrEqual(y.domain[1]);
  expect(scene.panels[0].yLimits).toEqual(y.domain);

  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  // The workspace opens on the newest saved layout.
  const workspace = page.getByTestId("figure-workspace");
  await expect(workspace.getByRole("textbox", { name: "Figure title", exact: true })).toHaveValue("Captured boxes");
  const inspector = workspace.getByTestId("figure-inspector");
  await inspector.getByRole("button", { name: "Interactive panels", exact: true }).click();
  const panels = inspector.getByTestId("render-plan-chart");
  await expect(panels).toHaveCount(2);
  await expect(panels.first().locator("canvas").first()).toBeVisible();
  expect(await panels.evaluateAll((items) => items.map((item) => item.getAttribute("data-plan-hash")))).toEqual(planHashes);
  await inspector.screenshot({ path: test.info().outputPath("captured-boxes.png") });
});

test("line and scatter previews are drawn from plans, restyle, and carry a fit", async ({ page }) => {
  test.setTimeout(180_000);
  type Mark = { kind: string; color: string; lineStyle?: string };
  const plans: { type: string; marks: Mark[] }[] = [];
  page.on("response", async (response) => {
    if (response.url().endsWith("/chart-analysis") && response.ok()) {
      const body = await response.json();
      if (body.plan) plans.push({ type: response.request().postDataJSON().chart.type, marks: body.plan.panels[0].marks });
    }
  });
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/api/v1/local/session") && r.request().method() === "POST");
  await page.goto(`/#labviz-access=${encodeURIComponent(process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY!)}`);
  expect((await unlocked).ok()).toBeTruthy();
  await page.locator('input[type="file"]').setInputFiles({
    name: "live-line-scatter.csv", mimeType: "text/csv",
    buffer: Buffer.from("time,response\n0,1.1\n1,2.8\n2,5.2\n3,6.9\n4,9.1\n5,10.8\n6,13.2\n7,15.1\n"),
  });
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Review data quality" }).click();
  const keep = page.getByRole("button", { name: "Keep all remaining as is" });
  if (await keep.isVisible()) await keep.click();
  await page.getByRole("button", { name: "Create chart", exact: true }).click();
  const chart = page.locator('main [role="img"]').first();
  await expect(chart.locator("svg")).toBeVisible();
  const latest = (type: string, wanted: (mark: Mark) => boolean) =>
    expect.poll(() => plans.findLast((plan) => plan.type === type)?.marks.some(wanted) ?? false).toBe(true);
  const chooseType = async (name: string) => {
    await page.getByRole("combobox", { name: "Chart type" }).click();
    await page.getByRole("option", { name, exact: true }).click();
  };

  await chooseType("Line chart");
  await latest("line", (mark) => mark.kind === "polyline");
  await page.getByRole("button", { name: "Series appearance", exact: true }).click();
  await page.locator('input[type="color"]').first().fill("#dc2626");
  // The restyle reaches the plan and the drawing, not only the request.
  await latest("line", (mark) => mark.kind === "polyline" && mark.color.toLowerCase() === "#dc2626");
  await expect.poll(() => chart.locator('svg path[stroke="#dc2626"]').count()).toBeGreaterThan(0);

  await chooseType("Scatter plot");
  await latest("scatter", (mark) => mark.kind === "points" && mark.color.toLowerCase() === "#dc2626");
  await expect.poll(() => chart.locator('svg path[fill="#dc2626"]').count()).toBeGreaterThan(0);

  await page.getByRole("button", { name: "Fitting and uncertainty", exact: true }).click();
  await page.getByRole("combobox", { name: "Fit model", exact: true }).click();
  await page.getByRole("option", { name: "Linear", exact: true }).click();
  await latest("scatter", (mark) => mark.kind === "polyline" && mark.lineStyle === "dashed");
  await chart.screenshot({ path: test.info().outputPath("scatter-fit-preview.png") });
});
