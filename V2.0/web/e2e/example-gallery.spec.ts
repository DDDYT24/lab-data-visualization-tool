import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

const examples = [
  ["time-series", "Time-series response", "时间响应曲线"],
  ["repeated-runs", "Repeated experiment runs", "重复实验运行"],
  ["scatter-fit", "Dose-response scatter", "剂量-响应散点"],
  ["categorical-comparison", "Categorical group comparison", "分类分组比较"],
  ["distribution", "Measurement distributions", "测量值分布"],
  ["correlation-heatmap", "Multi-variable correlation", "多变量相关性"],
  ["surface-3d", "Regular X/Y/Z surface", "规则 X/Y/Z 3D 曲面"],
] as const;

test("shows the versioned example catalog and opens a selected example", async ({ page }) => {
  const observations = await installMockApi(page);

  await page.goto("/");
  await page.getByRole("button", { name: "View examples" }).click();

  const dialog = page.getByRole("dialog");
  await expect(
    dialog.getByRole("heading", { name: "Choose an example dataset", exact: true }),
  ).toBeVisible();
  await expect(dialog.locator("[data-example-slug]")).toHaveCount(7);
  const surface = dialog.locator('[data-example-slug="surface-3d"]');
  await expect(surface).toContainText("Regular X/Y/Z surface");
  await expect(surface).toContainText("3D surface");
  await expect(surface.getByText(/441 rows/)).toBeVisible();

  await surface.getByRole("button", { name: "Open this example" }).focus();
  await surface.getByRole("button", { name: "Open this example" }).press("Enter");
  await expect(page).toHaveURL(/\/workspace\/project-e2e$/);
  expect(observations.sampleSlugs).toEqual(["surface-3d"]);
});

test("keeps the example chooser bilingual", async ({ page }) => {
  await installMockApi(page);

  await page.goto("/");
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文" }).click();
  await page.getByRole("button", { name: "查看示例" }).click();

  const dialog = page.getByRole("dialog");
  await expect(
    dialog.getByRole("heading", { name: "选择示例数据", exact: true }),
  ).toBeVisible();
  for (const [slug, , chineseTitle] of examples) {
    await expect(dialog.locator(`[data-example-slug="${slug}"]`)).toContainText(chineseTitle);
  }
  await expect(dialog.getByRole("button", { name: "打开此示例" }).first()).toBeVisible();
});

for (const [slug, englishTitle] of examples) {
  test(`keeps ${slug} metadata and route stable through refresh and back`, async ({ page }) => {
    const observations = await installMockApi(page);

    await page.goto("/");
    await page.getByRole("button", { name: "View examples" }).click();
    const card = page.locator(`[data-example-slug="${slug}"]`);
    await expect(card).toContainText(englishTitle);
    await expect(card.getByText(/Fields:/)).toBeVisible();
    await expect(card.getByText(/Learn:/)).toBeVisible();
    if (slug === "time-series") {
      await expect(card.getByText(/Quality lesson:/)).toBeVisible();
    }
    await card.getByRole("button", { name: "Open this example" }).press("Enter");

    await expect(page).toHaveURL(/\/workspace\/project-e2e$/);
    await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
    await page.reload();
    await expect(page.getByRole("heading", { name: "Create your chart" })).toBeVisible();
    await page.goBack();
    await expect(page).toHaveURL(/\/$/);
    expect(observations.sampleSlugs).toEqual([slug]);
  });
}
