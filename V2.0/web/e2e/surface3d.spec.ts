import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("keeps the 21 x 21 surface contract aligned through preview and every export", async ({
  page,
}) => {
  const observations = await installMockApi(page, { dataset: "surface" });
  await page.goto("/workspace/project-e2e?step=chart");

  await expect(page.getByRole("heading", { name: "Create your chart" })).toBeVisible();
  await page.getByRole("combobox", { name: "Chart type" }).click();
  await page.getByRole("option", { name: "3D surface" }).click();

  await page.getByRole("combobox", { name: "X field" }).click();
  await page.getByRole("option", { name: "X", exact: true }).click();
  await page.getByRole("combobox", { name: "Surface Y field" }).click();
  await page.getByRole("option", { name: "Y", exact: true }).click();
  await page.getByRole("combobox", { name: "Surface Z field" }).click();
  await page.getByRole("option", { name: "Z", exact: true }).click();

  const surface = page.locator('main [role="img"][data-surface-support="ready"]');
  await expect(surface).toBeVisible();
  await expect(surface).toHaveAttribute(
    "data-chart-components",
    "grid3D,xAxis3D,yAxis3D,zAxis3D",
  );
  await expect(surface).toHaveAttribute("data-surface-point-count", "441");
  await expect(surface).toHaveAttribute("data-camera-controls", "true");
  const canvas = surface.locator("canvas[data-zr-dom-id]");
  await expect(canvas).toBeVisible();

  const bounds = await canvas.boundingBox();
  expect(bounds).not.toBeNull();
  if (bounds) {
    const centerX = bounds.x + bounds.width / 2;
    const centerY = bounds.y + bounds.height / 2;
    await page.mouse.move(centerX, centerY);
    await page.mouse.down();
    await page.mouse.move(centerX + 30, centerY + 20, { steps: 4 });
    await page.mouse.up();
    await page.mouse.wheel(0, -120);
  }
  await expect(canvas).toBeVisible();

  const expectedSurfaceFields = {
    type: "surface3d",
    xAxis: { field: "x" },
    series: [{ field: "y" }, { field: "z" }],
  };
  await expect.poll(() =>
    observations.analysisCharts.findLast((chart) => chart.type === "surface3d"),
  ).toMatchObject(expectedSurfaceFields);

  await page.getByRole("button", { name: "Customize export" }).click();
  await expect(
    page.getByRole("heading", { name: "Prepare publication export" }),
  ).toBeVisible();
  expect(observations.savedCharts.at(-1)).toMatchObject(expectedSurfaceFields);

  for (const format of ["PNG", "SVG", "PDF"] as const) {
    if (format !== "PNG") {
      await page.getByLabel("Format").click();
      await page.getByRole("option", { name: format }).click();
    }
    await page.getByRole("button", { name: "Prepare export" }).click();
    await expect(page.getByText(`${format} export is ready.`)).toBeVisible();
  }

  expect(observations.exportFormats).toEqual(["png", "svg", "pdf"]);
  expect(observations.exportCharts).toHaveLength(3);
  for (const exportedChart of observations.exportCharts) {
    expect(exportedChart).toMatchObject(expectedSurfaceFields);
  }
});
