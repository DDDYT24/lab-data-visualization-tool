import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("About renders editable Markdown, real synthetic LabViz exports and feedback", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/");
  await page.getByRole("link", { name: "About", exact: true }).first().click();
  await expect(page).toHaveURL(/\/about$/);
  await expect(page.getByRole("heading", { name: "About LabViz" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Example: a 2D time series" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Example: a 3D surface" })).toBeVisible();
  await expect(page.getByRole("img", { name: /Synthetic time-series chart exported by LabViz/ })).toBeVisible();
  await expect(page.getByRole("img", { name: /Synthetic gridded surface exported by LabViz/ })).toBeVisible();
  await expect(page.getByRole("link", { name: "Email liyutao982@gmail.com" })).toHaveAttribute(
    "href", "mailto:liyutao982@gmail.com",
  );
  await page.getByRole("link", { name: "Open the bundled examples" }).first().click();
  await expect(page).toHaveURL(/\?examples=1/);
  await expect(page.getByRole("dialog", { name: "Choose an example dataset" })).toBeVisible();
});

test("New analysis from history focuses the real file chooser", async ({ page }) => {
  await installMockApi(page, { history: "saved" });
  await page.goto("/history");
  await page.getByRole("link", { name: "New analysis" }).last().click();
  await expect(page).toHaveURL(/\/#import$/);
  await expect(page.getByRole("button", { name: "Choose a file" })).toBeFocused();
});
