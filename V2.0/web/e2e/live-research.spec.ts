import { expect, test } from "./fixtures";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires an isolated local API.");
test("research preview, sample counts, exclusions and revision history survive reopening", async ({ page }, testInfo) => {
  const key = process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY!;
  const unlocked = page.waitForResponse((response) => response.url().endsWith("/api/v1/local/session"));
  await page.goto(`/#labviz-access=${key}`);
  expect((await unlocked).ok()).toBeTruthy();
  const rows = ["sample,unitId,repeat,value,unit"];
  for (let sample = 1; sample <= 6; sample += 1) {
    for (let repeat = 1; repeat <= 3; repeat += 1) {
      rows.push(`s${sample},u${sample},${repeat},${sample + repeat - 2},mV`);
    }
  }
  const source = { name: "research-repeat.csv", mimeType: "text/csv", buffer: Buffer.from(rows.join("\n") + "\n") };
  await page.locator('input[type="file"]').setInputFiles(source);
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  const research = page.getByTestId("research-workspace");
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  await research.locator('input[type="file"]').first().setInputFiles(source);
  await expect(research.getByText(/18 source rows/)).toBeVisible();
  for (const [label, field] of [
    ["Measurement value", "value"], ["Sample ID", "sample"],
    ["Independent experimental unit", "unitId"], ["Technical repeat", "repeat"], ["Unit column", "unit"],
  ]) {
    await research.getByRole("combobox", { name: label, exact: true }).fill(field);
    await research.getByRole("combobox", { name: label, exact: true }).press("Tab");
  }
  await page.getByRole("tab", { name: "Quality", exact: true }).focus();
  await page.keyboard.press("Enter");
  await page.getByRole("tab", { name: "Data", exact: true }).click();
  await expect(research.getByRole("combobox", { name: "Measurement value", exact: true })).toHaveValue("value");
  await research.getByRole("combobox", { name: "Analysis unit", exact: true }).click();
  await page.getByRole("option", { name: "Independent experimental unit", exact: true }).click();
  await research.getByRole("button", { name: "Preview preparation", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Independent units: 6");
  await expect(research.getByTestId("research-counts")).toContainText("Observations: 18");
  await expect(research.getByTestId("research-mean")).toContainText("3.5");
  await research.getByRole("button", { name: "Apply preview and save revision" }).click();
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  await research.getByRole("button", { name: "Select", exact: true }).first().click();
  await expect(research.getByTestId("research-counts")).toContainText("Excluded: 0");
  await research.getByRole("textbox", { name: "Exclusion reason", exact: true }).fill("Instrument log identifies this reading as invalid.");
  await research.getByRole("button", { name: "Exclude selected observation", exact: true }).click();
  await page.getByRole("tab", { name: "Data", exact: true }).click();
  await research.getByRole("button", { name: "Preview preparation", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Excluded: 1");
  await expect(research.getByTestId("research-counts")).toContainText("Independent units: 6");
  await research.getByRole("button", { name: "Apply preview and save revision" }).click();
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  await research.getByRole("button", { name: "Undo research change", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Excluded: 0");
  await research.getByRole("button", { name: "Redo", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Excluded: 1");
  await page.reload();
  await expect(page.getByRole("button", { name: "Back to quick plot", exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Quality", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Excluded: 1");
  await page.screenshot({ path: testInfo.outputPath("research-desktop.png"), fullPage: true });
  await page.getByRole("tab", { name: "Data", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(research.getByRole("button", { name: "Preview preparation" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("research-mobile.png"), fullPage: true });
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文", exact: true }).click();
  await expect(research.getByRole("heading", { name: "研究工作台", exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "质量", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("独立实验单位: 6");
  await page.screenshot({ path: testInfo.outputPath("research-mobile-zh.png"), fullPage: true });
});


test("two-file stacking exposes observations beyond the sampled first page", async ({ page }) => {
  page.setDefaultTimeout(10_000);
  const key = process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY!;
  const unlocked = page.waitForResponse((response) => response.url().endsWith("/api/v1/local/session"));
  await page.goto(`/#labviz-access=${key}`);
  expect((await unlocked).ok()).toBeTruthy();
  await page.locator('input[type="file"]').setInputFiles({
    name: "quick.csv", mimeType: "text/csv", buffer: Buffer.from("x,y\n0,1\n"),
  });
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  const research = page.getByTestId("research-workspace");
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  await research.locator('input[type="file"]').first().setInputFiles([
    { name: "part-1.csv", mimeType: "text/csv", buffer: Buffer.from("v\n" + Array.from({ length: 100 }, (_, index) => index).join("\n") + "\n") },
    { name: "part-2.csv", mimeType: "text/csv", buffer: Buffer.from("v\n" + Array.from({ length: 105 }, (_, index) => index + 100).join("\n") + "\n") },
  ]);
  await expect(research.getByText(/205 source rows/)).toBeVisible();
  await research.getByRole("combobox", { name: "Measurement value", exact: true }).fill("v");
  await research.getByRole("combobox", { name: "Measurement value", exact: true }).press("Tab");
  await research.getByRole("button", { name: "Preview preparation", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Observations: 205");
  await expect(research.getByTestId("research-mean")).toContainText("102");
  await expect(research.getByRole("button", { name: "Next page", exact: true })).toBeEnabled();
  await research.getByRole("button", { name: "Next page", exact: true }).focus();
  await research.getByRole("button", { name: "Next page", exact: true }).press("Enter");
  await expect(research.getByText("101–200 / 205", { exact: true })).toBeVisible();
  await expect(research.getByRole("button", { name: "Next page", exact: true })).toBeEnabled();
  await research.getByRole("button", { name: "Next page", exact: true }).focus();
  await research.getByRole("button", { name: "Next page", exact: true }).press("Enter");
  await expect(research.getByText("201–205 / 205", { exact: true })).toBeVisible();
  const table = research.getByRole("table", { name: "Research observations" });
  await expect(table.getByRole("row")).toHaveCount(6);
  await expect(table).toContainText("204");
});
