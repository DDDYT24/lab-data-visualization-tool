import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

function contrastRatio(foreground: string, background: string) {
  const parse = (value: string) => {
    const channels = value.match(/\d+(?:\.\d+)?/g)?.slice(0, 3).map(Number) ?? [];
    return channels.map((channel) => {
      const normalized = channel / 255;
      return normalized <= 0.03928
        ? normalized / 12.92
        : ((normalized + 0.055) / 1.055) ** 2.4;
    });
  };
  const [fr, fg, fb] = parse(foreground);
  const [br, bg, bb] = parse(background);
  const foregroundLuminance = 0.2126 * fr + 0.7152 * fg + 0.0722 * fb;
  const backgroundLuminance = 0.2126 * br + 0.7152 * bg + 0.0722 * bb;
  const lighter = Math.max(foregroundLuminance, backgroundLuminance);
  const darker = Math.min(foregroundLuminance, backgroundLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

test("persists an explicit dark theme across navigation", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/settings");

  await page.getByRole("combobox", { name: "Appearance" }).click();
  await page.getByRole("option", { name: "Dark" }).click();

  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "dark");
  await expect.poll(() =>
    page.evaluate(() => {
      const value = window.localStorage.getItem("labviz:user-preferences:v1");
      return value ? JSON.parse(value).appearance : null;
    }),
  ).toBe("dark");

  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "dark");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
});

test("switches between light and dark directly from the application header", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/");

  const switchToDark = page.getByRole("button", { name: "Switch to dark theme" });
  await expect(switchToDark).toBeVisible();
  await switchToDark.click();
  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "dark");
  await expect(page.locator('svg[aria-label="LabViz"] text')).toHaveAttribute(
    "fill", "#ECECEC",
  );
  await expect(page.locator('[data-logo-part="flask-outline"]')).toHaveAttribute(
    "stroke", "#ECECEC",
  );
  await expect(page.locator('[data-logo-part="flask-level"]')).toHaveAttribute(
    "stroke", "#ECECEC",
  );
  await expect(page.locator('[data-logo-part="waveform"]')).toHaveAttribute(
    "stroke", "#ECECEC",
  );
  await expect(page.locator('[data-logo-part="flask-gradient-start"]')).toHaveAttribute(
    "stop-color", "#ECECEC",
  );
  await expect(page.locator('[data-logo-part="flask-gradient-end"]')).toHaveAttribute(
    "stop-color", "#ECECEC",
  );

  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "light");
  await expect(page.locator('svg[aria-label="LabViz"] text')).toHaveAttribute(
    "fill", "#172033",
  );
  await expect(page.locator('[data-logo-part="flask-outline"]')).toHaveAttribute(
    "stroke", "#2563EB",
  );
  await expect(page.locator('[data-logo-part="waveform"]')).toHaveAttribute(
    "stroke", "#0F766E",
  );
});

test("follows the operating-system color scheme when System is selected", async ({ page }) => {
  await installMockApi(page);
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/settings");

  const appearance = page.getByRole("combobox", { name: "Appearance" });
  await appearance.click();
  await page.getByRole("option", { name: "System" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "dark");

  await page.emulateMedia({ colorScheme: "light" });
  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "light");
});

test("persists every figure default and opens both privacy destinations", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/settings");

  for (const [label, option] of [
    ["Figure text", "English"],
    ["Font", "Times New Roman"],
    ["Figure size", "Custom size"],
    ["Unit", "cm"],
    ["DPI", "600"],
  ]) {
    await page.getByRole("combobox", { name: label }).click();
    await page.getByRole("option", { name: option, exact: true }).click();
  }
  await page.getByRole("switch", { name: "Enable" }).click();

  await expect.poll(() =>
    page.evaluate(() => JSON.parse(window.localStorage.getItem("labviz:user-preferences:v1") ?? "{}")),
  ).toMatchObject({
    figureLanguage: "en",
    fontFamily: "Times New Roman",
    sizePreset: "custom",
    unit: "cm",
    dpi: 600,
    grayscalePreview: true,
  });

  await page.reload();
  await expect(page.getByRole("combobox", { name: "Figure text" })).toContainText("English");
  await expect(page.getByRole("combobox", { name: "Font" })).toContainText("Times New Roman");
  await expect(page.getByRole("combobox", { name: "Figure size" })).toContainText("Custom size");
  await expect(page.getByRole("combobox", { name: "Unit" })).toContainText("cm");
  await expect(page.getByRole("combobox", { name: "DPI" })).toContainText("600");
  await expect(page.getByRole("switch", { name: "Enable" })).toBeChecked();

  await page.getByRole("link", { name: "View local projects" }).click();
  await expect(page).toHaveURL(/\/history$/);
  await page.goto("/settings");
  await page.getByRole("link", { name: "Retention details" }).click();
  await expect(page).toHaveURL(/\/help$/);
});

test("keeps dark surfaces and primary actions readable", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/settings");
  await page.getByRole("combobox", { name: "Appearance" }).click();
  await page.getByRole("option", { name: "Dark" }).click();
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "dark");

  const actionButton = page.getByRole("button", { name: "Choose a file" });
  const readColors = () => actionButton.evaluate((action) => {
    const actionStyle = getComputedStyle(action);
    return {
      body: getComputedStyle(document.body).backgroundColor,
      actionBackground: actionStyle.backgroundColor,
      actionForeground: actionStyle.color,
    };
  });

  await expect.poll(async () => {
    const colors = await readColors();
    return contrastRatio(colors.actionForeground, colors.actionBackground);
  }).toBeGreaterThanOrEqual(4.5);

  const colors = await readColors();
  expect(colors.body).toBe("rgb(33, 33, 33)");
  expect(contrastRatio(colors.actionForeground, colors.actionBackground)).toBeGreaterThanOrEqual(
    4.5,
  );
});
