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

test("keeps dark surfaces and primary actions readable", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/settings");
  await page.getByRole("combobox", { name: "Appearance" }).click();
  await page.getByRole("option", { name: "Dark" }).click();
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "dark");

  const colors = await page.evaluate(() => {
    const action = document.querySelector('main button[variant="contained"]') ??
      document.querySelector("main button");
    const actionStyle = action ? getComputedStyle(action) : null;
    return {
      body: getComputedStyle(document.body).backgroundColor,
      actionBackground: actionStyle?.backgroundColor ?? "",
      actionForeground: actionStyle?.color ?? "",
    };
  });

  expect(colors.body).toBe("rgb(15, 23, 42)");
  expect(contrastRatio(colors.actionForeground, colors.actionBackground)).toBeGreaterThanOrEqual(
    4.5,
  );
});
