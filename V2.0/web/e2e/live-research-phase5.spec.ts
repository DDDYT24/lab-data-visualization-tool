import AxeBuilder from "@axe-core/playwright";
import { figureMarkAt } from "../src/domain/figure-point-hit";
import { expect, test } from "./fixtures";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires isolated local API.");
test("cancelled preview and failed apply preserve the preparation draft", async ({ page, expectConsoleError }) => {
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/api/v1/local/session"));
  await page.goto(`/#labviz-access=${process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY}`);
  expect((await unlocked).ok()).toBeTruthy();
  const source = { name: "phase5.csv", mimeType: "text/csv", buffer: Buffer.from("v\n1\n2\n3\n") };
  await page.locator('input[type="file"]').setInputFiles(source);
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  const research = page.getByTestId("research-workspace");
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  await research.locator('input[type="file"]').first().setInputFiles(source);
  await expect(research.getByText(/3 source rows/)).toBeVisible();
  const value = research.getByRole("combobox", { name: "Measurement value", exact: true });
  await value.fill("v"); await value.press("Tab");
  await research.getByRole("button", { name: "More metadata, limits and experimental records", exact: true }).click();
  const unit = research.getByRole("textbox", { name: "Unit when no unit column is mapped", exact: true });
  await unit.fill("mV");
  await page.getByRole("tab", { name: "Quality", exact: true }).click();
  await expect(unit).toHaveCount(0);
  await page.getByRole("tab", { name: "Data", exact: true }).click();
  await expect(unit).toBeVisible();
  await expect(unit).toHaveValue("mV");

  let release!: () => void; let started!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const intercepted = new Promise<void>((resolve) => { started = resolve; });
  await page.route("**/research/preview", async (route) => {
    started(); await gate; await route.abort("aborted").catch(() => {});
  });
  await research.getByRole("button", { name: "Preview preparation", exact: true }).click();
  await intercepted;
  await research.getByRole("button", { name: "Cancel preparation preview", exact: true }).click();
  release();
  await expect(research.getByText("Preview cancelled — draft retained", { exact: true })).toBeVisible();
  await research.getByRole("button", { name: "Save and computation details", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Save and computation details", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Return to active task", exact: true }).click();
  await expect(value).toHaveValue("v");
  await page.unroute("**/research/preview");
  const projectId = new URL(page.url()).pathname.split("/").at(-1)!;
  const statePath = `/api/v1/projects/${projectId}/research`;
  const stateUrl = process.env.LABVIZ_API_PROXY_TARGET ? new URL(statePath, process.env.LABVIZ_API_PROXY_TARGET).href : statePath;
  expect((await (await page.request.get(stateUrl, { timeout: 10_000, headers: { Connection: "close" } })).json()).revisionId).toBeNull();
  await research.getByRole("button", { name: "Preview preparation", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Observations: 3");
  await page.route("**/research/apply", (route) => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ message: "Injected apply failure", code: "test-apply-failure" }) }));
  expectConsoleError(/503/);
  await research.getByRole("button", { name: "Apply preview and save revision", exact: true }).click();
  await expect(research.getByRole("alert").filter({ hasText: "Injected apply failure" })).toBeVisible();
  expect((await (await page.request.get(stateUrl, { timeout: 10_000, headers: { Connection: "close" } })).json()).revisionId).toBeNull();
  await page.getByRole("tab", { name: "Data", exact: true }).click();
  await research.getByRole("button", { name: "Reload saved status", exact: true }).click();
  await expect(value).toHaveValue("v");
  await expect(research.getByRole("button", { name: "Discard preparation draft", exact: true })).toBeVisible();
  await page.unroute("**/research/apply");
  await page.getByRole("tab", { name: "Quality", exact: true }).click();
  await research.getByRole("button", { name: "Apply preview and save revision", exact: true }).click();
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  expect((await (await page.request.get(stateUrl, { timeout: 10_000, headers: { Connection: "close" } })).json()).revisionId).not.toBeNull();
  const sourceRows = research.locator("#research-region-quality tbody").last();
  await expect(sourceRows.locator("tr")).toHaveCount(3);
  await research.getByRole("textbox", { name: "Exclusion reason", exact: true }).fill("Retain my pending note");
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  await expect(sourceRows.locator("tr")).toHaveCount(0);
  await page.getByRole("tab", { name: "Quality", exact: true }).click();
  await expect(sourceRows.locator("tr")).toHaveCount(3);
  await expect(research.getByRole("textbox", { name: "Exclusion reason", exact: true })).toHaveValue("Retain my pending note");
});


test("quick plot table state and source data commands survive workspace switching", async ({ page, context }, testInfo) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.request.post("/api/v1/local/session", { headers: { "X-LabViz-Local-Key": process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY! } });
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({ name: "retained.csv", mimeType: "text/csv", buffer: Buffer.from("v\n" + Array.from({ length: 50 }, (_, i) => i + 1).join("\n") + "\n") });
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  const grid = page.getByRole("grid");
  await grid.getByRole("columnheader").filter({ hasText: "v" }).click();
  await page.getByRole("button", { name: "Go to next page" }).click();
  await expect(page.getByText("26–50 of 50", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  await expect(page.locator(".MuiDataGrid-root")).toHaveCount(0);
  const research = page.getByTestId("research-workspace");
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  await research.getByRole("button", { name: "Save and computation details", exact: true }).click();
  const tools = page.getByTestId("source-data-tools");
  await expect(tools.getByRole("textbox", { name: "SQLite database", exact: true })).toHaveValue(/v23-phase5.*labviz.db$/);
  await tools.getByRole("button", { name: "Copy backup command", exact: true }).click();
  await expect(tools.getByRole("textbox", { name: "PowerShell command", exact: true })).toHaveValue(/-Action Backup -DataDirectory/);
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain("-Action Backup");
  await tools.getByRole("textbox", { name: "Existing backup folder", exact: true }).fill(String.raw`C:\fixture\researcher's backup`);
  await tools.getByRole("button", { name: "Copy restore command", exact: true }).click();
  await expect(tools.getByRole("textbox", { name: "PowerShell command", exact: true })).toHaveValue(/researcher''s backup/);
  expect((await new AxeBuilder({ page }).include('[data-testid="source-data-tools"]').withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze()).violations).toEqual([]);
  await page.getByRole("heading", { name: "Save and computation details", exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("source-data-tools-desktop.png") });
  await tools.getByText(/Research output ZIPs contain selected outputs/).scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("source-data-tools-desktop-bottom.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  await page.getByRole("heading", { name: "Save and computation details", exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("source-data-tools-mobile.png") });
  await tools.getByText(/Research output ZIPs contain selected outputs/).scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("source-data-tools-mobile-bottom.png") });
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.getByRole("button", { name: "Return to active task", exact: true }).click();
  await page.getByRole("button", { name: "Back to quick plot", exact: true }).click();
  await expect(grid).toBeVisible();
  await expect(page.getByText("26–50 of 50", { exact: true })).toBeVisible();
  await expect(grid.getByRole("columnheader").filter({ hasText: "v" })).toHaveAttribute("aria-sort", "ascending");
});

test("dense hit detection matches native SVG stroke and paint order", async ({ page }) => {
  const marks = [
    { id: "edge", kind: "point", cx: 0.3005, cy: 0.3005 },
    { id: "density", kind: "density", cx: 0.2, cy: 0.2 },
    { id: "top", kind: "point", cx: 0.2, cy: 0.2 },
  ].map((mark) => ({ ...mark, panel: "A", label: mark.id, inputCount: 1, highlighted: false }));
  await page.setContent(`<style>body{margin:0}svg{display:block}</style>
    <svg width="1000" height="1000" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="Native SVG hit oracle">
    ${marks.map((mark) => `<circle id="${mark.id}" cx="${mark.cx * 1000}" cy="${mark.cy * 1000}"
      r="${mark.kind === "point" ? 5 : 9}" fill="transparent" stroke="transparent" stroke-width="1.5" />`).join("")}
    </svg>`);
  const svg = page.getByLabel("Native SVG hit oracle");
  await svg.evaluate((element) => element.addEventListener("click", (event) => {
    const box = element.getBoundingClientRect();
    element.setAttribute("data-actual", JSON.stringify({
      x: ((event as MouseEvent).clientX - box.left) * 1000 / box.width,
      y: ((event as MouseEvent).clientY - box.top) * 1000 / box.height,
      id: (event.target as Element).closest("circle")?.id,
    }));
  }));
  for (const [x, y] of [[200, 200], [206, 200], [306, 300], [311, 300], [500, 500]]) {
    await svg.click({ position: { x, y } });
    const actual = JSON.parse((await svg.getAttribute("data-actual"))!);
    expect(figureMarkAt(marks, actual.x, actual.y)).toBe(actual.id);
  }
});
