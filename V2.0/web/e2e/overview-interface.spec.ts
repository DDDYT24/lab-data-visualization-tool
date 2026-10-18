import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, test } from "./fixtures";
import { installMockApi } from "./support/mock-api";

async function visitSettled(page: Page, url: string) {
  await page.waitForLoadState("networkidle");
  await page.goto(url);
  await page.waitForLoadState("networkidle");
}

async function reloadSettled(page: Page) {
  await page.waitForLoadState("networkidle");
  await page.reload();
  await page.waitForLoadState("networkidle");
}

async function expectFitsPage(page: Page, width: number) {
  const overflow = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth,
    elements: [...document.querySelectorAll("body *")].filter((el) => el.getBoundingClientRect().right > innerWidth + 1).slice(0, 12)
      .map((el) => ({ tag: el.tagName, class: el.getAttribute("class"), text: el.textContent?.slice(0, 80), right: el.getBoundingClientRect().right })) }));
  expect(overflow.scroll, JSON.stringify(overflow)).toBeLessThanOrEqual(width + 1);
}

test("overview uses actual project summaries, survives mode changes and handles empty/error states", async ({ page, browserName, expectConsoleError }, testInfo) => {
  page.setDefaultTimeout(10_000);
  await installMockApi(page, { authenticated: true });
  await visitSettled(page, "/?ui=overview");
  await expect(page.getByRole("heading", { name: "From experimental data to clear insight" })).toBeVisible();
  const recent = page.getByTestId("overview-projects");
  await expect(recent.getByTestId("overview-project-card")).toHaveCount(1);
  await expect(recent).toContainText("Thermal response");
  await expect(recent).toContainText("experiment.csv");
  await expect(recent.getByRole("img")).toHaveCount(0);
  await page.getByRole("button", { name: "Switch to classic interface", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Make experiment data clear and insightful." })).toBeVisible();
  await reloadSettled(page);
  await expect(page.locator("html")).toHaveAttribute("data-labviz-interface", "classic");
  await page.getByRole("button", { name: "Switch to overview A interface", exact: true }).click();
  await reloadSettled(page);
  await expect(recent).toBeVisible();
  for (const width of [1440, 1280, 1024, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await expectFitsPage(page, width);
    await page.screenshot({ path: testInfo.outputPath(`overview-home-${width}.png`), fullPage: true });
  }
  await page.getByRole("button", { name: "Open navigation and account menu", exact: true }).click();
  await expect(page.getByRole("menuitem", { name: "Research workspace", exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("menuitem", { name: "Research workspace", exact: true })).not.toBeVisible();
  await expect(page.getByRole("menu", { includeHidden: true })).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 1000 });
  const axe = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  expect(axe.violations).toEqual([]);
  await page.route("**/api/v1/projects", (route) => route.fulfill({ json: { apiVersion: "v1", projects: [] } }));
  await reloadSettled(page);
  await expect(recent).toContainText("Import your first file");
  await page.route("**/api/v1/projects", (route) => route.fulfill({ status: 503, json: { error: { code: "unavailable", message: "Test unavailable" } } }));
  if (browserName !== "firefox") expectConsoleError(/503/);
  await reloadSettled(page);
  await expect(recent.getByRole("alert")).toContainText("Saved projects are unavailable");
  await page.route("**/api/v1/projects", (route) => route.fulfill({ json: { apiVersion: "v1", projects: [] } }));
  await recent.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(recent).toContainText("Import your first file");
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  await page.locator('input[type="file"]').setInputFiles({ name: "workbook.xlsx", mimeType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", buffer: Buffer.from("unsupported research input") });
  await expect(page.locator("main").getByRole("alert")).toContainText("Research preparation accepts CSV/TSV");
  await expect(page).toHaveURL(/workspace=research/);
  await visitSettled(page, "/settings");
  await page.getByRole("combobox", { name: "Interface mode", exact: true }).click();
  await page.getByRole("option", { name: "Classic", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-labviz-interface", "classic");
  await page.getByRole("combobox", { name: "Interface mode", exact: true }).click();
  await page.getByRole("option", { name: "Overview A", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-labviz-interface", "overview");
  await page.screenshot({ path: testInfo.outputPath("overview-settings.png"), fullPage: true });
  await page.unroute("**/api/v1/projects");
  await visitSettled(page, "/history");
  await expect(page.getByText("Thermal response", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Grid view", exact: true }).click();
  await expect(page.getByRole("button", { name: "Grid view", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "List view", exact: true }).click();
  await expect(page.getByRole("link", { name: "Open research workspace", exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("overview-project-list.png"), fullPage: true });
  await visitSettled(page, "/help");
  await expect(page.locator("main").getByRole("heading", { level: 1 })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("overview-help.png"), fullPage: true });
  await visitSettled(page, "/about");
  await expect(page.locator("main").getByRole("heading", { level: 1 })).toBeVisible();
});

test("overview preserves research drafts, real counts, saved figures and exports across modes", async ({ page, browserName, expectConsoleError }, testInfo) => {
  page.setDefaultTimeout(10_000);
  test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires isolated local API.");
  const writes: string[] = [];
  page.on("request", (request) => { if (request.method() === "POST") writes.push(new URL(request.url()).pathname); });
  const unlocked = page.waitForResponse((response) => response.url().endsWith("/api/v1/local/session"));
  await page.goto(`/?ui=overview&workspace=research#labviz-access=${process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY}`);
  expect((await unlocked).ok()).toBeTruthy();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.locator('input[type="file"]').setInputFiles({ name: "overview-research.csv", mimeType: "text/csv", buffer: Buffer.from("sample,value,unit\ns1,1,mV\ns2,2,mV\ns3,3,mV\n") });
  const research = page.getByTestId("research-workspace");
  await research.getByRole("button", { name: /^Use the file just imported:/ }).click();
  await expect(research.getByText(/3 source rows/)).toBeVisible();
  const projectPath = new URL(page.url()).pathname;
  expect(projectPath).not.toBe("/workspace/new");
  for (const [label, field] of [["Measurement value", "value"], ["Sample ID", "sample"], ["Independent experimental unit", "sample"], ["Unit column", "unit"]]) {
    await research.getByRole("combobox", { name: label, exact: true }).fill(field);
    await research.getByRole("combobox", { name: label, exact: true }).press("Tab");
  }
  await page.getByRole("button", { name: "Switch to classic interface", exact: true }).click();
  await expect(research.getByRole("combobox", { name: "Measurement value", exact: true })).toHaveValue("value");
  await page.getByRole("button", { name: "Switch to overview A interface", exact: true }).click();
  await research.getByRole("button", { name: "Preview preparation", exact: true }).click();
  const contents = research.getByTestId("overview-project-contents");
  await expect(contents).toContainText("Independent units: 3");
  await expect(contents).toContainText("Observations: 3");
  await expect(contents).toContainText("Unsaved preview");
  await research.getByRole("button", { name: "Apply preview and save revision" }).click();
  await expect(contents).not.toContainText("Unsaved preview");
  await expect(contents).toContainText("overview-research.csv");
  await page.getByRole("tab", { name: "Analysis", exact: true }).click();
  const charts = page.getByTestId("research-charts");
  await charts.getByRole("textbox", { name: "Chart title", exact: true }).fill("Overview measured values");
  await charts.getByRole("button", { name: "Generate and save chart", exact: true }).click();
  await expect(charts.getByRole("img", { name: "Overview measured values", exact: true })).toBeVisible();
  await charts.getByRole("textbox", { name: "Chart title", exact: true }).fill("Overview alternate values");
  await charts.getByRole("button", { name: "Generate and save chart", exact: true }).click();
  await expect(charts.getByRole("img", { name: "Overview alternate values", exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  const board = page.getByTestId("figure-workspace");
  const chooseLayoutOption = async (label: string, option: string | RegExp) => {
    const control = board.getByRole("combobox", { name: label, exact: true, includeHidden: true });
    await control.focus();
    await control.press("ArrowDown");
    await expect(control).toHaveAttribute("aria-expanded", "true");
    const item = page.getByRole("option", { name: option, exact: typeof option === "string" });
    await item.focus();
    await item.press("Enter");
    await expect(control).toHaveAttribute("aria-expanded", "false");
    await expect(control).toContainText(option);
    await expect(page.getByRole("listbox")).toHaveCount(0);
    await expect(board.getByRole("combobox", { name: label, exact: true })).toBeVisible();
  };
  const editorToggle = board.getByRole("button", { name: /Edit draft and panel layout/ });
  await expect(editorToggle).toHaveAttribute("aria-expanded", "false");
  await expect(board.getByRole("heading", { name: "No saved preview yet", exact: true })).toBeVisible();
  await editorToggle.click();
  const title = board.getByRole("textbox", { name: "Figure title", exact: true });
  await title.fill("Overview draft preserved");
  await chooseLayoutOption("Chart for panel A", /Overview measured values/);
  await chooseLayoutOption("Panel count", "4");
  for (const label of "BCD") {
    await chooseLayoutOption(`Chart for panel ${label}`, /Overview measured values/);
  }
  await page.getByRole("button", { name: "Switch to classic interface", exact: true }).click();
  await expect(title).toHaveValue("Overview draft preserved");
  await page.getByRole("button", { name: "Switch to overview A interface", exact: true }).click();
  await expect(title).toHaveValue("Overview draft preserved");
  await page.getByRole("navigation", { name: "Primary navigation", exact: true }).getByRole("link", { name: "Quick plot", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Confirm your data", exact: true })).toBeVisible();
  await page.getByRole("navigation", { name: "Primary navigation", exact: true }).getByRole("link", { name: "Research workspace", exact: true }).click();
  await expect(title).toHaveValue("Overview draft preserved");
  await page.route("**/research/figure-boards/*/interaction?**", (route) => route.fulfill({ status: 503,
    json: { error: { code: "ui-test-source-failed", message: "Synthetic source lookup failure" } } }), { times: 1 });
  if (browserName !== "firefox") expectConsoleError(/503/);
  await board.getByRole("button", { name: "Save new layout revision", exact: true }).click();
  await expect(board.getByTestId("figure-save-state")).toHaveText("Saved locally");
  await board.getByRole("button", { name: "Retry source lookup", exact: true }).click();
  await expect(board.getByTestId("figure-inspector").getByRole("status")).toContainText("Matching panel rows: 12");
  await expect(board.getByRole("img", { name: "Saved composed figure", exact: true })).toBeVisible();
  const download = board.getByRole("link", { name: "Download figure SVG", exact: true });
  const svg = await page.request.get((await download.getAttribute("href"))!);
  expect(svg.ok()).toBeTruthy();
  const savedSvg = await svg.text();
  expect(savedSvg).toContain("Overview draft preserved");
  for (const format of ["PNG", "PDF"]) {
    const link = board.getByRole("link", { name: `Download figure ${format}`, exact: true });
    const file = await page.request.get((await link.getAttribute("href"))!);
    expect(file.ok()).toBeTruthy();
    const bytes = await file.body();
    expect(format === "PDF" ? bytes.subarray(0, 4).toString() === "%PDF" : bytes.subarray(1, 4).toString() === "PNG").toBe(true);
  }
  await page.getByRole("button", { name: "Switch to classic interface", exact: true }).click();
  await page.getByRole("button", { name: "Switch to overview A interface", exact: true }).click();
  expect(await (await page.request.get((await download.getAttribute("href"))!)).text()).toBe(savedSvg);
  // Draft disclosure and workflow switches must preserve both source selection and saved bytes.
  const dockToggle = board.getByRole("button", { name: /Linked source rows/ });
  await dockToggle.click();
  const mark = board.getByRole("combobox", { name: "Select a mark (keyboard)", exact: true });
  await mark.click();
  await page.getByRole("option").first().click();
  await expect(board.getByTestId("figure-inspector").getByRole("status").filter({ hasText: /Selected inputs: [1-9]/ })).toBeVisible();
  const selectedMark = await mark.inputValue();
  await title.fill("Draft excluded from saved figure");
  await chooseLayoutOption("Chart for panel A", /Overview alternate values/);
  const panelInspector = board.getByRole("complementary", { name: "Selected panel inspector", exact: true });
  await board.getByRole("button", { name: /^Panel A ·/ }).click();
  await expect(panelInspector).toContainText("Saved preview binding");
  await expect(panelInspector).toContainText("Source chart: Overview measured values");
  await board.getByRole("button", { name: "Inspect panel A", exact: true }).click();
  await expect(panelInspector).toContainText("Draft panel binding");
  await expect(panelInspector).toContainText("Source chart: Overview alternate values");
  await editorToggle.click();
  await expect(title).not.toBeVisible();
  await page.getByRole("tab", { name: "Quality", exact: true }).click();
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  await expect(mark).toHaveValue(selectedMark);
  await editorToggle.click();
  await expect(title).toHaveValue("Draft excluded from saved figure");
  expect(await (await page.request.get((await download.getAttribute("href"))!)).text()).toBe(savedSvg);
  await board.getByRole("button", { name: "Discard draft and restore saved layout", exact: true }).click();
  await editorToggle.click();
  await expect(title).not.toBeVisible();
  await dockToggle.click();
  await page.evaluate(() => scrollTo(0, document.documentElement.scrollHeight));
  const stickyStatus = page.getByTestId("figure-status-sticky");
  await expect(stickyStatus).toBeInViewport();
  expect((await stickyStatus.boundingBox())!.y).toBeLessThan(120);
  // Actual DOM order follows the visual canvas -> source dock -> editor order.
  expect(await page.evaluate(() => !!(document.querySelector('[data-testid="figure-inspector"]')!
    .compareDocumentPosition(document.querySelector('[data-testid="figure-draft-editor"]')!) & Node.DOCUMENT_POSITION_FOLLOWING))).toBe(true);
  // A failed save retains the draft and the last immutable export.
  await editorToggle.click();
  await title.fill("Overview revision after retry");
  await chooseLayoutOption("Chart for panel A", /Overview alternate values/);
  await page.route("**/research/figure-boards?deferArtifacts=true", async (route) => {
    await route.fulfill({ status: 503, json: { error: { code: "ui-test-save-failed", message: "Synthetic save failure" } } });
  }, { times: 1 });
  if (browserName !== "firefox") expectConsoleError(/503/);
  await board.getByRole("button", { name: "Save new layout revision", exact: true }).click();
  await expect(board.getByTestId("figure-save-state")).toHaveText("Save failed — draft retained");
  await expect(title).toHaveValue("Overview revision after retry");
  expect(await (await page.request.get((await download.getAttribute("href"))!)).text()).toBe(savedSvg);
  // Hold only the transport state of the new real API revision. Scientific inputs/results remain real.
  let heldId = "";
  let holdGeneration = true;
  let heldState = "running";
  await page.route(/\/research\/figure-boards\/[^/?]+$/, async (route) => {
    const response = await route.fetch();
    const json = await response.json();
    if (holdGeneration && json.id === heldId) await route.fulfill({ response, json: { ...json, composed: false, artifactState: heldState } });
    else await route.fulfill({ response });
  });
  const committed = page.waitForResponse((r) => r.request().method() === "POST" && r.url().endsWith("/research/figure-boards?deferArtifacts=true"));
  await board.getByRole("button", { name: "Save new layout revision", exact: true }).click();
  heldId = (await (await committed).json()).id;
  await expect(board.getByTestId("figure-status")).toContainText("Generating");
  await expect(board.getByTestId("figure-preview-revision")).toContainText("Previous saved preview");
  await expect(board.getByRole("img", { name: "Saved composed figure", exact: true })).toBeVisible();
  await expect(download).toHaveCount(0);
  await expect(board.getByRole("button", { name: "Download figure SVG", exact: true })).toBeDisabled();
  await expect(page.getByTestId("figure-status-compact").first()).toContainText("Generating");
  await expect(page.getByTestId("overview-project-contents").getByTestId("figure-status-compact")).toContainText("Generating");
  await board.getByRole("button", { name: /^Panel A ·/ }).click();
  await expect(panelInspector).toContainText("Source chart: Overview measured values");
  heldState = "failed";
  await expect(board.getByTestId("figure-status")).toContainText("Generation failed");
  let releaseRetry!: () => void;
  const retryGate = new Promise<void>((resolve) => { releaseRetry = resolve; });
  await page.route("**/research/figure-boards/*/retry", async (route) => {
    // The failed state above is transport-injected; return the actual ready
    // revision to cover a retry response that is already complete.
    const response = await page.request.get(route.request().url().replace(/\/retry$/, ""));
    await retryGate;
    await route.fulfill({ status: 202, json: await response.json() });
  }, { times: 1 });
  holdGeneration = false;
  await board.getByRole("button", { name: "Retry figure generation", exact: true }).click();
  await expect(board.getByTestId("figure-status")).toContainText("Retrying");
  releaseRetry();
  await expect(board.getByTestId("figure-status")).toContainText("Files ready");
  await expect(panelInspector).toContainText("Source chart: Overview alternate values");
  await expect(download).toBeVisible();
  await editorToggle.click();
  expect(writes.filter((path) => path === "/api/v1/projects")).toHaveLength(1);
  expect(writes.filter((path) => path.endsWith("/research/apply"))).toHaveLength(1);
  for (const width of [1440, 1280, 1024, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await expectFitsPage(page, width);
    if (width < 1400) {
      const opener = board.getByRole("button", { name: "Selected panel settings", exact: true });
      await opener.click();
      const dialog = page.getByRole("dialog", { name: "Selected panel inspector", exact: true });
      await expect(dialog).toBeVisible();
      await dialog.getByRole("button", { name: "Return to figure", exact: true }).focus();
      await page.keyboard.press("Shift+Tab");
      expect(await dialog.evaluate((el) => el.contains(document.activeElement))).toBe(true);
      await expect(page.getByRole("tab", { name: "Panel method", exact: true })).toBeVisible();
      await page.getByRole("tab", { name: "Panel appearance", exact: true }).click();
      await page.getByRole("button", { name: "Return to figure", exact: true }).click();
      await expect(page.getByRole("button", { name: "Return to figure", exact: true })).not.toBeVisible();
      await expect(opener).toBeFocused();
      await opener.click();
      await page.keyboard.press("Escape");
      await expect(dialog).not.toBeVisible();
      await expect(opener).toBeFocused();
    }
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({ path: testInfo.outputPath(`overview-figure-${width}.png`), fullPage: true });
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  for (const mode of ["light", "dark"]) {
    if (mode === "dark") await page.getByRole("button", { name: "Switch to dark theme", exact: true }).click();
    const axe = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
    expect(axe.violations).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath(`overview-figure-${mode}.png`), fullPage: true });
  }
  const historyLoads: string[] = [];
  const countHistory = (request: import("@playwright/test").Request) => {
    if (request.method() === "GET" && /\/research\/figure-boards$/.test(new URL(request.url()).pathname)) historyLoads.push(request.url());
  };
  page.on("request", countHistory);
  await page.reload();
  await expect(page.getByRole("tab", { name: "Figures", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByTestId("figure-inspector").getByRole("img", { name: "Saved composed figure", exact: true })).toBeVisible();
  expect(historyLoads).toHaveLength(1);
  page.off("request", countHistory);
  await expect(editorToggle).toHaveAttribute("aria-expanded", "false");
  await editorToggle.click();
  await expect(title).toHaveValue("Overview revision after retry");
  await editorToggle.click();
  let releaseCharts!: () => void;
  let chartsStarted!: () => void;
  const chartsGate = new Promise<void>((resolve) => { releaseCharts = resolve; });
  const chartsRequested = new Promise<void>((resolve) => { chartsStarted = resolve; });
  const chartListPattern = /\/research\/charts$/;
  await page.route(chartListPattern, async (route) => { chartsStarted(); await chartsGate; await route.continue(); });
  await page.getByRole("tab", { name: "Analysis", exact: true }).click();
  await chartsRequested;
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  const staleDetails: string[] = [];
  const countDetails = (request: import("@playwright/test").Request) => {
    if (request.method() === "GET" && /\/research\/charts\/[^/]+$/.test(new URL(request.url()).pathname)) staleDetails.push(request.url());
  };
  page.on("request", countDetails);
  const listFinished = page.waitForResponse((response) => chartListPattern.test(new URL(response.url()).pathname));
  releaseCharts();
  await (await listFinished).finished();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  expect(staleDetails).toEqual([]);
  page.off("request", countDetails);
  await page.unroute(chartListPattern);
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文", exact: true }).click();
  await expect(page.getByRole("tab", { name: "图表", exact: true })).toHaveAttribute("aria-selected", "true");
  await page.screenshot({ path: testInfo.outputPath("overview-figure-zh-dark.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "项目内容", exact: true }).click();
  await expect(page.getByRole("heading", { name: "项目内容", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "返回当前任务", exact: true }).click();
  await expect(page.getByRole("button", { name: "返回当前任务", exact: true })).not.toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("overview-figure-zh-mobile.png"), fullPage: true });
  await page.goto("/");
  const createdCard = page.locator(`[data-testid="overview-project-card"][href="${projectPath}"]`);
  await expect(createdCard).toBeVisible();
  await expect(createdCard).toContainText("overview-research.csv");
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({ path: testInfo.outputPath("overview-home-zh-dark-real.png"), fullPage: true });
  await page.getByRole("button", { name: "切换到浅色主题", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-labviz-theme", "light");
  await page.screenshot({ path: testInfo.outputPath("overview-home-zh-light-real.png"), fullPage: true });
});
