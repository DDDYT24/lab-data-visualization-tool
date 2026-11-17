import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, test } from "./fixtures";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires isolated local API.");

test("figure layouts preserve saved panels and recover drafts", async ({ page, expectConsoleError }, testInfo) => {
  // Five widths, two languages and accessibility scans: about a minute here, more on a busy computer.
  test.setTimeout(180_000);
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/api/v1/local/session"));
  await page.goto(`/#labviz-access=${process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY}`);
  expect((await unlocked).ok()).toBeTruthy();
  const rows = ["sample,value,unit,time,pair,group"];
  for (let s = 1; s <= 6; s++) {
    rows.push(`s${s},${s},mV,before,p${s},A`, `s${s},${s + 2},mV,after,p${s},A`);
  }
  const source = { name: "paired-chart.csv", mimeType: "text/csv", buffer: Buffer.from(rows.join("\n") + "\n") };
  await page.locator('input[type="file"]').setInputFiles(source);
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  const research = page.getByTestId("research-workspace");
  await expect(research.getByText("Saved state", { exact: true })).toBeVisible();
  await research.locator('input[type="file"]').first().setInputFiles(source);
  await expect(research.getByText(/12 source rows/)).toBeVisible();
  await research.getByRole("button", { name: "More metadata, limits and experimental records" }).click();
  for (const [label, field] of [
    ["Measurement value", "value"], ["Sample ID", "sample"],
    ["Independent experimental unit", "sample"], ["Unit column", "unit"],
    ["Group", "group"], ["Time point", "time"], ["Pair key", "pair"],
  ]) {
    await research.getByRole("combobox", { name: label, exact: true }).fill(field);
    await research.getByRole("combobox", { name: label, exact: true }).press("Tab");
  }
  await research.getByRole("button", { name: "Preview preparation", exact: true }).click();
  await expect(research.getByTestId("research-counts")).toContainText("Observations: 12");
  await research.getByRole("button", { name: "Apply preview and save revision" }).click();
  await page.getByRole("tab", { name: "Analysis", exact: true }).click();
  const charts = page.getByTestId("research-charts");
  await expect(charts.getByRole("button", { name: "Generate and save chart" })).toBeEnabled();
  for (const [family, title] of [["Grouped raw points", "Raw layout"], ["Violin distribution", "Distribution layout"]]) {
    await charts.getByRole("combobox", { name: "Chart family", exact: true }).click();
    await page.getByRole("option", { name: family, exact: true }).click();
    await charts.getByRole("textbox", { name: "Chart title", exact: true }).fill(title);
    await charts.getByRole("button", { name: "Generate and save chart" }).click();
    await expect(charts.getByRole("img", { name: title, exact: true })).toBeVisible();
  }
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  const board = page.getByTestId("figure-workspace");
  const select = async (label: string, option: string | RegExp) => {
    const control = board.getByRole("combobox", { name: label, exact: true });
    // A click on a disabled selector is ignored, and the layout selector is disabled while the
    // workspace loads; the option then never appears.
    await expect(control).toBeEnabled();
    await control.click();
    await page.getByRole("option", { name: option, exact: typeof option === "string" }).click();
  };
  const save = async (title: string) => {
    await board.getByRole("textbox", { name: "Figure title", exact: true }).fill(title);
    const committed = page.waitForResponse((response) => response.request().method() === "POST"
      && response.url().endsWith("/research/figure-boards?deferArtifacts=true"));
    await board.getByRole("button", { name: "Save new layout revision" }).click();
    const response = await committed;
    expect(response.status()).toBe(202);
    const revision = await response.json();
    expect(revision.artifactState).toBe("queued");
    expect(revision.composed).toBe(false);
    const reopened = await page.request.get(response.url().split("?")[0] + "/" + revision.id);
    expect((await reopened.json()).spec.title).toBe(title);
    await expect(board.getByTestId("figure-save-state")).toHaveText("Saved locally");
    await expect(board.getByRole("img", { name: "Saved composed figure", exact: true })).toBeVisible();
  };
  await select("Chart for panel A", /Raw layout/);
  await save("One panel");
  await board.getByRole("button", { name: "Duplicate panel A", exact: true }).click();
  await expect(board.getByTestId("figure-panel-B").getByRole("img")).toHaveAttribute("alt", "B: Raw layout");
  await board.getByRole("button", { name: "Clear panel B", exact: true }).click();
  await expect(board.getByTestId("figure-panel-B").getByRole("img")).toHaveCount(0);
  await board.getByRole("button", { name: "Undo layout edit", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(board.getByTestId("figure-panel-B").getByRole("img")).toHaveAttribute("alt", "B: Raw layout");
  await board.getByRole("button", { name: "Redo layout edit", exact: true }).click();
  await expect(board.getByTestId("figure-panel-B").getByRole("img")).toHaveCount(0);
  await select("Chart for panel B", /Distribution layout/);
  await save("Two panels");
  await expect(board.getByTestId("figure-panel-A").getByRole("img")).toHaveCount(0);
  await board.getByRole("button", { name: "Show source chart thumbnails", exact: true }).click();
  const imageA = board.getByTestId("figure-panel-A").getByRole("img");
  const originalSrc = await imageA.getAttribute("src");
  const originalBytes = await (await page.request.get(originalSrc!)).body();
  await board.getByRole("button", { name: "Move panel A later" }).focus();
  await page.keyboard.press("Enter");
  await expect(imageA).toHaveAttribute("alt", "A: Distribution layout");
  await page.getByRole("tab", { name: "Data", exact: true }).click();
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  await expect(imageA).toHaveAttribute("alt", "A: Distribution layout");
  await save("Reordered panels");
  await select("Panel count", "4");
  await select("Chart for panel C", /Raw layout/);
  await select("Chart for panel D", /Distribution layout/);
  await select("Panel count", "2");
  await select("Panel count", "4");
  await expect(board.getByTestId("figure-panel-C").getByRole("img")).toHaveAttribute("alt", "C: Raw layout");
  await save("Four panels");
  for (const label of "ABCD") {
    const img = board.getByTestId(`figure-panel-${label}`).getByRole("img");
    await expect.poll(() => img.evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  }
  await expect.poll(() => board.getByRole("img", { name: "Saved composed figure", exact: true })
    .evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await expect.poll(() => board.getByTestId("figure-panel-A").getByRole("img")
    .evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBe(480);
  await board.getByRole("button", { name: "Hide source chart thumbnails", exact: true }).click();
  await expect(board.getByTestId("figure-panel-A").getByRole("img")).toHaveCount(0);
  for (const width of [1440, 1280, 1024, 720, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    await board.screenshot({ path: testInfo.outputPath(`phase5-figure-${width}.png`) });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: testInfo.outputPath(`phase5-workspace-${width}.png`) });
    if (width === 390) {
      await board.getByRole("button", { name: "Selected panel settings", exact: true }).click();
      await expect(page.getByRole("button", { name: "Return to figure", exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Return to figure", exact: true }).click();
      await page.getByRole("button", { name: "Project list", exact: true }).click();
      await expect(page.getByRole("button", { name: "Return to active task", exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Return to active task", exact: true }).click();
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  const accessibility = await new AxeBuilder({ page }).include('[data-testid="research-workspace"]').withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze();
  await testInfo.attach("phase5-axe", { body: JSON.stringify(accessibility), contentType: "application/json" });
  expect(accessibility.violations).toEqual([]);
  await page.getByRole("button", { name: "Switch to dark theme", exact: true }).click();
  const darkAccessibility = await new AxeBuilder({ page }).include('[data-testid="research-workspace"]').withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze();
  expect(darkAccessibility.violations).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("phase5-workspace-dark.png") });
  await page.getByRole("button", { name: "Switch to light theme", exact: true }).click();
  await page.emulateMedia({ reducedMotion: "reduce" });
  await board.screenshot({ path: testInfo.outputPath("figure-desktop.png") });
  await page.reload();
  await expect(page.getByRole("button", { name: "Back to quick plot", exact: true })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Figures", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".MuiDataGrid-root")).toHaveCount(0);
  await expect(board.getByTestId("figure-panel-D")).toBeVisible();
  await select("Saved layouts", /Two panels/);
  await board.getByRole("button", { name: "Show source chart thumbnails", exact: true }).click();
  await expect(imageA).toHaveAttribute("src", originalSrc!);
  expect(await (await page.request.get(originalSrc!)).body()).toEqual(originalBytes);
  await expect(board.getByTestId("figure-panel-D")).toHaveCount(0);
  await board.getByRole("button", { name: "Figure appearance and shared axes", exact: true }).click();
  await select("Axis sharing", "Shared X and Y");
  await select("Figure palette", "Grayscale");
  await select("Point layer export", "300 dpi point layer");
  await board.getByRole("textbox", { name: "Panel A title (figure only)", exact: true }).fill("Panel-only title");
  await save("Shared scientific figure");
  const inspector = board.getByTestId("figure-inspector");
  const composed = inspector.getByRole("img", { name: "Saved composed figure" });
  const svgDownload = await inspector.getByRole("link", { name: "Download figure SVG", exact: true }).getAttribute("href");
  const hybridSvg = await (await page.request.get(svgDownload!)).text();
  expect(hybridSvg).toContain("<image");
  expect(hybridSvg).toContain("Panel-only title");
  await expect.poll(() => composed.evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  const scene = await (await page.request.get(svgDownload!.replace(/artifact\/svg.*$/, "interaction"))).json();
  const planHashes = scene.panels.map((panel: { planHash: string | null }) => panel.planHash).filter(Boolean);
  expect(planHashes.length).toBeGreaterThan(0);
  await inspector.getByRole("button", { name: "Interactive panels", exact: true }).click();
  const panelCharts = inspector.getByTestId("render-plan-chart");
  await expect(panelCharts).toHaveCount(planHashes.length);
  await expect(panelCharts.first().locator("canvas").first()).toBeVisible();
  expect(await panelCharts.evaluateAll((items) => items.map((item) => item.getAttribute("data-plan-hash")))).toEqual(planHashes);
  await inspector.getByRole("button", { name: "Composed figure", exact: true }).click();
  await expect(composed).toBeVisible();
  // Choose the uppermost visible point; same-height targets can occlude one another.
  // Keyboard/table paths below cover selecting inputs independently of pointer overlap.
  {
    const target = await inspector.locator('circle[data-testid^="figure-mark-A:"][r="5"]')
      .evaluateAll((points) => points.reduce((best, next) =>
        Number(next.getAttribute("cy")) <= Number(best.getAttribute("cy")) ? next : best)
        .getAttribute("data-testid")!);
    await inspector.getByTestId(target).click();
  }
  await expect(inspector.getByRole("status")).toContainText("Selected inputs: 1");
  const sourceTable = inspector.getByRole("table", { name: "Figure source observations" });
  await expect(sourceTable.getByRole("row")).toHaveCount(5);
  await expect(inspector.locator('circle[stroke="#8d5000"]')).not.toHaveCount(0);
  await page.getByRole("tab", { name: "Quality", exact: true }).click();
  await expect(inspector).toHaveCount(0);
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  await expect(inspector.getByRole("status")).toContainText("Selected inputs: 1");
  await expect(sourceTable.getByRole("row")).toHaveCount(5);

  await inspector.getByRole("button", { name: "Reverse source order" }).click();
  await expect(sourceTable.getByRole("row")).toHaveCount(5);
  await inspector.getByRole("textbox", { name: "Filter source table" }).fill("no-match-at-all");
  await expect(sourceTable.getByRole("row")).toHaveCount(1);
  await inspector.getByRole("textbox", { name: "Filter source table" }).fill("");
  await inspector.getByRole("button", { name: "Clear highlight" }).click();
  await inspector.getByRole("combobox", { name: "Select a mark (keyboard)" }).fill("density");
  await page.getByRole("option", { name: /density.*Density input subset/ }).click();
  await expect(inspector.getByText(/This aggregate\/density mark/)).toBeVisible();
  await expect(inspector.getByRole("status")).toContainText("Selected inputs: 12");
  for (const format of ["SVG", "PNG", "PDF"]) {
    const href = await inspector.getByRole("link", { name: `Download figure ${format}`, exact: true }).getAttribute("href");
    const artifact = await page.request.get(href!);
    expect(artifact.ok()).toBeTruthy();
    const bytes = await artifact.body();
    expect(bytes.length).toBeGreaterThan(1000);
    if (format === "PNG") expect(bytes.subarray(1, 4).toString()).toBe("PNG");
    if (format === "PDF") expect(bytes.subarray(0, 4).toString()).toBe("%PDF");
  }
  await inspector.screenshot({ path: testInfo.outputPath("figure-linked-desktop.png") });
  const projectId = new URL(page.url()).pathname.split("/").at(-1)!;
  const workspace = await (await page.request.get(`/api/v1/projects/${projectId}/workspace`)).json();
  const mainChart = { ...workspace.chart, type: "scatter", title: "Legacy captured", panelCount: 1,
    xAxis: { field: "sample", title: "Sample", unit: "" }, yAxis: { field: "value", title: "Value", unit: "mV" },
    series: [{ field: "value", label: "value", color: "#1768ac", panel: 1, yAxis: "primary", lineStyle: "solid" }] };
  expect((await page.request.put(`/api/v1/projects/${projectId}/chart`, { data: { chart: mainChart } })).ok()).toBeTruthy();
  await board.getByRole("button", { name: "Capture from Quick plot (optional)", exact: true }).click();
  await board.getByRole("textbox", { name: "Sample column for cross-workspace linking (optional)", exact: true }).fill("sample");
  const capturedResponse = page.waitForResponse((r) => r.url().includes("/research/figure-sources?panel="));
  await board.getByRole("button", { name: "Capture saved main-workspace chart" }).click();
  expect((await capturedResponse).ok()).toBeTruthy();
  await expect(board.getByRole("button", { name: "Capture saved main-workspace chart" })).toBeEnabled();
  await select("Chart for panel A", /Legacy captured/);
  await select("Axis sharing", "Independent");
  await save("Mixed composed figure");
  await expect.poll(() => composed.evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await page.reload();
  await expect(page.getByRole("button", { name: "Back to quick plot", exact: true })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Figures", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".MuiDataGrid-root")).toHaveCount(0);
  await expect(board.getByRole("textbox", { name: "Figure title", exact: true })).toHaveValue("Mixed composed figure");
  await expect(inspector.getByTestId("figure-mark-A:0")).toBeVisible();
  // Choose the uppermost visible point; same-height targets can occlude one another.
  // Keyboard/table paths below cover selecting inputs independently of pointer overlap.
  {
    const target = await inspector.locator('circle[data-testid^="figure-mark-A:"][r="5"]')
      .evaluateAll((points) => points.reduce((best, next) =>
        Number(next.getAttribute("cy")) <= Number(best.getAttribute("cy")) ? next : best)
        .getAttribute("data-testid")!);
    await inspector.getByTestId(target).click();
  }
  await expect(inspector.getByRole("status")).toContainText("Selected inputs: 1");
  await expect(inspector.getByRole("table").getByRole("row")).toHaveCount(5);
  await inspector.screenshot({ path: testInfo.outputPath("figure-mixed-desktop.png") });
  await page.route("**/research/figure-boards?deferArtifacts=true", async (route) => {
    if (route.request().method() === "POST") await route.fulfill({ status: 422, contentType: "application/json", body: JSON.stringify({ message: "Injected save failure", code: "test-save-failure" }) });
    else await route.continue();
  });
  expectConsoleError(/422/);
  await board.getByRole("textbox", { name: "Figure title", exact: true }).fill("Keep draft");
  await board.getByRole("button", { name: "Save new layout revision" }).click();
  await expect(board.getByText(/Injected save failure/)).toBeVisible();
  await expect(board.getByRole("textbox", { name: "Figure title", exact: true })).toHaveValue("Keep draft");
  await expect(board.getByTestId("figure-save-state")).toHaveText("Save failed — draft retained");
  await board.getByRole("button", { name: "Discard draft and restore saved layout" }).click();
  await expect(board.getByRole("textbox", { name: "Figure title", exact: true })).toHaveValue("Mixed composed figure");
  // Simulate a layout-only Phase 3A read; upgrading must not require a dummy edit.
  await page.unroute("**/research/figure-boards?deferArtifacts=true");
  const boardBase = `/api/v1/projects/${projectId}/research/figure-boards`;
  const latest = (await (await page.request.get(boardBase)).json()).boards[0].id;
  const legacyRead = "**/research/figure-boards/" + latest;
  await page.route(legacyRead, async (route) => {
    const response = await route.fetch();
    const revision = await response.json();
    await route.fulfill({ response, json: { ...revision, composed: false, artifactState: "unavailable" } });
  });
  await page.reload();
  await expect(page.getByRole("button", { name: "Back to quick plot", exact: true })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Figures", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".MuiDataGrid-root")).toHaveCount(0);
  await expect(board.getByTestId("figure-save-state")).toHaveText("Saved locally");
  await expect(board.getByRole("img", { name: "Saved composed figure", exact: true })).toHaveCount(0);
  await expect(board.getByRole("button", { name: "Save new layout revision" })).toBeEnabled();
  await page.unroute(legacyRead);
  await board.getByRole("button", { name: "Save new layout revision" }).click();
  await expect(board.getByRole("img", { name: "Saved composed figure", exact: true })).toBeVisible();
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文", exact: true }).click();
  await expect(board.getByRole("heading", { name: "组合图布局", exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  await inspector.getByRole("button", { name: "放大组合图阅读" }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  await inspector.screenshot({ path: testInfo.outputPath("figure-mobile-enlarged-zh.png") });
  await inspector.getByRole("button", { name: "组合图适应屏幕" }).click();
  await board.screenshot({ path: testInfo.outputPath("figure-mobile-zh.png") });
});

/** A project with an applied research revision, a raw-points chart and a violin, open on the Figures tab. */
async function openFigures(page: Page) {
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/api/v1/local/session"));
  await page.goto(`/#labviz-access=${process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY}`);
  expect((await unlocked).ok()).toBeTruthy();
  const rows = ["sample,group,value,unit"];
  for (let s = 1; s <= 8; s++) rows.push(`s${s},${s % 2 ? "Control" : "Treated"},${s * 3},mV`);
  const source = rows.join("\n") + "\n";
  const upload = await page.request.post("/api/v1/projects", { multipart: { file: { name: "figure.csv", mimeType: "text/csv", buffer: Buffer.from(source) } } });
  expect(upload.ok()).toBeTruthy();
  const uploaded = await upload.json();
  await expect.poll(async () => (await (await page.request.get(`/api/v1/jobs/${uploaded.job.id}`)).json()).stage).toBe("ready");
  const base = `/api/v1/projects/${uploaded.projectId}/research`;
  const prepared = await page.request.post(base + "/preview", { multipart: {
    plan: JSON.stringify({ mapping: { value: "value", sample: "sample", experimentalUnit: "sample", group: "group", unit: "unit" }, analysisUnit: "experimental_unit" }),
    files: { name: "figure.csv", mimeType: "text/csv", buffer: Buffer.from(source) },
  } });
  expect(prepared.ok(), await prepared.text()).toBeTruthy();
  const applied = await page.request.post(base + "/apply", { data: { token: (await prepared.json()).token } });
  expect(applied.ok()).toBeTruthy();
  const revisionId = (await applied.json()).revisionId;
  const charts: Record<string, string> = {};
  for (const [kind, title] of [["grouped_strip", "Points"], ["violin", "Spread"]]) {
    const created = await page.request.post(base + "/charts", { data: { revisionId, spec: { kind, title } } });
    expect(created.ok()).toBeTruthy();
    charts[title!] = (await created.json()).id;
  }
  await page.goto(`/workspace/${uploaded.projectId}`);
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  const select = async (label: string, option: string | RegExp) => {
    const control = page.getByRole("combobox", { name: label, exact: true });
    await expect(control).toBeEnabled();
    await control.click();
    await page.getByRole("option", { name: option, exact: typeof option === "string" }).click();
  };
  return { charts, select };
}

test("a six-panel mosaic with a log axis and a bracket saves, draws and reopens", async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 1440, height: 1000 });
  const { select } = await openFigures(page);
  const board = page.getByTestId("figure-workspace");
  await select("Panel count", "6");
  await board.getByRole("button", { name: "A across 2 columns, C down 2 rows, D down 2 rows", exact: true }).click();
  for (const [index, label] of [..."ABCDEF"].entries()) await select(`Chart for panel ${label}`, index % 2 ? /Spread/ : /Points/);
  await board.getByRole("button", { name: "Inspect panel A", exact: true }).click();
  const inspector = page.getByRole("complementary", { name: "Selected panel inspector" });
  await select("Inspector section", "Appearance");
  const editor = inspector.getByTestId("figure-panel-editor");
  await expect(editor.getByText("Categories", { exact: true })).toBeVisible();
  await select("Y axis scale", "Log 10");
  await editor.getByRole("button", { name: "Add annotation to panel A", exact: true }).click();
  await page.getByRole("menuitem", { name: "Significance bracket", exact: true }).click();
  const bracket = editor.getByRole("region", { name: "Significance bracket 1" });
  await expect(bracket.getByRole("combobox", { name: "X from", exact: true })).toHaveText("Control");
  await bracket.getByRole("textbox", { name: "Height", exact: true }).fill("20");
  await bracket.getByRole("textbox", { name: "Label", exact: true }).fill("p < 0.01");
  await expect(board.getByTestId("figure-panel-A")).toContainText("Own axis settings · 1 annotation");
  const accessibility = await new AxeBuilder({ page }).include('[data-testid="figure-workspace"]').withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze();
  expect(accessibility.violations).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("figure-six-panel-editor.png"), fullPage: true });
  const committed = page.waitForResponse((response) => response.request().method() === "POST"
    && response.url().endsWith("/research/figure-boards?deferArtifacts=true"));
  await board.getByRole("button", { name: "Save new layout revision" }).click();
  const response = await committed;
  expect(response.status(), await response.text()).toBe(202);
  const revision = await response.json();
  expect(revision.spec).toMatchObject({ schemaVersion: 2, layout: { kind: "mosaic", mosaic: ["AAB", "CDE", "CDF"] } });
  const url = response.url().split("?")[0] + "/" + revision.id;
  await expect.poll(async () => (await (await page.request.get(url)).json()).artifactState, { timeout: 180_000 }).toBe("ready");
  const plan = await (await page.request.get(url + "/plans/A")).json();
  expect(plan.panels[0].axes.y.scale).toBe("log10");
  expect(plan.panels[0].notes).toEqual([expect.objectContaining({ kind: "bracket", x: [0, 1], y: 20, label: "p < 0.01" })]);
  const composed = board.getByRole("img", { name: "Saved composed figure", exact: true });
  await expect.poll(() => composed.evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await board.getByRole("button", { name: "Interactive panels", exact: true }).click();
  await expect(board.getByTestId("render-plan-chart")).toHaveCount(6);
  await page.screenshot({ path: testInfo.outputPath("figure-six-panel-saved.png"), fullPage: true });
  await page.reload();
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Panel count", exact: true })).toHaveText("6");
  await expect(board.getByTestId("figure-panel-F")).toBeVisible();
  await expect(board.getByTestId("figure-panel-A")).toContainText("Own axis settings · 1 annotation");
});

test("a violin added under raw points saves as two layers in order and reopens", async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 1440, height: 1000 });
  const { charts, select } = await openFigures(page);
  const board = page.getByTestId("figure-workspace");
  await select("Chart for panel A", /Points/);
  await board.getByRole("button", { name: "Inspect panel A", exact: true }).click();
  const inspector = page.getByRole("complementary", { name: "Selected panel inspector" });
  await select("Inspector section", "Appearance");
  const layers = inspector.getByRole("list", { name: "Layers of panel A", exact: true });
  await expect(layers.getByRole("listitem")).toHaveText([/Points/]);
  await inspector.getByRole("button", { name: "Add layer to panel A", exact: true }).click();
  await page.getByRole("menuitem", { name: /Spread/ }).click();
  // Listed top first: a new layer goes on top, then moves under the points.
  await expect(layers.getByRole("listitem")).toHaveText([/Spread/, /Points/]);
  await layers.getByRole("button", { name: /^Move Spread .* down$/ }).click();
  await expect(layers.getByRole("listitem")).toHaveText([/Points/, /Spread/]);
  await expect(inspector.getByRole("button", { name: /^Remove layer Points/ })).toHaveCount(0);
  await expect(board.getByTestId("figure-panel-A")).toContainText("2 layers");
  const accessibility = await new AxeBuilder({ page }).include('[data-testid="figure-workspace"]').withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze();
  expect(accessibility.violations).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("figure-layers-editor.png"), fullPage: true });
  const committed = page.waitForResponse((response) => response.request().method() === "POST"
    && response.url().endsWith("/research/figure-boards?deferArtifacts=true"));
  await board.getByRole("button", { name: "Save new layout revision" }).click();
  const response = await committed;
  expect(response.status(), await response.text()).toBe(202);
  const revision = await response.json();
  // The panel's own chart is listed first; zIndex holds the drawing order.
  expect(revision.spec.panels[0].layers.map((layer: { chartId: string; style: { zIndex: number } }) => [layer.chartId, layer.style.zIndex]))
    .toEqual([[charts.Points, 1], [charts.Spread, 0]]);
  const url = response.url().split("?")[0] + "/" + revision.id;
  await expect.poll(async () => (await (await page.request.get(url)).json()).artifactState, { timeout: 180_000 }).toBe("ready");
  const marks: { kind: string; id: string; layer?: number }[] = (await (await page.request.get(url + "/plans/A")).json()).panels[0].marks;
  expect(marks[0]!.kind).toBe("violin");
  const top = marks.findIndex((mark) => mark.layer === 1);
  expect(top).toBeGreaterThan(0);
  expect(marks.slice(0, top).every((mark) => mark.layer === undefined)).toBe(true);
  expect(marks.slice(top).every((mark) => mark.layer === 1 && mark.id.startsWith("L1:"))).toBe(true);
  expect(marks.slice(top).some((mark) => mark.kind === "points")).toBe(true);
  await board.getByRole("button", { name: "Interactive panels", exact: true }).click();
  await expect(board.getByTestId("render-plan-chart")).toHaveCount(1);
  await page.screenshot({ path: testInfo.outputPath("figure-layers-saved.png"), fullPage: true });
  await page.reload();
  await page.getByRole("tab", { name: "Figures", exact: true }).click();
  await expect(board.getByTestId("figure-panel-A")).toContainText("2 layers");
  await board.getByRole("button", { name: "Inspect panel A", exact: true }).click();
  await select("Inspector section", "Appearance");
  await expect(inspector.getByRole("list", { name: "Layers of panel A", exact: true }).getByRole("listitem")).toHaveText([/Points/, /Spread/]);
});
