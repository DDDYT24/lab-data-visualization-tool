import { readFile } from "node:fs/promises";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

import { expect, test } from "./fixtures";
import type { ChartAnalysis } from "../src/domain/api-contract";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires the real local API and launcher bootstrap.");

for (const filename of ["04_group_comparison.txt", "05_distribution.json"]) {
  test(`raw ${filename}: histogram intervals match the exported SVG`, async ({ page }) => {
    test.setTimeout(180_000);
    const source = path.resolve(__dirname, "../../api/samples/v22", filename);
    const raw = await readFile(source, "utf8");
    const values: number[] = filename.endsWith("json")
      ? JSON.parse(raw).map((row: { measurement: number }) => row.measurement)
      : raw.trim().split(/\r?\n/).slice(1).map((row) => Number(row.split("\t")[1]));
    const unlocked = page.waitForResponse((r) => r.url().endsWith("/api/v1/local/session") && r.request().method() === "POST");
    await page.goto(`/#labviz-access=${encodeURIComponent(process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY!)}`);
    expect((await unlocked).ok()).toBeTruthy();
    await page.locator('input[type="file"]').setInputFiles(source);
    await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({ timeout: 60_000 });
    await page.getByRole("button", { name: "Review data quality" }).click();
    const keep = page.getByRole("button", { name: "Keep all remaining as is" });
    if (await keep.isVisible()) await keep.click();
    await expect(page.getByRole("button", { name: "Create chart" })).toBeEnabled();
    const analysisResponse = page.waitForResponse((r) => r.url().endsWith("/chart-analysis") && r.request().method() === "POST" && r.ok());
    await page.getByRole("button", { name: "Create chart" }).click();
    const analysis: ChartAnalysis = await (await analysisResponse).json();
    const bins = analysis.preview.histograms[0].bins;
    expect(bins.length).toBeGreaterThan(1);
    expect(bins[0].start).toBeCloseTo(Math.min(...values), 8);
    expect(bins.at(-1)!.end).toBeCloseTo(Math.max(...values), 8);
    for (const [index, bin] of bins.entries()) {
      expect(bin.count).toBe(values.filter((v) => v >= bin.start && (index === bins.length - 1 ? v <= bin.end : v < bin.end)).length);
    }
    expect(bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(values.length);
    await expect(page.getByRole("combobox", { name: "Chart type" })).toHaveText("Histogram");
    await expect(page.getByRole("combobox", { name: "X field", exact: true })).toHaveCount(0);
    // A distinctive color allows checking the actual SVG geometry, not only chart options.
    await page.getByRole("button", { name: "Series appearance", exact: true }).click();
    await page.locator('input[type="color"]').first().fill("#dc2626");
    await page.getByRole("textbox", { name: "Figure title" }).fill(`Parity ${filename}`);
    await page.getByRole("button", { name: "Customize export" }).click();
    const chart = page.locator('main [role="img"]').first();
    await expect(chart.locator("svg")).toBeVisible();
    await expect.poll(() => chart.locator('path[fill="#dc2626"]').count()).toBeGreaterThan(bins.filter((bin) => bin.count > 0).length);
    const rectangles = await chart.locator("svg").evaluate((svg) =>
      Array.from(svg.querySelectorAll('path[fill="#dc2626"]')).map((element) => {
        const box = (element as SVGGraphicsElement).getBBox();
        return { x: box.x, y: box.y, width: box.width, height: box.height, opacity: element.getAttribute("fill-opacity") ?? element.getAttribute("opacity") };
      }).filter((box) => box.height > 20 && box.width > 1).sort((a, b) => a.x - b.x),
    );
    const nonempty = bins.filter((bin) => bin.count > 0);
    expect(rectangles).toHaveLength(nonempty.length);
    const left = rectangles[0].x;
    const right = rectangles.at(-1)!.x + rectangles.at(-1)!.width;
    const span = bins.at(-1)!.end - bins[0].start;
    const baseline = Math.max(...rectangles.map((box) => box.y + box.height));
    const heightUnit = Math.max(...rectangles.map((box) => box.height)) / Math.max(...bins.map((bin) => bin.count));
    for (const [index, bin] of nonempty.entries()) {
      const box = rectangles[index];
      expect((box.x - left) / (right - left)).toBeCloseTo((bin.start - bins[0].start) / span, 2);
      expect(box.width / (right - left)).toBeCloseTo((bin.end - bin.start) / span, 2);
      expect(box.height / heightUnit).toBeCloseTo(bin.count, 1);
      expect(box.y + box.height).toBeCloseTo(baseline, 1);
      expect(Number(box.opacity)).toBeCloseTo(0.55, 2);
    }
    await chart.screenshot({ path: test.info().outputPath("preview.png") });
    for (const format of ["SVG", "PNG", "PDF"]) {
      await page.getByRole("combobox", { name: "Format", exact: true }).click();
      await page.getByRole("option", { name: format, exact: true }).click();
      await page.getByRole("button", { name: "Prepare export" }).click();
      const link = page.getByRole("link", { name: `Download ${format}`, exact: true });
      await expect(link).toBeVisible({ timeout: 60_000 });
      const event = page.waitForEvent("download");
      await link.click({ force: true });
      const download = await event;
      expect(await download.failure()).toBeNull();
      const target = test.info().outputPath(`export.${format.toLowerCase()}`);
      await download.saveAs(target);
      const bytes = await readFile(target);
      if (format === "PNG") expect(bytes.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
      if (format === "PDF") expect(bytes.subarray(0, 4).toString()).toBe("%PDF");
      if (format === "SVG") {
        const exported = bytes.toString();
        expect(exported).toContain(`Parity ${filename}`);
        const geometry = await page.evaluate((text) => {
          const doc = new DOMParser().parseFromString(text, "image/svg+xml");
          const stairs = Array.from(doc.querySelectorAll("path")).find((p) => p.getAttribute("style")?.includes("fill: #dc2626") && p.hasAttribute("clip-path"));
          return { path: stairs?.getAttribute("d"), style: stairs?.getAttribute("style") };
        }, exported);
        expect(geometry.style).toContain("opacity: 0.55");
        expect(geometry.path).toBeTruthy();
        const numbers = geometry.path!.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi)!.map(Number);
        const points = Array.from({ length: numbers.length / 2 }, (_, i) => ({ x: numbers[2 * i], y: numbers[2 * i + 1] }));
        const minX = Math.min(...points.map((p) => p.x));
        const maxX = Math.max(...points.map((p) => p.x));
        const floor = Math.max(...points.map((p) => p.y));
        const scale = (floor - Math.min(...points.map((p) => p.y))) / Math.max(...bins.map((bin) => bin.count));
        for (const [index, bin] of bins.entries()) {
          const start = points[1 + index * 2];
          const end = points[2 + index * 2];
          expect((start.x - minX) / (maxX - minX)).toBeCloseTo((bin.start - bins[0].start) / span, 4);
          expect((end.x - minX) / (maxX - minX)).toBeCloseTo((bin.end - bins[0].start) / span, 4);
          expect((floor - start.y) / scale).toBeCloseTo(bin.count, 4);
        }
      }
    }
    const python = process.env.LABVIZ_E2E_ARTIFACT_PYTHON ?? path.resolve(__dirname, "../../api/.venv/Scripts/python.exe");
    await promisify(execFile)(python, [path.resolve(__dirname, "support/verify-distribution-export.py"),
      test.info().outputDir, JSON.stringify(bins)], { timeout: 30_000 });
    // A reopened workspace defaults to the import step; explicitly reopen its export route.
    await page.goto(`${new URL(page.url()).pathname}?step=export`, { waitUntil: "networkidle" });
    await expect(page.getByRole("heading", { name: "Prepare publication export" })).toBeVisible();
    await expect(page.locator('main [role="img"] svg').first()).toBeVisible();
  });
}
