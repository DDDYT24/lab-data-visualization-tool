import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { sampleBrowserMemory } from "./phase5-memory";
import { expect, test } from "./fixtures";

// Opt-in measurements run alone against the private test API; ordinary CI stays short.
// DOM snapshot tracing changes the workload; opt in only for diagnostic comparison.
const traceMode = process.env.LABVIZ_PHASE5_TRACE === "1" ? "retain-on-failure" : "off";
test.use({ trace: traceMode });
test.skip(!process.env.LABVIZ_PHASE5_PERF || !process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Opt-in isolated performance run.");
for (const [rows, mixed] of [[10000, false], [100000, false], [100000, true]] as const) {
  test(`Phase 5 real browser ${mixed ? "mixed " : ""}${rows} rows`, async ({ page, browser }, testInfo) => {
    test.setTimeout(900_000);
    const outputDir = resolve("../../outputs/v23-phase5");
    const reportSuffix = process.env.LABVIZ_PHASE5_REPORT_SUFFIX ?? "";
    const payloadPath = resolve(outputDir, `legacy-linear-${rows}.csv`);
    const payload = readFileSync(payloadPath);
    const bufferInput = process.env.LABVIZ_PHASE5_BUFFER_INPUT === "1";
    const runs: Record<string, number>[] = [];
    const memory: unknown[] = [];
    const projectIds: string[] = [];
    let actualApiPid = Number(process.env.LABVIZ_PHASE5_API_PID);
    if (process.env.LABVIZ_PHASE5_RESTART_URL) {
      const current = await fetch(process.env.LABVIZ_PHASE5_RESTART_URL.replace(/\/restart$/, "/status"), {
        method: "POST", headers: { "X-LabViz-Test-Control": process.env.LABVIZ_PHASE5_CONTROL_KEY! },
      });
      expect(current.ok).toBeTruthy();
      actualApiPid = (await current.json()).actualApiPid;
    }
    expect(Number.isSafeInteger(actualApiPid) && actualApiPid > 0).toBeTruthy();
    const serverTiming = (header: string | undefined, prefix: string, timings: Record<string, number>) => {
      for (const metric of (header ?? "").split(",")) {
        const match = metric.trim().match(/^(compute|render|persist|durableLayout);dur=([0-9.]+)$/);
        if (match) timings[prefix + match[1] + "ServerMs"] = Number(match[2]);
      }
    };
    const duration = (start: number) => Math.round((performance.now() - start) * 100) / 100;
    // Independent, read-only readiness timing. Keep the original sequential
    // assertion timing below until its polling overhead has been quantified.
    const observeFigureReadiness = (started: number, cold = false) => page.waitForFunction(({ expectedRows, cold }) => {
      const region = document.querySelector('[data-testid="figure-inspector"]');
      const image = region?.querySelector<HTMLImageElement>('img[alt="Saved composed figure"]');
      const status = region?.querySelector('[role="status"]')?.textContent ?? "";
      const coldReady = !cold || (document.getElementById("research-tab-figures")?.getAttribute("aria-selected") === "true"
        && document.querySelectorAll(".MuiDataGrid-root").length === 0);
      return coldReady && !!image?.naturalWidth && image.complete && status.includes(`Matching panel rows: ${expectedRows}`);
    }, { expectedRows: rows * 4, cold }, { polling: "raf", timeout: 120_000 }).then(async (handle) => {
      await handle.dispose();
      await page.getByTestId("figure-inspector").getByRole("img", { name: "Saved composed figure", exact: true }).evaluate(async (element) => {
        await (element as HTMLImageElement).decode();
        await new Promise<void>((resolvePaint) => requestAnimationFrame(() => requestAnimationFrame(() => resolvePaint())));
      });
      return duration(started);
    });
    for (let repeat = 0; repeat < 3; repeat++) {
      const timings: Record<string, number> = {};
      const recordStage = async (stage: string) => {
        if (!process.env.LABVIZ_PHASE5_DIAGNOSTICS) return;
        const session = await page.context().newCDPSession(page);
        try {
          await session.send("Performance.enable");
          const metrics = await session.send("Performance.getMetrics");
          const dom = await session.send("Memory.getDOMCounters");
          const heap = Object.fromEntries(metrics.metrics.filter((item) => ["JSHeapUsedSize", "JSHeapTotalSize"].includes(item.name)).map((item) => [item.name, item.value / 2 ** 20]));
          const content = await page.evaluate(() => ({ elements: document.querySelectorAll("*").length, circles: document.querySelectorAll("circle").length, images: [...document.images].map((img) => ({ width: img.naturalWidth, height: img.naturalHeight })) }));
          const navigation = await page.evaluate(() => ({
            document: performance.getEntriesByType("navigation").map((entry) => {
              const item = entry as PerformanceNavigationTiming;
              return { responseStart: item.responseStart, responseEnd: item.responseEnd, domInteractive: item.domInteractive, domContentLoaded: item.domContentLoadedEventEnd, load: item.loadEventEnd };
            }),
            resources: (performance.getEntriesByType("resource") as PerformanceResourceTiming[]).map((item) => ({
              path: new URL(item.name).pathname, initiator: item.initiatorType,
              start: item.startTime, duration: item.duration, responseStart: item.responseStart,
              responseEnd: item.responseEnd, transferBytes: item.transferSize, decodedBytes: item.decodedBodySize,
            })),
          }));
          memory.push({ stage, repeat: repeat + 1, heapMiB: heap, dom, content, navigation,
            rendererSeconds: Object.fromEntries(metrics.metrics.filter((item) => ["ScriptDuration", "TaskDuration", "LayoutDuration", "RecalcStyleDuration"].includes(item.name)).map((item) => [item.name, item.value])) });
        } finally { await session.detach(); }
      };
      const memorySession = await browser.newBrowserCDPSession();
      const initialProcesses = await memorySession.send("SystemInfo.getProcessInfo");
      await memorySession.detach();
      const root = initialProcesses.processInfo.find((p) => p.type === "browser");
      const sampler = sampleBrowserMemory(Number(root?.id), testInfo.outputPath(`memory-stop-${repeat}`));
      try {
      sampler.stage("local-session");
      const unlocked = await page.request.post("/api/v1/local/session", { headers: { "X-LabViz-Local-Key": process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY! } });
      expect(unlocked.ok()).toBeTruthy();
      sampler.stage("navigate-to-import");
      await page.goto("/");
      let start = performance.now();
      sampler.stage("import-and-data-grid");
      await page.locator('input[type="file"]').setInputFiles(bufferInput
        ? { name: "legacy-linear.csv", mimeType: "text/csv", buffer: payload } : payloadPath);
      await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible({ timeout: 120_000 });
      timings.importToReadyBrowserMs = duration(start);
      await recordStage("import-ready");
      const projectId = new URL(page.url()).pathname.split("/").at(-1)!;
      projectIds.push(projectId);
      const workspace = await (await page.request.get(`/api/v1/projects/${projectId}/workspace`)).json();
      expect(workspace.preview.totalRows).toBe(rows);
      expect(workspace.session.source.size).toBe(payload.length);
      const base = `/api/v1/projects/${projectId}/research`;
      start = performance.now();
      sampler.stage("research-preparation");
      const preview = await page.request.post(base + "/preview", { timeout: 120_000, multipart: {
        plan: JSON.stringify({ mapping: { value: "response_mV", defaultUnit: "mV" } }),
        files: { name: "legacy-linear.csv", mimeType: "text/csv", buffer: payload },
      } });
      expect(preview.ok(), await preview.text()).toBeTruthy();
      timings.researchPreparationHttpMs = duration(start);
      start = performance.now();
      const applied = await page.request.post(base + "/apply", { timeout: 120_000, data: { token: (await preview.json()).token } });
      expect(applied.ok(), await applied.text()).toBeTruthy();
      timings.researchApplyHttpMs = duration(start);
      const revisionId = (await applied.json()).revisionId;
      expect(revisionId).toBeTruthy();
      // Measure a real UI click through the first decoded preview and two paint opportunities.
      await page.reload();
      await page.getByRole("button", { name: "Research workspace", exact: true }).click();
      await page.getByRole("tab", { name: "Analysis", exact: true }).click();
      const chartRegion = page.getByTestId("research-charts");
      await chartRegion.getByRole("textbox", { name: "Chart title", exact: true }).fill("Large raw points");
      const generateButton = chartRegion.getByRole("button", { name: "Generate and save chart", exact: true });
      await expect(generateButton).toBeEnabled();
      await generateButton.evaluate((element) => element.addEventListener("click", () => {
        element.setAttribute("data-test-preview-start", String(performance.now()));
      }, { once: true, capture: true }));
      sampler.stage("chart-generation");
      const generatedResponse = page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname === base + "/charts", { timeout: 120_000 });
      await generateButton.click();
      const generated = await generatedResponse;
      expect(generated.ok(), await generated.text()).toBeTruthy();
      await generated.finished();
      const requestTiming = generated.request().timing();
      timings.computeRenderSaveChartHttpMs = requestTiming.responseEnd - requestTiming.requestStart;
      const firstPreview = chartRegion.getByRole("img", { name: "Large raw points", exact: true });
      await expect(firstPreview).toBeVisible();
      timings.generateClickToFirstPreviewPaintBrowserMs = await firstPreview.evaluate(async (element) => {
        const image = element as HTMLImageElement;
        await image.decode();
        const button = element.closest('[data-testid="research-charts"]')!.querySelector('[data-test-preview-start]')!;
        const started = Number(button.getAttribute("data-test-preview-start"));
        if (!started || !image.naturalWidth) throw new Error("First preview measurement boundary missing");
        return new Promise<number>((resolveTiming) => requestAnimationFrame(() => requestAnimationFrame(() => resolveTiming(Math.round((performance.now() - started) * 100) / 100))));
      });
      serverTiming(generated.headers()["server-timing"], "chart", timings);
      const chartId = (await generated.json()).id;
      let panelIds = [chartId, chartId, chartId, chartId];
      if (mixed) {
        start = performance.now();
        const captured = await page.request.post(base + "/figure-sources?panel=1", { timeout: 120_000 });
        expect(captured.ok(), await captured.text()).toBeTruthy();
        timings.legacyCaptureHttpMs = duration(start);
        const violin = await page.request.post(base + "/charts", { timeout: 120_000,
          data: { revisionId, spec: { kind: "violin", pointRendering: "auto" } } });
        expect(violin.ok(), await violin.text()).toBeTruthy();
        panelIds = [(await captured.json()).id, chartId, (await captured.json()).id, (await violin.json()).id];
      }
      start = performance.now();
      const figureStarted = start;
      sampler.stage("figure-generation");
      const board = await page.request.post(base + "/figure-boards?deferArtifacts=true", { timeout: 120_000, data: { title: "Four large panels", panelChartIds: panelIds, pointRendering: "auto" } });
      expect(board.status(), await board.text()).toBe(202);
      timings.durableLayoutSaveHttpMs = duration(start);
      serverTiming(board.headers()["server-timing"], "figure", timings);
      const boardId = (await board.json()).id;
      let artifactTimings: Record<string, number> = {};
      await expect.poll(async () => {
        const status = await page.request.get(base + `/figure-boards/${boardId}`);
        expect(status.ok()).toBeTruthy();
        const saved = await status.json();
        expect(saved.artifactState, saved.artifactError ?? "Generation failed").not.toBe("failed");
        artifactTimings = saved.artifactTimings ?? {};
        return saved.artifactState;
      }, { timeout: 180_000, intervals: [100, 250, 500] }).toBe("ready");
      timings.layoutToAllArtifactsReadyHttpMs = duration(figureStarted);
      for (const [key, value] of Object.entries(artifactTimings)) timings[`figureGeneration_${key}Ms`] = value;
      let frozenPngSha = "";
      for (const kind of ["png", "svg", "pdf"]) {
        start = performance.now();
        const artifact = await page.request.get(base + `/figure-boards/${boardId}/artifact/${kind}?download=true`, { timeout: 120_000 });
        expect(artifact.ok()).toBeTruthy();
        const bytes = await artifact.body();
        if (kind === "png") frozenPngSha = createHash("sha256").update(bytes).digest("hex");
        timings[`frozen${kind.toUpperCase()}DownloadHttpMs`] = duration(start);
      }
      start = performance.now();
      sampler.stage("warm-page-reopen");
      await page.reload();
      timings.warmNavigationMs = duration(start);
      const warmObserved = observeFigureReadiness(start);
      await expect(page.getByRole("button", { name: "Back to quick plot", exact: true })).toBeVisible();
      timings.warmWorkspaceReadyMs = duration(start);
      await page.getByRole("tab", { name: "Figures", exact: true }).click();
      timings.warmTaskSelectedMs = duration(start);
      const inspector = page.getByTestId("figure-inspector");
      await expect(inspector.getByRole("status")).toContainText(`Matching panel rows: ${rows * 4}`, { timeout: 120_000 });
      timings.warmInteractionReadyMs = duration(start);
      await expect.poll(() => inspector.getByRole("img", { name: "Saved composed figure", exact: true }).evaluate((el) => (el as HTMLImageElement).naturalWidth), { timeout: 120_000 }).toBeGreaterThan(0);
      timings.reopenToUsableFigureBrowserMs = duration(start);
      timings.warmObservedUsableBrowserMs = await warmObserved;
      if (process.env.LABVIZ_PHASE5_POINTER_PAINT === "1") {
        timings.warmFigureInteractionResourceMs = await page.evaluate(() => {
          const entries = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
          return entries.findLast((entry) => entry.name.includes("/interaction?"))?.duration ?? 0;
        });
      }
      await recordStage("figure-ready");
      if (process.env.LABVIZ_PHASE5_RESTART_URL) {
        const beforePeak = Number(execFileSync("powershell.exe", ["-NoProfile", "-Command", `(Get-Process -Id ${actualApiPid}).PeakWorkingSet64`], { encoding: "utf8", windowsHide: true }));
        memory.push({ backendPid: actualApiPid, phase: "generation-before-restart", backendProcessLifetimePeakMiB: beforePeak / 2 ** 20 });
        start = performance.now();
        sampler.stage("api-restart");
        const restarted = await fetch(process.env.LABVIZ_PHASE5_RESTART_URL, { method: "POST", headers: { "X-LabViz-Test-Control": process.env.LABVIZ_PHASE5_CONTROL_KEY! } });
        expect(restarted.ok).toBeTruthy();
        actualApiPid = (await restarted.json()).actualApiPid;
        timings.coldApiRestartStartupMs = duration(start);
        start = performance.now();
        sampler.stage("cold-page-reopen");
        await page.reload();
        const coldObserved = observeFigureReadiness(start, true);
        await expect(page.getByRole("button", { name: "Back to quick plot", exact: true })).toBeVisible();
        await expect(page.getByRole("tab", { name: "Figures", exact: true })).toHaveAttribute("aria-selected", "true");
        await expect(page.locator(".MuiDataGrid-root")).toHaveCount(0);
        await expect(inspector.getByRole("status")).toContainText(`Matching panel rows: ${rows * 4}`, { timeout: 120_000 });
        await expect.poll(() => inspector.getByRole("img", { name: "Saved composed figure", exact: true }).evaluate((el) => (el as HTMLImageElement).naturalWidth), { timeout: 120_000 }).toBeGreaterThan(0);
        timings.coldReopenToUsableFigureBrowserMs = duration(start);
        timings.coldObservedUsableBrowserMs = await coldObserved;
      if (process.env.LABVIZ_PHASE5_POINTER_PAINT === "1") {
        timings.coldFigureInteractionResourceMs = await page.evaluate(() => {
          const entries = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
          return entries.findLast((entry) => entry.name.includes("/interaction?"))?.duration ?? 0;
        });
      }
        timings.coldRestartAndReopenBrowserMs = timings.coldApiRestartStartupMs + timings.coldReopenToUsableFigureBrowserMs;
        await recordStage("cold-figure-ready");
      }

      // The overview source dock is a disclosure. Open it before the existing
      // row-highlight measurement; timing still covers the same interaction and paint.
      const sourceDockToggle = inspector.getByRole("button", { name: /Linked source rows/ });
      if (await sourceDockToggle.isVisible()) {
        await sourceDockToggle.click();
        // Disclosure opening is preparation, outside the row-click metric.
        // A synthetic click must not bypass a still-moving/collapsed control.
        await inspector.getByTestId("figure-source-dock").evaluate(async (element) => {
          const animations = element.getAnimations({ subtree: true }).filter((animation) => Number.isFinite(animation.effect?.getComputedTiming().endTime));
          await Promise.allSettled(animations.map((animation) => animation.finished));
        });
        await inspector.getByRole("table").getByRole("button").first().click({ trial: true });
      }
      // Measure inside the browser, through DOM update and two paint opportunities.
      sampler.stage("highlight");
      timings.highlightToPaintBrowserMs = await inspector.getByRole("table").getByRole("button").first().evaluate((button) => new Promise<number>((resolveTiming, reject) => {
        const region = button.closest('[data-testid="figure-inspector"]')!;
        const status = region.querySelector('[role="status"]')!;
        const start = performance.now();
        const timeout = setTimeout(() => { observer.disconnect(); reject(new Error("Highlight did not settle in 15 seconds")); }, 15_000);
        const observer = new MutationObserver(() => {
          if (status.textContent?.includes("Selected inputs: 1")) {
            observer.disconnect(); clearTimeout(timeout);
            requestAnimationFrame(() => requestAnimationFrame(() => resolveTiming(Math.round((performance.now() - start) * 100) / 100)));
          }
        });
        observer.observe(status, { subtree: true, childList: true, characterData: true });
        (button as HTMLButtonElement).click();
      }));
      await expect(inspector.getByRole("table").getByRole("row")).toHaveCount(mixed ? 3 : 5);
      await recordStage("highlighted");
      sampler.stage("dense-point-pointer-selection");
      const input = await (await page.request.get(base + `/figure-boards/${boardId}/interaction`)).json();
      const mark = input.marks.find((item: { kind: string }) => item.kind === "point");
      expect(mark).toBeTruthy();
      const overlay = inspector.getByLabel("Figure selection overlay", { exact: true });
      await expect(overlay.locator('circle[role="button"]')).toHaveCount(mixed ? 1 : 0);
      const bounds = await overlay.boundingBox();
      expect(bounds).toBeTruthy();
      // Capture the actual CSS click location: mouse coordinates may round on each platform.
      const pointerPaintDiagnostic = process.env.LABVIZ_PHASE5_POINTER_PAINT === "1";
      await overlay.evaluate((element, recordPaint) => element.addEventListener("click", (event) => {
        const rect = element.getBoundingClientRect();
        element.setAttribute("data-test-hit-start", String(performance.now()));
        element.setAttribute("data-test-hit-x", String(((event as MouseEvent).clientX - rect.left) * 1000 / rect.width));
        element.setAttribute("data-test-hit-y", String(((event as MouseEvent).clientY - rect.top) * 1000 / rect.height));
        if (recordPaint) {
          const status = element.closest('[data-testid="figure-inspector"]')!.querySelector('[role="status"]')!;
          const started = Number(element.getAttribute("data-test-hit-start"));
          let loadingSeen = false;
          const observer = new MutationObserver(() => {
            if (status.textContent?.includes("Loading")) loadingSeen = true;
            if (loadingSeen && status.textContent?.includes("Selected inputs:")) {
              observer.disconnect(); clearTimeout(timeout);
              // Linked rows and settled status are committed together. Retain the
              // original independent HTTP verification metric below.
              requestAnimationFrame(() => requestAnimationFrame(() => {
                element.setAttribute("data-test-pointer-paint-ms", String(performance.now() - started));
              }));
            }
          });
          const timeout = setTimeout(() => observer.disconnect(), 15_000);
          observer.observe(status, { subtree: true, childList: true, characterData: true });
        }
      }, { once: true, capture: true }), pointerPaintDiagnostic);
      const pointerResponse = page.waitForResponse((response) => {
        const url = new URL(response.url());
        return url.pathname.endsWith(`/figure-boards/${boardId}/interaction`) && url.searchParams.has("mark");
      }, { timeout: 30_000 });
      await overlay.click({ position: { x: mark.cx * bounds!.width, y: mark.cy * bounds!.height } });
      const actualResponse = await pointerResponse;
      expect(actualResponse.ok()).toBeTruthy();
      const click = await overlay.evaluate((element) => ({ x: Number(element.getAttribute("data-test-hit-x")), y: Number(element.getAttribute("data-test-hit-y")) }));
      const hit = [...input.marks].reverse().find((item) => (item.cx * 1000 - click.x) ** 2
        + (item.cy * 1000 - click.y) ** 2 <= (item.kind === "point" ? 5.75 : 9.75) ** 2);
      expect(new URL(actualResponse.url()).searchParams.get("mark")).toBe(hit.id);
      const expectedSelection = await (await page.request.get(base + `/figure-boards/${boardId}/interaction?mark=${encodeURIComponent(hit.id)}`)).json();
      await expect(inspector.getByRole("status")).toContainText(`Selected inputs: ${expectedSelection.selectedInputs}`);
      await expect(inspector.getByRole("status")).toContainText(`Matching panel rows: ${expectedSelection.totalRows}`);
      await expect(inspector.getByRole("table").getByRole("row")).toHaveCount(expectedSelection.rows.length + 1);
      timings.pointerClickToVerifiedPaintBrowserMs = await overlay.evaluate((element) => {
        const started = Number(element.getAttribute("data-test-hit-start"));
        if (!started) throw new Error("Actual pointer measurement boundary missing");
        return new Promise<number>((resolveTiming) => requestAnimationFrame(() => requestAnimationFrame(() => resolveTiming(Math.round((performance.now() - started) * 100) / 100))));
      });
      if (pointerPaintDiagnostic) {
        await expect.poll(() => overlay.getAttribute("data-test-pointer-paint-ms"), { timeout: 15_000 }).not.toBeNull();
        timings.pointerClickToUiPaintBrowserMs = Number(await overlay.getAttribute("data-test-pointer-paint-ms"));
        expect(timings.pointerClickToUiPaintBrowserMs).toBeGreaterThan(0);
      }
      const frozenAfterSelection = await (await page.request.get(base + `/figure-boards/${boardId}/artifact/png?download=true`)).body();
      expect(createHash("sha256").update(frozenAfterSelection).digest("hex")).toBe(frozenPngSha);
      const cdp = await browser.newBrowserCDPSession();
      const processes = await cdp.send("SystemInfo.getProcessInfo");
      await cdp.detach();
      const pids = processes.processInfo.map((p) => Number(p.id)).filter((pid) => Number.isSafeInteger(pid) && pid > 0);
      const pidList = pids.join(",");
      // Sum lifetime process peaks: a conservative bound, not a synchronized peak/RSS claim.
      const processPeaks = JSON.parse(execFileSync("powershell.exe", ["-NoProfile", "-Command", `Get-Process -Id ${pidList} -ErrorAction SilentlyContinue | Select-Object Id,PeakWorkingSet64 | ConvertTo-Json -Compress`], { encoding: "utf8", windowsHide: true }));
      const peaks = Array.isArray(processPeaks) ? processPeaks : [processPeaks];
      memory.push({ processes: processes.processInfo, processPeaks: peaks, complete: peaks.length === pids.length, browserProcessPeakSumMiB: peaks.reduce((sum, p) => sum + p.PeakWorkingSet64, 0) / 2 ** 20 });
      const apiPid = actualApiPid;
      if (Number.isSafeInteger(apiPid) && apiPid > 0) {
        const apiPeak = Number(execFileSync("powershell.exe", ["-NoProfile", "-Command", `(Get-Process -Id ${apiPid}).PeakWorkingSet64`], { encoding: "utf8", windowsHide: true }));
        memory.push({ backendPid: apiPid, backendProcessLifetimePeakMiB: apiPeak / 2 ** 20 });
      }
      runs.push(timings);
      writeFileSync(resolve(outputDir, `browser-performance-${mixed ? "mixed-" : ""}${rows}${reportSuffix}-progress.json`), JSON.stringify({ rows, runs, memory }, null, 2) + "\n");
      console.log(JSON.stringify({ rows, repeat: repeat + 1, timings }));
      sampler.stage("navigate-away-and-delete");
      await page.goto("/");
      await page.request.delete(`/api/v1/projects/${projectId}`);
      } finally {
        memory.push(await sampler.stop());
      }
    }
    const maxMs = Object.fromEntries(Object.keys(runs[0]).map((key) => [key, Math.max(...runs.map((run) => run[key]))]));
    const report = { rows, mixed, instrumentation: { reopenReadinessMeasurement: "legacy sequential assertions; parallel decoded-image/two-paint readiness audit", disclosurePreparation: "finite animations and native actionability before row-click timing", nativePointerPaintDiagnostic: process.env.LABVIZ_PHASE5_POINTER_PAINT === "1", fileInputMode: bufferInput ? "renderer-buffer-reconstruction" : "native-local-file-selection", tracing: traceMode, passiveStageDiagnostics: Boolean(process.env.LABVIZ_PHASE5_DIAGNOSTICS), forcedBrowserGC: false, coldApiRestart: Boolean(process.env.LABVIZ_PHASE5_RESTART_URL) }, datasetSha256: createHash("sha256").update(payload).digest("hex"), statistic: "maximum of three runs including first; not p95", scope: "Real production browser and HTTP. File selection uses a fixed local path by default; optional buffer-input diagnostics reconstruct a File in the renderer and are separately reported. Chart HTTP timing covers request start through complete response; the real Generate and save chart click ends after the decoded first preview and two paint opportunities. An opt-in native-pointer paint diagnostic ends at settled linked UI data and two paint opportunities; it does not replace the original verification metric or grant a budget exception. Pointer click timing includes HTTP selection, verified status/table, test assertion overhead and two paint opportunities; it is separate from the synthetic table-button highlight. Chart save includes computation, all three exports and persistence. Figure durableLayoutSaveHttpMs ends after the layout/job commit; layoutToAllArtifactsReadyHttpMs ends after all three frozen exports commit and remains distinct from the original complete-save budget. Per-format generation times exclude source loading and durable publication; cold vector timing must include geometry and rasterPreparation. Frozen downloads measure transfer only. When the private harness restart is enabled, coldReopen starts after a completely new API process is ready; startup and combined startup/reopen are also separately recorded. No application restart endpoint is added. Memory includes synchronized sampled working-set sums for the dedicated browser process tree, with actual sampling intervals/coverage, plus a separate sum of lifetime process peaks.", runs, maxMs, memory, projectIds, browserVersion: browser.version(), scriptSha256: createHash("sha256").update(readFileSync(testInfo.file)).digest("hex"), memorySamplerSha256: createHash("sha256").update(readFileSync(resolve("e2e/phase5-memory.ts"))).digest("hex") };
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(resolve(outputDir, `browser-performance-${mixed ? "mixed-" : ""}${rows}${reportSuffix}.json`), JSON.stringify(report, null, 2) + "\n");
    await testInfo.attach("performance-report", { body: JSON.stringify(report), contentType: "application/json" });
  });
}
