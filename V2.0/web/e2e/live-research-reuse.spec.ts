import { expect, test } from "./fixtures";

test.skip(!process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY, "Requires isolated local API.");

test("template replacement and frozen research output survive reopen", async ({ page }, testInfo) => {
  const unlocked = page.waitForResponse((r) => r.url().endsWith("/api/v1/local/session"));
  await page.goto(`/#labviz-access=${process.env.LABVIZ_E2E_LOCAL_ACCESS_KEY}`);
  expect((await unlocked).ok()).toBeTruthy();
  const source = "s,repeat,v,unit\na,1,1,mV\na,2,3,mV\nb,1,5,mV\nb,2,7,mV\n";
  const upload = await page.request.post("/api/v1/projects", { multipart: { file: { name: "original.csv", mimeType: "text/csv", buffer: Buffer.from(source) } } });
  expect(upload.ok()).toBeTruthy();
  const uploaded = await upload.json();
  const id = uploaded.projectId;
  await expect.poll(async () => (await (await page.request.get(`/api/v1/jobs/${uploaded.job.id}`)).json()).stage).toBe("ready");
  const base = `/api/v1/projects/${id}/research`;
  const prepared = await page.request.post(base + "/preview", { multipart: {
    plan: JSON.stringify({ mapping: { value: "v", sample: "s", experimentalUnit: "s", technicalRepeat: "repeat", unit: "unit" }, analysisUnit: "experimental_unit" }),
    files: { name: "original.csv", mimeType: "text/csv", buffer: Buffer.from(source) },
  } });
  expect(prepared.ok(), await prepared.text()).toBeTruthy();
  const applied = await page.request.post(base + "/apply", { data: { token: (await prepared.json()).token } });
  expect(applied.ok()).toBeTruthy();
  const chart = await page.request.post(base + "/charts", { data: { revisionId: (await applied.json()).revisionId, spec: { title: "Reusable points" } } });
  expect(chart.ok()).toBeTruthy();
  const figure = await page.request.post(base + "/figure-boards", { data: { title: "Original figure", panelChartIds: [(await chart.json()).id] } });
  expect(figure.ok()).toBeTruthy();
  await page.goto(`/workspace/${id}`);
  await page.getByRole("button", { name: "Research workspace", exact: true }).click();
  await page.getByRole("tab", { name: "Outputs", exact: true }).click();
  const reuse = page.getByTestId("reuse-workspace");
  const select = async (label: string, option: string | RegExp) => {
    await reuse.getByRole("combobox", { name: label, exact: true }).click();
    await page.getByRole("option", { name: option, exact: typeof option === "string" }).click();
  };
  await select("Saved figure for template / output", /Original figure/);
  await reuse.getByRole("textbox", { name: "Template name", exact: true }).fill("Repeat experiment");
  await reuse.getByRole("button", { name: "Save workflow template", exact: true }).click();
  await expect(reuse.getByRole("status")).toHaveText("Template saved locally.");
  await reuse.getByTestId("replacement-file").setInputFiles({ name: "next-batch.csv", mimeType: "text/csv", buffer: Buffer.from(source.replace(",v,", ",voltage,").replace(",7,", ",17,")) });
  await select("Map v", "voltage");
  await reuse.getByRole("button", { name: "Validate replacement", exact: true }).click();
  await expect(reuse.getByTestId("replacement-preview")).toContainText("A: ready");
  await reuse.getByRole("button", { name: "Apply validated replacement", exact: true }).click();
  await expect(reuse.getByRole("status")).toHaveText("Replacement saved as new revisions.");
  const boards = (await (await page.request.get(base + "/figure-boards")).json()).boards;
  await select("Saved figure for template / output", new RegExp(boards[0].id.slice(0, 8)));
  await reuse.getByRole("button", { name: "Preview included output files", exact: true }).click();
  await expect(reuse.getByTestId("output-preview")).toContainText("panels/A-source.csv");
  await reuse.getByRole("textbox", { name: "Figure legend (Chinese)", exact: true }).fill("已核对的新批次图注");
  await expect(reuse.getByRole("button", { name: "Save frozen output", exact: true })).toBeDisabled();
  await reuse.getByRole("checkbox", { name: "I reviewed the included data, identifiers and notes" }).check();
  await reuse.getByRole("button", { name: "Save frozen output", exact: true }).click();
  const link = reuse.getByRole("link", { name: /Download research ZIP/ });
  await expect(link).toBeVisible();
  const href = await link.getAttribute("href");
  const original = await (await page.request.get(href!)).body();
  expect(original.subarray(0, 2).toString()).toBe("PK");
  await reuse.screenshot({ path: testInfo.outputPath("reuse-desktop.png") });
  await page.reload();
  await expect(page.getByRole("button", { name: "Back to quick plot", exact: true })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Outputs", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".MuiDataGrid-root")).toHaveCount(0);
  await expect(link).toBeVisible();
  expect(await (await page.request.get(href!)).body()).toEqual(original);
  await page.setViewportSize({ width: 390, height: 844 });
  await reuse.scrollIntoViewIfNeeded();
  await reuse.screenshot({ path: testInfo.outputPath("reuse-mobile.png") });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});
