import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("sends optional experiment and replicate metadata before upload", async ({ page }) => {
  const observations = await installMockApi(page);
  await page.goto("/");

  await page.getByText("Group repeated acquisitions").click();
  await page.getByLabel("Experiment name").fill("Dose response study");
  await page.getByLabel("Acquisition / run name").fill("Acquisition 1");
  await page.getByLabel("Replicate ID").fill("R1");
  await page.getByLabel("Batch ID").fill("B-2026-09");
  await page.locator('input[type="file"]').setInputFiles({
    name: "replicate-1.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("time,response\n0,1\n1,2\n"),
  });

  await expect.poll(() => observations.uploadBodies.length).toBe(1);
  const multipartBody = observations.uploadBodies[0];
  expect(multipartBody).toContain('name="experimentTitle"');
  expect(multipartBody).toContain("Dose response study");
  expect(multipartBody).toContain('name="runLabel"');
  expect(multipartBody).toContain("Acquisition 1");
  expect(multipartBody).toContain('name="replicateId"');
  expect(multipartBody).toContain("R1");
  expect(multipartBody).toContain('name="batchId"');
  expect(multipartBody).toContain("B-2026-09");
});
