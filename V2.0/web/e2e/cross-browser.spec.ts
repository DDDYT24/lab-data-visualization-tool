import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("opens the local-first example workflow in each supported desktop engine", async ({ page }) => {
  const observations = await installMockApi(page);

  await page.goto("/");
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "Turn experiment data into a clear figure.",
    }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Try sample data" }).click();
  const example = page.locator('[data-example-slug="time-series"]');
  await expect(example).toContainText("Time-series response");
  await example.getByRole("button", { name: "Open this example" }).click();

  await expect(page).toHaveURL(/\/workspace\/project-e2e$/);
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  expect(observations.sampleSlugs).toEqual(["time-series"]);
});
