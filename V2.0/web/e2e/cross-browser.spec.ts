import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

test("opens the local-first example workflow in each supported desktop engine", async ({ page }) => {
  const observations = await installMockApi(page);

  await page.goto("/");
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "Make experiment data clear and insightful.",
    }),
  ).toBeVisible();

  await page.getByRole("button", { name: "View examples" }).click();
  const example = page.locator('[data-example-slug="time-series"]');
  await expect(example).toContainText("Time-series response");
  await example.getByRole("button", { name: "Open this example" }).click();

  await expect(page).toHaveURL(/\/workspace\/project-e2e$/);
  await expect(page.getByRole("heading", { name: "Confirm your data" })).toBeVisible();
  expect(observations.sampleSlugs).toEqual(["time-series"]);
});

test("navigates local pages and changes language without background route prefetch", async ({ page }) => {
  await installMockApi(page);
  const prefetchedRoutes: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (
      ["/history", "/help", "/about", "/settings"].includes(url.pathname) &&
      request.headers()["next-router-prefetch"] === "1"
    ) {
      prefetchedRoutes.push(url.pathname);
    }
  });

  await page.goto("/", { waitUntil: "networkidle" });
  for (const route of ["/history", "/help", "/about"]) {
    await page.locator(`header a[href="${route}"]:visible`).first().click();
    await expect(page).toHaveURL(new RegExp(`${route}$`));
    await expect(page.locator("main")).toBeVisible();
    await page.waitForLoadState("networkidle");
  }
  await page.getByLabel("Language").click();
  await page.getByRole("option", { name: "中文" }).click();
  await expect(page.getByLabel("语言")).toBeVisible();
  await page.waitForLoadState("networkidle");
  expect(prefetchedRoutes).toEqual([]);
});
