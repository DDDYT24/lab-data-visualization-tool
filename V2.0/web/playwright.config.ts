import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.PLAYWRIGHT_PORT ?? 3_000);
const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? `http://localhost:${port}`;
const nodeOptions = [process.env.NODE_OPTIONS, "--unhandled-rejections=strict"]
  .filter(Boolean)
  .join(" ");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? [["github"], ["line"]] : "list",
  timeout: 45_000,
  workers: process.env.CI ? 2 : 4,
  expect: {
    timeout: 10_000,
    toHaveScreenshot: {
      caret: "initial",
      maxDiffPixelRatio: 0.03,
    },
  },
  snapshotPathTemplate: "{testDir}/{testFilePath}-snapshots/{arg}{ext}",
  use: {
    baseURL,
    colorScheme: "light",
    locale: "en-US",
    screenshot: "only-on-failure",
    timezoneId: "UTC",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      testIgnore: /mobile\.spec\.ts/,
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "mobile-chromium",
      testMatch: /mobile\.spec\.ts/,
      use: { ...devices["Pixel 5"] },
    },
  ],
  webServer: process.env.PLAYWRIGHT_BASE_URL
    ? undefined
    : {
        command: `npm run dev -- --hostname localhost --port ${port}`,
        env: {
          ...process.env,
          NODE_OPTIONS: nodeOptions,
        },
        reuseExistingServer: !process.env.CI,
        stderr: "pipe",
        stdout: "pipe",
        timeout: 120_000,
        url: baseURL,
      },
});
