import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.PLAYWRIGHT_PORT ?? 3_100);
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
  timeout: 90_000,
  // Keep cold Turbopack compilation from competing with too many browser workers.
  workers: 2,
  expect: {
    timeout: 10_000,
    toHaveScreenshot: {
      caret: "initial",
      maxDiffPixelRatio: 0.03,
    },
  },
  snapshotPathTemplate: "{testDir}/{testFilePath}-snapshots/{arg}-{platform}{ext}",
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
        command: `npm run prepare:e2e && npm run build && npm run start -- --hostname localhost --port ${port}`,
        env: {
          ...process.env,
          LABVIZ_E2E_BUILD: "1",
          NEXT_TELEMETRY_DISABLED: "1",
          NODE_OPTIONS: nodeOptions,
        },
        reuseExistingServer: false,
        stderr: "pipe",
        stdout: "pipe",
        timeout: 120_000,
        url: baseURL,
      },
});
