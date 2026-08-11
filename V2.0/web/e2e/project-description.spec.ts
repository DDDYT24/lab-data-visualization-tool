import { expect, test } from "./fixtures";

import { installMockApi } from "./support/mock-api";

const workspaceUrl = "/workspace/project-e2e?step=export";

test("saves a UTF-8 plain-text description as a new project revision", async ({
  page,
}) => {
  const observations = await installMockApi(page, { authenticated: true });
  const description = "<b>Plain text</b> — 第一版";

  await page.goto(workspaceUrl);
  await expect(page.getByText("No description has been added yet.")).toBeVisible();
  await page.getByLabel("Description").fill(description);
  await page.getByRole("button", { name: "Save description" }).click();

  await expect(
    page.getByText("Description saved as a new project revision."),
  ).toBeVisible();
  await expect(page.getByLabel("Description")).toHaveValue(description);
  expect(observations.descriptionUpdates).toEqual([description]);
});

test("keeps description editing disabled while revision metadata is loading", async ({
  page,
}) => {
  await installMockApi(page, { authenticated: true, descriptionLoading: true });

  await page.goto(workspaceUrl);

  await expect(page.getByText("Loading the current project revision…")).toBeVisible();
  await expect(page.getByLabel("Description")).toBeDisabled();
  await expect(page.getByRole("button", { name: "Save description" })).toBeDisabled();
});

test("reloads the latest project description after an optimistic conflict", async ({
  expectConsoleError,
  page,
}) => {
  expectConsoleError(/Failed to load resource:.*status of 409 \(Conflict\)/);
  await installMockApi(page, {
    authenticated: true,
    descriptionFailure: "conflict",
  });

  await page.goto(workspaceUrl);
  await page.getByLabel("Description").fill("Conflicting edit");
  await page.getByRole("button", { name: "Save description" }).click();
  await expect(page.getByText(/changed in another session/i)).toBeVisible();
  await page.getByRole("button", { name: "Reload" }).click();

  await expect(page.getByText(/changed in another session/i)).toHaveCount(0);
  await expect(page.getByLabel("Description")).toHaveValue("");
});

for (const scenario of [
  {
    failure: "unauthorized" as const,
    status: /Failed to load resource:.*status of 403 \(Forbidden\)/,
    message: "You no longer have permission to edit this project.",
  },
  {
    failure: "deleted" as const,
    status: /Failed to load resource:.*status of 404 \(Not Found\)/,
    message: "This project was deleted and can no longer be edited here.",
  },
  {
    failure: "server" as const,
    status: /Failed to load resource:.*status of 500 \(Internal Server Error\)/,
    message: "Temporary failure.",
  },
]) {
  test(`preserves the draft for a recoverable ${scenario.failure} description failure`, async ({
    expectConsoleError,
    page,
  }) => {
    expectConsoleError(scenario.status);
    await installMockApi(page, {
      authenticated: true,
      descriptionFailure: scenario.failure,
    });

    await page.goto(workspaceUrl);
    await page.getByLabel("Description").fill("Unsaved draft");
    await page.getByRole("button", { name: "Save description" }).click();

    await expect(page.getByText(scenario.message)).toBeVisible();
    await expect(page.getByLabel("Description")).toHaveValue("Unsaved draft");
    await expect(
      page.getByRole("button", { name: "Save description" }),
    ).toBeEnabled();
  });
}
