import { expect, test as base } from "@playwright/test";

type ConsoleErrorExpectation = string | RegExp;

type RuntimeErrorGuard = {
  expectConsoleError: (expected: ConsoleErrorExpectation) => void;
};

export const test = base.extend<RuntimeErrorGuard>({
  expectConsoleError: [
    async ({ page }, use) => {
      const consoleErrors: string[] = [];
      const expectedConsoleErrors: ConsoleErrorExpectation[] = [];
      const pageErrors: string[] = [];

      page.on("console", (message) => {
        if (message.type() === "error") {
          consoleErrors.push(message.text());
        }
      });
      page.on("pageerror", (error) => {
        pageErrors.push(error.stack ?? error.message);
      });

      await use((expected) => {
        expectedConsoleErrors.push(expected);
      });

      expect(pageErrors, "Unhandled browser page errors").toEqual([]);

      const unexpectedConsoleErrors = [...consoleErrors];
      const missingConsoleErrors: string[] = [];
      for (const expected of expectedConsoleErrors) {
        const matchIndex = unexpectedConsoleErrors.findIndex((actual) =>
          typeof expected === "string" ? actual === expected : expected.test(actual),
        );
        if (matchIndex === -1) {
          missingConsoleErrors.push(String(expected));
        } else {
          unexpectedConsoleErrors.splice(matchIndex, 1);
        }
      }

      expect(missingConsoleErrors, "Expected browser console errors not observed").toEqual([]);
      expect(unexpectedConsoleErrors, "Unexpected browser console errors").toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
