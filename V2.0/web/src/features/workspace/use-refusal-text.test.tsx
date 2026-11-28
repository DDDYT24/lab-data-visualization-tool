import { NextIntlClientProvider } from "next-intl";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { refusals } from "@/domain/capabilities";
import { LabVizApiError } from "@/lib/api/labviz-api";
import en from "@/messages/en.json";
import zh from "@/messages/zh.json";
import { useRefusalText } from "./use-refusal-text";

function Worded({ cause }: { cause: unknown }) {
  return <>{useRefusalText()(cause)}</>;
}
function shown(cause: unknown, locale: "en" | "zh" = "en") {
  return renderToStaticMarkup(
    <NextIntlClientProvider locale={locale} messages={locale === "en" ? en : zh} timeZone="UTC"><Worded cause={cause} /></NextIntlClientProvider>,
  );
}

describe("refusal text", () => {
  it.each(Object.entries(refusals))("%s is worded in both languages with its values", (code, needed) => {
    const params = Object.fromEntries(needed!.map((name) => [name, `<${name}>`]));
    for (const locale of ["en", "zh"] as const) {
      const text = shown(new LabVizApiError("API message", 422, code, params), locale);
      expect(text).not.toContain("API message");
      expect(text).not.toContain("refusals.");
      for (const name of needed!) expect(text).toContain(`&lt;${name}&gt;`);
    }
  });

  it("writes a row number as it is", () => {
    const error = new LabVizApiError("", 422, "row-width", { file: "a.csv", row: 1234 });
    expect(shown(error)).toBe("a.csv, row 1234: the number of cells differs from the header.");
    expect(shown(error, "zh")).toBe("a.csv 第 1234 行的单元格数与表头不一致。");
  });

  it("has the same service errors in both languages", () => {
    expect(Object.keys(zh.serviceErrors)).toEqual(Object.keys(en.serviceErrors));
    expect(Object.keys(en.serviceErrors).filter((code) => code in refusals)).toEqual([]);
  });

  it.each(Object.entries(zh.serviceErrors))("words the service error %s by its code", (code, text) => {
    expect(shown(new LabVizApiError("API message", 503, code), "zh")).toBe(text);
    expect(shown(new LabVizApiError("API message", 503, code))).toBe(en.serviceErrors[code as keyof typeof en.serviceErrors]);
  });

  it("keeps the message of a stored refusal without its values, and of any other error", () => {
    const stored = { code: "figure-shared-axis-control", params: undefined, message: "A shared x axis…" };
    expect(shown(stored, "zh")).toBe("A shared x axis…");
    expect(shown(new LabVizApiError("Too few points.", 422, "fit-not-suitable"), "zh")).toBe("Too few points.");
    expect(shown(new Error("Broken."))).toBe("Broken.");
    expect(shown(new DOMException("Stopped.", "AbortError"), "zh")).toBe("Stopped.");
  });
});
