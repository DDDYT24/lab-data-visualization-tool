import { describe, expect, it } from "vitest";
import { figureMarkAt } from "./figure-point-hit";
import type { FigureInteraction } from "./research-figure-board";

const point = (id: string, cx: number, cy: number, kind = "point"): FigureInteraction["marks"][number] =>
  ({ id, cx, cy, kind, panel: "A", label: id, inputCount: 1, highlighted: false });

describe("dense point selection without individual SVG hit nodes", () => {
  it("matches the centre and original circle and stroke boundary", () => {
    const marks = [point("known", 0.2, 0.4)];
    expect(figureMarkAt(marks, 200, 400)).toBe("known");
    expect(figureMarkAt(marks, 203, 404)).toBe("known");
    expect(figureMarkAt(marks, 205.75, 400)).toBe("known");
    expect(figureMarkAt(marks, 205.7501, 400)).toBeUndefined();
  });
  it("preserves the last-painted overlapping point", () => {
    expect(figureMarkAt([point("first", 0.2, 0.4), point("last", 0.2, 0.4)], 200, 400)).toBe("last");
  });
  it("preserves aggregate radius, stacking and unselected whitespace", () => {
    const density = point("density", 0.2, 0.4, "density");
    expect(figureMarkAt([density], 209.75, 400)).toBe("density");
    expect(figureMarkAt([density, point("on-top", 0.2, 0.4)], 200, 400)).toBe("on-top");
    expect(figureMarkAt([point("under", 0.2, 0.4), density], 200, 400)).toBe("density");
    expect(figureMarkAt([point("point", 0.2, 0.4)], 500, 500)).toBeUndefined();
  });
  it("retains first and last coordinates in a large hit set", () => {
    const marks = Array.from({ length: 2000 }, (_, index) => point(String(index), index / 2000, 0.5));
    expect(figureMarkAt(marks, 0, 500)).toBe("11");
    expect(figureMarkAt(marks, 999.5, 500)).toBe("1999");
  });
});
