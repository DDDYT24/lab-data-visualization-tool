import type { FigureInteraction } from "./research-figure-board";

// Match the original 5/9-unit SVG circles plus their 1.5-unit stroke, in paint order.
// Keep every supplied hit coordinate; the backend remains the owner of input selection.
export function figureMarkAt(marks: FigureInteraction["marks"], x: number, y: number) {
  for (let index = marks.length - 1; index >= 0; index--) {
    const mark = marks[index];
    const radius = (mark.kind === "point" ? 5 : 9) + 0.75;
    if ((mark.cx * 1000 - x) ** 2 + (mark.cy * 1000 - y) ** 2 <= radius ** 2) {
      return mark.id;
    }
  }
  return undefined;
}
