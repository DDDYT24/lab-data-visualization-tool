import { describe, expect, it } from "vitest";
import matrix from "../../../contracts/fixtures/v24/figure-spec.json";
import { figureSpecSchema, parseFigureSpec, projectBoard, upgradeBoard, type FigureSpec } from "./figure-spec";
import type { FigureBoardSpec } from "./research-figure-board";

describe("FigureSpec v2", () => {
  it.each(matrix.roundTrip)("upgrades and projects v1 board %#", (pair) => {
    const board = pair.v1 as FigureBoardSpec;
    expect(upgradeBoard(board)).toEqual(pair.v2);
    expect(projectBoard(parseFigureSpec(pair.v2))).toEqual(board);
  });

  it.each(matrix.onlyV2)("accepts v2-only figure %# and gives it no v1 view", (body) => {
    expect(projectBoard(parseFigureSpec(body))).toBeNull();
  });

  it.each(matrix.invalid)("rejects invalid figure %#", (body) => {
    expect(figureSpecSchema.safeParse(body).success).toBe(false);
  });

  it("keeps key order stable so the projection can compare whole figures", () => {
    const figure = matrix.roundTrip[1].v2 as unknown as FigureSpec;
    expect(JSON.stringify(upgradeBoard(projectBoard(figure)!))).toBe(JSON.stringify(figure));
  });
});
