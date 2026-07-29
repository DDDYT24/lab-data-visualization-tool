// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from "vitest";

import { defaultChartSpec } from "@/domain/chart-spec";

import {
  chartWithUserPreferences,
  defaultUserPreferences,
  loadUserPreferences,
  saveUserPreferences,
} from "./user-preferences";

describe("user preferences", () => {
  beforeEach(() => window.localStorage.clear());

  it("falls back safely when no browser defaults exist", () => {
    expect(loadUserPreferences()).toEqual(defaultUserPreferences);
  });

  it("persists publication defaults and applies them to a new chart", () => {
    saveUserPreferences({
      ...defaultUserPreferences,
      dpi: 600,
      fontFamily: "Times New Roman",
      grayscalePreview: true,
      sizePreset: "single-column",
      unit: "cm",
    });

    const chart = chartWithUserPreferences(defaultChartSpec);
    expect(chart.export).toMatchObject({
      dpi: 600,
      fontFamily: "Times New Roman",
      grayscalePreview: true,
      sizePreset: "single-column",
      unit: "cm",
    });
  });

  it("removes malformed stored settings instead of breaking the workspace", () => {
    window.localStorage.setItem("labviz:user-preferences:v1", "not-json");
    expect(loadUserPreferences()).toEqual(defaultUserPreferences);
    expect(window.localStorage.length).toBe(0);
  });
});
