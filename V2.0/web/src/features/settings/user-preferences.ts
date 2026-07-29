import { z } from "zod";

import { defaultChartSpec, type ChartSpec } from "@/domain/chart-spec";

const storageKey = "labviz:user-preferences:v1";

const userPreferencesSchema = z.object({
  figureLanguage: z.enum(["same", "en", "zh"]),
  fontFamily: z.enum(["Arial", "Times New Roman"]),
  sizePreset: z.enum(["single-column", "double-column", "a4", "custom"]),
  unit: z.enum(["mm", "cm", "in"]),
  dpi: z.union([z.literal(300), z.literal(600)]),
  grayscalePreview: z.boolean(),
});

export type UserPreferences = z.infer<typeof userPreferencesSchema>;

export const defaultUserPreferences: UserPreferences = {
  figureLanguage: "same",
  fontFamily: defaultChartSpec.export.fontFamily,
  sizePreset: defaultChartSpec.export.sizePreset,
  unit: defaultChartSpec.export.unit,
  dpi: defaultChartSpec.export.dpi,
  grayscalePreview: defaultChartSpec.export.grayscalePreview,
};

export function loadUserPreferences(): UserPreferences {
  if (typeof window === "undefined") return defaultUserPreferences;
  const stored = window.localStorage.getItem(storageKey);
  if (!stored) return defaultUserPreferences;
  try {
    return userPreferencesSchema.parse(JSON.parse(stored));
  } catch {
    window.localStorage.removeItem(storageKey);
    return defaultUserPreferences;
  }
}

export function saveUserPreferences(preferences: UserPreferences) {
  if (typeof window !== "undefined") {
    window.localStorage.setItem(storageKey, JSON.stringify(preferences));
  }
}

export function chartWithUserPreferences(chart: ChartSpec): ChartSpec {
  const preferences = loadUserPreferences();
  return {
    ...chart,
    export: {
      ...chart.export,
      dpi: preferences.dpi,
      fontFamily: preferences.fontFamily,
      grayscalePreview: preferences.grayscalePreview,
      sizePreset: preferences.sizePreset,
      unit: preferences.unit,
    },
  };
}
