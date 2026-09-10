import { z } from "zod";

const axisSchema = z.object({
  field: z.string().min(1),
  title: z.string().max(120),
  unit: z.string().max(40),
});

const seriesSchema = z.object({
  field: z.string().min(1),
  label: z.string().min(1).max(120),
  color: z.string().regex(/^#[0-9A-Fa-f]{6}$/),
  lineStyle: z.enum(["solid", "dashed", "dotted", "dashdot"]).default("solid"),
  panel: z.number().int().min(1).max(4).default(1),
  yAxis: z.enum(["primary", "secondary"]).default("primary"),
});

const fittingSchema = z.object({
  model: z
    .enum(["none", "linear", "polynomial", "exponential", "logarithmic", "power"])
    .default("none"),
  polynomialOrder: z.union([z.literal(1), z.literal(2), z.literal(3)]).default(2),
  fitMethod: z
    .enum(["ordinary-least-squares", "weighted-least-squares"])
    .default("ordinary-least-squares"),
  showEquation: z.boolean().default(true),
  showRSquared: z.boolean().default(true),
  confidenceBand: z.boolean().default(false),
  confidenceMethod: z.enum(["student-t", "bootstrap"]).default("student-t"),
  confidenceLevel: z.union([z.literal(90), z.literal(95), z.literal(99)]).default(95),
});

const uncertaintySchema = z.object({
  mode: z
    .enum([
      "none",
      "standard-deviation",
      "standard-error",
      "confidence-interval",
      "column",
    ])
    .default("none"),
  errorField: z.string().nullable().default(null),
  confidenceLevel: z.union([z.literal(90), z.literal(95), z.literal(99)]).default(95),
});

const secondaryYAxisSchema = z.object({
  enabled: z.boolean().default(false),
  field: z.string().nullable().default(null),
  title: z.string().max(120).default("Secondary response"),
  unit: z.string().max(40).default(""),
});

export const chartSpecSchema = z.object({
  schemaVersion: z.literal(1),
  type: z.enum([
    "line",
    "scatter",
    "bar",
    "histogram",
    "box",
    "heatmap",
    "surface3d",
  ]),
  title: z.string().max(200),
  subtitle: z.string().max(200).default(""),
  xAxis: axisSchema,
  yAxis: axisSchema,
  series: z.array(seriesSchema).min(1).max(20),
  groupField: z.string().min(1).nullable().default(null),
  panelCount: z.number().int().min(1).max(4),
  fitting: fittingSchema.prefault({}),
  uncertainty: uncertaintySchema.prefault({}),
  secondaryYAxis: secondaryYAxisSchema.prefault({}),
  export: z.object({
    format: z.enum(["png", "svg", "pdf"]),
    dpi: z.union([z.literal(300), z.literal(600)]),
    sizePreset: z.enum(["single-column", "double-column", "a4", "custom"]),
    grayscalePreview: z.boolean(),
    width: z.number().positive().max(2_000).nullable().default(null),
    height: z.number().positive().max(2_000).nullable().default(null),
    unit: z.enum(["mm", "cm", "in"]).default("mm"),
    fontFamily: z.enum(["Arial", "Times New Roman"]).default("Arial"),
    fontSize: z.number().min(6).max(36).default(10),
    lineWidth: z.number().min(0.25).max(10).default(1.5),
    markerSize: z.number().min(1).max(20).default(4),
    legendPosition: z
      .enum(["auto", "top", "bottom", "left", "right", "none"])
      .default("auto"),
    transparentBackground: z.boolean().default(false),
    gridVisible: z.boolean().default(true),
    backgroundColor: z.string().regex(/^#[0-9A-Fa-f]{6}$/).default("#FFFFFF"),
  }),
});

export type ChartSpec = z.infer<typeof chartSpecSchema>;

export const defaultChartSpec: ChartSpec = {
  schemaVersion: 1,
  type: "line",
  title: "Response over time",
  subtitle: "",
  xAxis: {
    field: "time",
    title: "Time",
    unit: "min",
  },
  yAxis: {
    field: "response",
    title: "Response",
    unit: "mV",
  },
  series: [
    {
      field: "response",
      label: "Sample A",
      color: "#2563EB",
      lineStyle: "solid",
      panel: 1,
      yAxis: "primary",
    },
  ],
  groupField: null,
  panelCount: 1,
  fitting: {
    model: "none",
    polynomialOrder: 2,
    fitMethod: "ordinary-least-squares",
    showEquation: true,
    showRSquared: true,
    confidenceBand: false,
    confidenceMethod: "student-t",
    confidenceLevel: 95,
  },
  uncertainty: {
    mode: "none",
    errorField: null,
    confidenceLevel: 95,
  },
  secondaryYAxis: {
    enabled: false,
    field: null,
    title: "Secondary response",
    unit: "",
  },
  export: {
    format: "png",
    dpi: 300,
    sizePreset: "double-column",
    grayscalePreview: false,
    width: 180,
    height: 120,
    unit: "mm",
    fontFamily: "Arial",
    fontSize: 10,
    lineWidth: 1.5,
    markerSize: 4,
    legendPosition: "auto",
    transparentBackground: false,
    gridVisible: true,
    backgroundColor: "#FFFFFF",
  },
};
