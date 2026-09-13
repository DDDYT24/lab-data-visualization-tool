"use client";

import type { PaletteMode } from "@mui/material";
import { alpha, createTheme } from "@mui/material/styles";

export const chartPalette = [
  "#2563EB",
  "#0F766E",
  "#C2415C",
  "#7C3AED",
  "#B7791F",
  "#137C8B",
] as const;

export const darkChartPalette = [
  "#8DB7FF",
  "#69D2C7",
  "#FF9BAA",
  "#C7A5FF",
  "#F4C56F",
  "#67DCE8",
] as const;

export function createLabvizTheme(mode: PaletteMode) {
  const dark = mode === "dark";
  const actionContrastText = dark ? "#0F172A" : "#FFFFFF";
  return createTheme({
  cssVariables: {
    cssVarPrefix: "labviz",
  },
  palette: {
    mode,
    primary: {
      main: dark ? "#8DB7FF" : "#2563EB",
      dark: dark ? "#B7D0FF" : "#1D4ED8",
      light: dark ? "#203A6C" : "#EAF1FF",
      contrastText: actionContrastText,
    },
    secondary: {
      main: dark ? "#69D2C7" : "#0F766E",
      dark: dark ? "#9AE5DD" : "#0B5F59",
      light: dark ? "#164B49" : "#E8F5F2",
      contrastText: actionContrastText,
    },
    background: {
      default: dark ? "#0F172A" : "#F6F8FB",
      paper: dark ? "#172033" : "#FFFFFF",
    },
    text: {
      primary: dark ? "#F4F7FB" : "#172033",
      secondary: dark ? "#B6C2D5" : "#667085",
    },
    divider: dark ? "#334155" : "#D8DEE8",
    success: {
      main: dark ? "#75D6A2" : "#1B6848",
      dark: dark ? "#A2E7BF" : "#145038",
      light: dark ? "#183C31" : "#E9F6EF",
      contrastText: actionContrastText,
    },
    warning: {
      main: dark ? "#F4C56F" : "#B7791F",
      light: dark ? "#4A3820" : "#FFF6E5",
      contrastText: dark ? "#0F172A" : "#FFFFFF",
    },
    error: {
      main: dark ? "#FF9BAA" : "#C43D4B",
      light: dark ? "#4C2530" : "#FFF0F2",
      contrastText: actionContrastText,
    },
  },
  typography: {
    fontFamily:
      'Inter, "Noto Sans SC", "Microsoft YaHei UI", "Segoe UI", Arial, sans-serif',
    h1: {
      fontSize: "clamp(2.25rem, 4vw, 4rem)",
      fontWeight: 720,
      letterSpacing: "-0.035em",
      lineHeight: 1.08,
    },
    h2: {
      fontSize: "clamp(1.55rem, 2.4vw, 2.25rem)",
      fontWeight: 700,
      letterSpacing: "-0.02em",
      lineHeight: 1.2,
    },
    h3: {
      fontSize: "1.125rem",
      fontWeight: 680,
      lineHeight: 1.35,
    },
    button: {
      fontWeight: 650,
      textTransform: "none",
    },
    body1: {
      lineHeight: 1.65,
    },
    body2: {
      lineHeight: 1.55,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        "::selection": {
          backgroundColor: alpha("#2563EB", 0.18),
        },
        body: {
          minWidth: 320,
        },
      },
    },
    MuiButton: {
      defaultProps: {
        disableElevation: true,
      },
      styleOverrides: {
        root: {
          minHeight: 40,
          borderRadius: 7,
          paddingInline: 18,
        },
      },
    },
    MuiPaper: {
      defaultProps: {
        elevation: 0,
      },
      styleOverrides: {
        root: {
          backgroundImage: "none",
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 650,
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 10,
        },
      },
    },
    MuiTooltip: {
      defaultProps: {
        arrow: true,
      },
    },
  },
  });
}

export const theme = createLabvizTheme("light");
