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

export const chatGptDarkPalette = {
  background: "#212121",
  border: "#424242",
  surface: "#2F2F2F",
  text: "#ECECEC",
  textSecondary: "#B4B4B4",
} as const;

export function createLabvizTheme(mode: PaletteMode) {
  const dark = mode === "dark";
  const actionContrastText = dark ? "#0D0D0D" : "#FFFFFF";
  return createTheme({
  cssVariables: {
    cssVarPrefix: "labviz",
  },
  palette: {
    mode,
    primary: {
      main: dark ? chatGptDarkPalette.text : "#2563EB",
      dark: dark ? "#D1D1D1" : "#1D4ED8",
      light: dark ? chatGptDarkPalette.border : "#EAF1FF",
      contrastText: actionContrastText,
    },
    secondary: {
      main: dark ? chatGptDarkPalette.textSecondary : "#0F766E",
      dark: dark ? "#D1D1D1" : "#0B5F59",
      light: dark ? chatGptDarkPalette.surface : "#E8F5F2",
      contrastText: actionContrastText,
    },
    background: {
      default: dark ? chatGptDarkPalette.background : "#F6F8FB",
      paper: dark ? chatGptDarkPalette.surface : "#FFFFFF",
    },
    text: {
      primary: dark ? chatGptDarkPalette.text : "#172033",
      secondary: dark ? chatGptDarkPalette.textSecondary : "#667085",
    },
    divider: dark ? chatGptDarkPalette.border : "#D8DEE8",
    success: {
      main: dark ? "#19C37D" : "#1B6848",
      dark: dark ? "#4DD8A0" : "#145038",
      light: dark ? "#18392E" : "#E9F6EF",
      contrastText: actionContrastText,
    },
    warning: {
      main: dark ? "#F4AC36" : "#B7791F",
      light: dark ? "#43351F" : "#FFF6E5",
      contrastText: dark ? "#0D0D0D" : "#FFFFFF",
    },
    error: {
      main: dark ? "#FF8583" : "#C43D4B",
      light: dark ? "#472928" : "#FFF0F2",
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
          backgroundColor: alpha(dark ? "#FFFFFF" : "#2563EB", 0.18),
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
