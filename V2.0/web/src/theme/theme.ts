"use client";

import { alpha, createTheme } from "@mui/material/styles";

export const chartPalette = [
  "#2563EB",
  "#0F766E",
  "#C2415C",
  "#7C3AED",
  "#B7791F",
  "#137C8B",
] as const;

export const theme = createTheme({
  cssVariables: {
    cssVarPrefix: "labviz",
  },
  palette: {
    mode: "light",
    primary: {
      main: "#2563EB",
      dark: "#1D4ED8",
      light: "#EAF1FF",
      contrastText: "#FFFFFF",
    },
    secondary: {
      main: "#0F766E",
      dark: "#0B5F59",
      light: "#E8F5F2",
      contrastText: "#FFFFFF",
    },
    background: {
      default: "#F6F8FB",
      paper: "#FFFFFF",
    },
    text: {
      primary: "#172033",
      secondary: "#667085",
    },
    divider: "#D8DEE8",
    success: {
      main: "#1B6848",
      dark: "#145038",
      light: "#E9F6EF",
      contrastText: "#FFFFFF",
    },
    warning: {
      main: "#B7791F",
      light: "#FFF6E5",
    },
    error: {
      main: "#C43D4B",
      light: "#FFF0F2",
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
