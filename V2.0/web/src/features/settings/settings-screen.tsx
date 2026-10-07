"use client";

import ComputerOutlinedIcon from "@mui/icons-material/ComputerOutlined";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import PaletteOutlinedIcon from "@mui/icons-material/PaletteOutlined";
import {
  Alert,
  Box,
  Button,
  Divider,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Switch,
  Typography,
} from "@mui/material";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { type ReactNode, useEffect, useState } from "react";

import { LanguageSwitcher } from "@/components/layout/language-switcher";

import {
  defaultUserPreferences,
  loadUserPreferences,
  saveUserPreferences,
  type UserPreferences,
} from "./user-preferences";

type SettingsRowProps = {
  title: string;
  description: string;
  control: ReactNode;
};

function SettingsRow({ title, description, control }: SettingsRowProps) {
  return (
    <Stack
      direction={{ xs: "column", sm: "row" }}
      spacing={2}
      sx={{
        alignItems: { sm: "center" },
        justifyContent: "space-between",
        py: 2,
      }}
    >
      <Box sx={{ maxWidth: 620 }}>
        <Typography sx={{ fontWeight: 700 }} variant="body2">
          {title}
        </Typography>
        <Typography color="text.secondary" sx={{ mt: 0.5 }} variant="body2">
          {description}
        </Typography>
      </Box>
      <Box sx={{ flex: "0 0 auto", minWidth: { sm: 220 } }}>{control}</Box>
    </Stack>
  );
}

export function SettingsScreen() {
  const t = useTranslations("settings");
  // Keep server and first client render identical; localStorage is available after hydration.
  const [preferences, setPreferences] = useState<UserPreferences>(defaultUserPreferences);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setPreferences(loadUserPreferences());
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const updatePreferences = (patch: Partial<UserPreferences>) => {
    const next = { ...preferences, ...patch };
    setPreferences(next);
    saveUserPreferences(next);
    setSaved(true);
  };

  return (
    <Stack spacing={3} sx={{ maxWidth: 980 }}>
      <Box>
        <Typography component="h1" sx={{ fontSize: 28, fontWeight: 750 }}>
          {t("title")}
        </Typography>
        <Typography color="text.secondary" sx={{ mt: 0.5 }} variant="body2">
          {t("description")}
        </Typography>
      </Box>

      <Paper sx={{ border: 1, borderColor: "divider", px: 3, py: 1 }}>
        <Typography component="h2" sx={{ fontWeight: 750, pt: 2 }}>
          {t("languageSection")}
        </Typography>
        <SettingsRow
          control={
            <FormControl fullWidth size="small">
              <InputLabel id="appearance-label">{t("appearance")}</InputLabel>
              <Select
                label={t("appearance")}
                labelId="appearance-label"
                onChange={(event) =>
                  updatePreferences({
                    appearance: event.target.value as UserPreferences["appearance"],
                  })
                }
                startAdornment={<PaletteOutlinedIcon sx={{ mr: 1 }} />}
                value={preferences.appearance}
              >
                <MenuItem value="system">{t("appearanceOptions.system")}</MenuItem>
                <MenuItem value="light">{t("appearanceOptions.light")}</MenuItem>
                <MenuItem value="dark">{t("appearanceOptions.dark")}</MenuItem>
              </Select>
            </FormControl>
          }
          description={t("appearanceDescription")}
          title={t("appearanceTitle")}
        />
        <Divider />
        <SettingsRow
          control={<LanguageSwitcher />}
          description={t("interfaceLanguageDescription")}
          title={t("interfaceLanguage")}
        />
        <Divider />
        <SettingsRow
          control={
            <FormControl fullWidth size="small">
              <InputLabel id="figure-language-label">{t("figureLanguage")}</InputLabel>
              <Select
                label={t("figureLanguage")}
                labelId="figure-language-label"
                onChange={(event) =>
                  updatePreferences({
                    figureLanguage: event.target.value as UserPreferences["figureLanguage"],
                  })
                }
                value={preferences.figureLanguage}
              >
                <MenuItem value="same">{t("sameAsInterface")}</MenuItem>
                <MenuItem value="en">English</MenuItem>
                <MenuItem value="zh">简体中文</MenuItem>
              </Select>
            </FormControl>
          }
          description={t("figureLanguageDescription")}
          title={t("figureLanguageTitle")}
        />
        <Divider />
        <SettingsRow
          control={
            <FormControl fullWidth size="small">
              <InputLabel id="figure-font-label">{t("font")}</InputLabel>
              <Select
                label={t("font")}
                labelId="figure-font-label"
                onChange={(event) =>
                  updatePreferences({
                    fontFamily: event.target.value as UserPreferences["fontFamily"],
                  })
                }
                value={preferences.fontFamily}
              >
                <MenuItem value="Arial">Arial</MenuItem>
                <MenuItem value="Times New Roman">Times New Roman</MenuItem>
              </Select>
            </FormControl>
          }
          description={t("fontDescription")}
          title={t("fontTitle")}
        />
        <Divider />
        <SettingsRow
          control={
            <FormControl fullWidth size="small">
              <InputLabel id="default-size-label">{t("figureSize")}</InputLabel>
              <Select
                label={t("figureSize")}
                labelId="default-size-label"
                onChange={(event) =>
                  updatePreferences({
                    sizePreset: event.target.value as UserPreferences["sizePreset"],
                  })
                }
                value={preferences.sizePreset}
              >
                <MenuItem value="single-column">{t("singleColumn")}</MenuItem>
                <MenuItem value="double-column">{t("doubleColumn")}</MenuItem>
                <MenuItem value="a4">{t("a4")}</MenuItem>
                <MenuItem value="custom">{t("custom")}</MenuItem>
              </Select>
            </FormControl>
          }
          description={t("figureSizeDescription")}
          title={t("figureSizeTitle")}
        />
        <Divider />
        <SettingsRow
          control={
            <Stack direction="row" spacing={1}>
              <FormControl fullWidth size="small">
                <InputLabel id="default-unit-label">{t("unit")}</InputLabel>
                <Select
                  label={t("unit")}
                  labelId="default-unit-label"
                  onChange={(event) =>
                    updatePreferences({
                      unit: event.target.value as UserPreferences["unit"],
                    })
                  }
                  value={preferences.unit}
                >
                  <MenuItem value="mm">mm</MenuItem>
                  <MenuItem value="cm">cm</MenuItem>
                  <MenuItem value="in">in</MenuItem>
                </Select>
              </FormControl>
              <FormControl fullWidth size="small">
                <InputLabel id="default-dpi-label">DPI</InputLabel>
                <Select
                  label="DPI"
                  labelId="default-dpi-label"
                  onChange={(event) =>
                    updatePreferences({ dpi: Number(event.target.value) as 300 | 600 })
                  }
                  value={String(preferences.dpi)}
                >
                  <MenuItem value="300">300</MenuItem>
                  <MenuItem value="600">600</MenuItem>
                </Select>
              </FormControl>
            </Stack>
          }
          description={t("resolutionDescription")}
          title={t("resolutionTitle")}
        />
        <Divider />
        <SettingsRow
          control={
            <FormControlLabel
              control={
                <Switch
                  checked={preferences.grayscalePreview}
                  onChange={(event) =>
                    updatePreferences({ grayscalePreview: event.target.checked })
                  }
                />
              }
              label={t("enable")}
            />
          }
          description={t("grayscaleDescription")}
          title={t("grayscaleTitle")}
        />
      </Paper>

      <Paper sx={{ border: 1, borderColor: "divider", px: 3, py: 1 }}>
        <Typography component="h2" sx={{ fontWeight: 750, pt: 2 }}>
          {t("privacySection")}
        </Typography>
        <SettingsRow
          control={
            <Button
              component={Link}
              href="/history"
              startIcon={<ComputerOutlinedIcon />}
              variant="outlined"
            >
              {t("viewLocal")}
            </Button>
          }
          description={t("localHistoryDescription")}
          title={t("localHistory")}
        />
        <Divider />
        <SettingsRow
          control={
            <Button
              component={Link}
              href="/help"
              startIcon={<LockOutlinedIcon />}
              variant="outlined"
            >
              {t("retentionDetails")}
            </Button>
          }
          description={t("retentionDescription")}
          title={t("retentionTitle")}
        />
      </Paper>

      <Alert severity={saved ? "success" : "info"}>
        {saved
          ? t("saved")
          : t("defaultsNote")}
      </Alert>
    </Stack>
  );
}
