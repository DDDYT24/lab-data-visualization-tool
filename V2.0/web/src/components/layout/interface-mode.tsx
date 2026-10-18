"use client";

import ViewQuiltOutlinedIcon from "@mui/icons-material/ViewQuiltOutlined";
import { FormControl, IconButton, InputLabel, MenuItem, Select, Tooltip } from "@mui/material";
import { useLocale } from "next-intl";
import { createContext, useContext } from "react";
import type { InterfaceMode } from "@/features/settings/interface-mode";

export const InterfaceModeContext = createContext<{
  mode: InterfaceMode;
  setMode: (mode: InterfaceMode) => void;
  persistent: boolean;
  ready: boolean;
}>({ mode: "classic", setMode: () => {}, persistent: true, ready: false });

export function useInterfaceMode() { return useContext(InterfaceModeContext); }

export function InterfaceModeToggle() {
  const { mode, setMode, ready } = useInterfaceMode();
  const zh = useLocale().startsWith("zh");
  const label = mode === "classic"
    ? (zh ? "切换到概览 A 界面" : "Switch to overview A interface")
    : (zh ? "切换到经典界面" : "Switch to classic interface");
  return <Tooltip title={label}><IconButton disabled={!ready} aria-label={label} aria-pressed={mode === "overview"}
    onClick={() => setMode(mode === "classic" ? "overview" : "classic")}
    sx={{ display: { xs: "none", lg: "inline-flex" }, minWidth: 40, minHeight: 44 }}><ViewQuiltOutlinedIcon /></IconButton></Tooltip>;
}

export function InterfaceModeSelect() {
  const { mode, setMode, ready } = useInterfaceMode();
  const zh = useLocale().startsWith("zh");
  const label = zh ? "界面模式" : "Interface mode";
  return <FormControl fullWidth size="small"><InputLabel id="interface-mode-label">{label}</InputLabel>
    <Select disabled={!ready} label={label} labelId="interface-mode-label" value={mode}
      onChange={(event) => setMode(event.target.value as InterfaceMode)}>
      <MenuItem value="classic">{zh ? "经典" : "Classic"}</MenuItem>
      <MenuItem value="overview">{zh ? "概览 A" : "Overview A"}</MenuItem>
    </Select></FormControl>;
}
