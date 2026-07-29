"use client";

import AddchartOutlinedIcon from "@mui/icons-material/AddchartOutlined";
import HelpOutlineRoundedIcon from "@mui/icons-material/HelpOutlineRounded";
import HistoryRoundedIcon from "@mui/icons-material/HistoryRounded";
import HomeOutlinedIcon from "@mui/icons-material/HomeOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import { Box, ButtonBase, Stack, Typography } from "@mui/material";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { useTranslations } from "next-intl";

const navigationItems = [
  { href: "/", labelKey: "home", icon: HomeOutlinedIcon },
  { href: "/workspace/new", labelKey: "newAnalysis", icon: AddchartOutlinedIcon },
  { href: "/history", labelKey: "history", icon: HistoryRoundedIcon },
  { href: "/settings", labelKey: "settings", icon: SettingsOutlinedIcon },
  { href: "/help", labelKey: "help", icon: HelpOutlineRoundedIcon },
] as const;

export function PrimarySidebar() {
  const pathname = usePathname();
  const t = useTranslations("app");

  return (
    <Stack
      component="nav"
      direction={{ xs: "row", md: "column" }}
      spacing={0.5}
      sx={{
        bgcolor: "background.paper",
        borderBottom: { xs: 1, md: 0 },
        borderColor: "divider",
        height: { md: "100%" },
        overflowX: "auto",
        p: 2,
      }}
    >
      {navigationItems.map((item, index) => {
        const active =
          item.href !== "/" ? pathname.startsWith(item.href) : pathname === "/";
        const Icon = item.icon;

        return (
          <ButtonBase
            key={`${item.labelKey}-${index}`}
            component={Link}
            href={item.href}
            sx={{
              borderRadius: 1.5,
              color: active ? "primary.main" : "text.secondary",
              flex: { xs: "0 0 auto", md: "0 0 auto" },
              justifyContent: "flex-start",
              minHeight: 42,
              px: 1.25,
              width: { xs: "auto", md: "100%" },
              ...(active && { bgcolor: "primary.light" }),
            }}
          >
            <Box sx={{ alignItems: "center", display: "flex", gap: 1.25 }}>
              <Icon sx={{ fontSize: 20 }} />
              <Typography
                sx={{ fontWeight: active ? 700 : 600 }}
                variant="body2"
              >
                {t(item.labelKey)}
              </Typography>
            </Box>
          </ButtonBase>
        );
      })}
    </Stack>
  );
}
