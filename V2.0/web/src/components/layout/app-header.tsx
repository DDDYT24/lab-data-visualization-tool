"use client";

import AccountCircleOutlinedIcon from "@mui/icons-material/AccountCircleOutlined";
import Brightness4RoundedIcon from "@mui/icons-material/Brightness4Rounded";
import Brightness7RoundedIcon from "@mui/icons-material/Brightness7Rounded";
import HelpOutlineRoundedIcon from "@mui/icons-material/HelpOutlineRounded";
import HistoryRoundedIcon from "@mui/icons-material/HistoryRounded";
import LoginRoundedIcon from "@mui/icons-material/LoginRounded";
import LogoutRoundedIcon from "@mui/icons-material/LogoutRounded";
import MenuRoundedIcon from "@mui/icons-material/MenuRounded";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import {
  Box,
  Button,
  Chip,
  Container,
  Divider,
  IconButton,
  ListItemIcon,
  Menu,
  MenuItem,
  Stack,
  Tooltip,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { EmailCodeDialog } from "@/features/auth/email-code-dialog";
import {
  loadUserPreferences,
  saveUserPreferences,
} from "@/features/settings/user-preferences";
import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

import { LanguageSwitcher } from "./language-switcher";
import { LabvizLogo } from "./labviz-logo";

export function AppHeader({ landing = false }: { landing?: boolean }) {
  const t = useTranslations("app");
  const pathname = usePathname();
  const theme = useTheme();
  const queryClient = useQueryClient();
  const [signInOpen, setSignInOpen] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  const authQuery = useQuery({
    queryKey: ["auth-state"],
    // This small public request is intentionally allowed to finish across route
    // changes. WebKit reports a cancelled same-origin fetch as a page error,
    // while React Query still ignores a stale result after navigation.
    queryFn: () => labvizApi.getAuthState(),
  });
  const logoutMutation = useMutation({
    mutationFn: () => labvizApi.logout(),
    onSuccess: (state) => {
      queryClient.setQueryData(["auth-state"], state);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
      setMenuAnchor(null);
    },
  });
  const user = authQuery.data?.authenticated ? authQuery.data.user : null;
  const localUser = user?.id === "local-profile" || (
    authQuery.error instanceof LabVizApiError &&
    authQuery.error.code === "local-session-required"
  );
  const toggleTheme = () => {
    saveUserPreferences({
      ...loadUserPreferences(),
      appearance: theme.palette.mode === "dark" ? "light" : "dark",
    });
  };

  return (
    <>
      <Box
        component="header"
        sx={{
          bgcolor: landing ? "background.default" : "background.paper",
          borderBottom: landing ? 0 : 1,
          borderColor: "divider",
          position: "relative",
          zIndex: 10,
        }}
      >
        <Container maxWidth={landing ? "xl" : false} sx={{ px: { xs: 2, md: 3 } }}>
          <Stack
            direction="row"
            spacing={2}
            sx={{
              alignItems: "center",
              justifyContent: "space-between",
              minHeight: landing ? 72 : 64,
            }}
          >
            <Stack direction="row" spacing={2.5} sx={{ alignItems: "center" }}>
              <Stack
                component={Link}
                prefetch={false}
                direction="row"
                href="/"
                spacing={1.25}
                sx={{ alignItems: "center" }}
              >
                <LabvizLogo />
              </Stack>
              <Stack
                direction="row"
                spacing={0.25}
                sx={{ display: landing ? "none" : { xs: "none", md: "flex" } }}
              >
                <Button component={Link} href="/history" prefetch={false} size="small">
                  {t("history")}
                </Button>
                <Button component={Link} href="/help" prefetch={false} size="small">
                  {t("help")}
                </Button>
                <Button component={Link} href="/about" prefetch={false} size="small">
                  {t("navAbout")}
                </Button>
              </Stack>
            </Stack>

            {landing ? (
              <Stack
                component="nav"
                aria-label={t("primaryNavigation")}
                direction="row"
                spacing={0.5}
                sx={{
                  alignItems: "center",
                  display: { xs: "none", md: "flex" },
                  left: "50%",
                  position: "absolute",
                  top: "50%",
                  transform: "translate(-50%, -50%)",
                }}
              >
                {[
                  { href: "/", label: t("navWorkspace"), active: pathname === "/" },
                  { href: "/history", label: t("navProjects"), active: pathname === "/history" },
                  { href: "/help", label: t("navDocs"), active: pathname === "/help" },
                  { href: "/about", label: t("navAbout"), active: pathname === "/about" },
                ].map((item) => (
                  <Button
                    key={item.href}
                    aria-current={item.active ? "page" : undefined}
                    component={Link}
                    href={item.href}
                    prefetch={false}
                    sx={{
                      borderRadius: 2,
                      color: item.active ? "primary.main" : "text.secondary",
                      fontWeight: item.active ? 650 : 550,
                      minWidth: "auto",
                      px: 1.5,
                    }}
                  >
                    {item.label}
                  </Button>
                ))}
              </Stack>
            ) : null}

            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              {!landing ? (
                <Chip
                  color="secondary"
                  label={t("cloudPreview")}
                  size="small"
                  variant="outlined"
                  sx={{ display: { xs: "none", sm: "inline-flex" } }}
                />
              ) : null}
              <Tooltip
                title={
                  theme.palette.mode === "dark"
                    ? t("switchToLightTheme")
                    : t("switchToDarkTheme")
                }
              >
                <IconButton
                  aria-label={
                    theme.palette.mode === "dark"
                      ? t("switchToLightTheme")
                      : t("switchToDarkTheme")
                  }
                  onClick={toggleTheme}
                  size="small"
                >
                  {theme.palette.mode === "dark" ? (
                    <Brightness7RoundedIcon />
                  ) : (
                    <Brightness4RoundedIcon />
                  )}
                </IconButton>
              </Tooltip>
              <LanguageSwitcher />
              {authQuery.isPending || localUser ? null : landing ? (
                <Tooltip title={user?.email ?? t("signIn")}>
                  <IconButton
                    aria-label={user?.email ?? t("signIn")}
                    onClick={(event) =>
                      user
                        ? setMenuAnchor(event.currentTarget)
                        : setSignInOpen(true)
                    }
                    sx={{ display: { xs: "none", sm: "inline-flex" } }}
                  >
                    <AccountCircleOutlinedIcon />
                  </IconButton>
                </Tooltip>
              ) : (
                <Button
                  onClick={(event) =>
                    user
                      ? setMenuAnchor(event.currentTarget)
                      : setSignInOpen(true)
                  }
                  startIcon={user ? <AccountCircleOutlinedIcon /> : undefined}
                  sx={{ display: { xs: "none", sm: "inline-flex" } }}
                >
                  {user?.email ?? t("signIn")}
                </Button>
              )}
              <IconButton
                aria-label={t("menu")}
                onClick={(event) => setMenuAnchor(event.currentTarget)}
                sx={{ display: { xs: "inline-flex", sm: "none" } }}
              >
                <MenuRoundedIcon />
              </IconButton>
            </Stack>
          </Stack>
        </Container>
      </Box>
      {!localUser ? (
        <EmailCodeDialog onClose={() => setSignInOpen(false)} open={signInOpen} />
      ) : null}
      <Menu
        anchorEl={menuAnchor}
        onClose={() => setMenuAnchor(null)}
        open={Boolean(menuAnchor)}
      >
        <MenuItem component={Link} href="/history" prefetch={false} onClick={() => setMenuAnchor(null)}>
          <ListItemIcon><HistoryRoundedIcon fontSize="small" /></ListItemIcon>
          {t("history")}
        </MenuItem>
        <MenuItem component={Link} href="/help" prefetch={false} onClick={() => setMenuAnchor(null)}>
          <ListItemIcon><HelpOutlineRoundedIcon fontSize="small" /></ListItemIcon>
          {t("help")}
        </MenuItem>
        <MenuItem component={Link} href="/about" prefetch={false} onClick={() => setMenuAnchor(null)}>
          {t("navAbout")}
        </MenuItem>
        <MenuItem component={Link} href="/settings" prefetch={false} onClick={() => setMenuAnchor(null)}>
          <ListItemIcon><SettingsOutlinedIcon fontSize="small" /></ListItemIcon>
          {t("settings")}
        </MenuItem>
        {!localUser ? <Divider /> : null}
        {localUser ? null : user ? (
          <MenuItem disabled={logoutMutation.isPending} onClick={() => logoutMutation.mutate()}>
            <ListItemIcon><LogoutRoundedIcon fontSize="small" /></ListItemIcon>
            {t("signOut")}
          </MenuItem>
        ) : (
          <MenuItem
            onClick={() => {
              setMenuAnchor(null);
              setSignInOpen(true);
            }}
          >
            <ListItemIcon><LoginRoundedIcon fontSize="small" /></ListItemIcon>
            {t("signIn")}
          </MenuItem>
        )}
      </Menu>
    </>
  );
}
