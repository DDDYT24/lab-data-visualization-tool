"use client";

import AccountCircleOutlinedIcon from "@mui/icons-material/AccountCircleOutlined";
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
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

import labvizLogo from "../../../../assets/labviz-logo.png";

import { EmailCodeDialog } from "@/features/auth/email-code-dialog";
import { labvizApi } from "@/lib/api/labviz-api";

import { LanguageSwitcher } from "./language-switcher";

export function AppHeader() {
  const t = useTranslations("app");
  const queryClient = useQueryClient();
  const [signInOpen, setSignInOpen] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  const authQuery = useQuery({
    queryKey: ["auth-state"],
    queryFn: ({ signal }) => labvizApi.getAuthState(signal),
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

  return (
    <>
      <Box
        component="header"
        sx={{
          bgcolor: "background.paper",
          borderBottom: 1,
          borderColor: "divider",
          position: "relative",
          zIndex: 10,
        }}
      >
        <Container maxWidth={false} sx={{ px: { xs: 2, md: 3 } }}>
          <Stack
            direction="row"
            spacing={2}
            sx={{
              alignItems: "center",
              justifyContent: "space-between",
              minHeight: 64,
            }}
          >
            <Stack direction="row" spacing={2.5} sx={{ alignItems: "center" }}>
              <Stack
                component={Link}
                direction="row"
                href="/"
                spacing={1.25}
                sx={{ alignItems: "center" }}
              >
                <Image
                  alt="LabViz"
                  priority
                  src={labvizLogo}
                  style={{
                    height: "34px",
                    objectFit: "contain",
                    width: "102px",
                  }}
                />
              </Stack>
              <Stack
                direction="row"
                spacing={0.25}
                sx={{ display: { xs: "none", md: "flex" } }}
              >
                <Button component={Link} href="/history" size="small">
                  {t("history")}
                </Button>
                <Button component={Link} href="/help" size="small">
                  {t("help")}
                </Button>
              </Stack>
            </Stack>

            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <Chip
                color="secondary"
                label={t("cloudPreview")}
                size="small"
                variant="outlined"
                sx={{ display: { xs: "none", sm: "inline-flex" } }}
              />
              <LanguageSwitcher />
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
      <EmailCodeDialog
        onClose={() => setSignInOpen(false)}
        open={signInOpen}
      />
      <Menu
        anchorEl={menuAnchor}
        onClose={() => setMenuAnchor(null)}
        open={Boolean(menuAnchor)}
      >
        <MenuItem component={Link} href="/history" onClick={() => setMenuAnchor(null)}>
          <ListItemIcon><HistoryRoundedIcon fontSize="small" /></ListItemIcon>
          {t("history")}
        </MenuItem>
        <MenuItem component={Link} href="/help" onClick={() => setMenuAnchor(null)}>
          <ListItemIcon><HelpOutlineRoundedIcon fontSize="small" /></ListItemIcon>
          {t("help")}
        </MenuItem>
        <MenuItem component={Link} href="/settings" onClick={() => setMenuAnchor(null)}>
          <ListItemIcon><SettingsOutlinedIcon fontSize="small" /></ListItemIcon>
          {t("settings")}
        </MenuItem>
        <Divider />
        {user ? (
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
