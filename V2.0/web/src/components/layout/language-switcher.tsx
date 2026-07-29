"use client";

import { FormControl, MenuItem, Select } from "@mui/material";
import type { SelectChangeEvent } from "@mui/material/Select";
import { useLocale, useTranslations } from "next-intl";
import { useRouter } from "next/navigation";
import { useTransition } from "react";

export function LanguageSwitcher() {
  const locale = useLocale();
  const router = useRouter();
  const t = useTranslations("app");
  const [isPending, startTransition] = useTransition();

  const changeLocale = (event: SelectChangeEvent) => {
    const nextLocale = event.target.value === "zh" ? "zh" : "en";
    startTransition(async () => {
      await fetch("/api/locale", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ locale: nextLocale }),
      });
      router.refresh();
    });
  };

  return (
    <FormControl size="small" sx={{ minWidth: 112 }}>
      <Select
        aria-label={t("language")}
        disabled={isPending}
        onChange={changeLocale}
        value={locale}
      >
        <MenuItem value="en">English</MenuItem>
        <MenuItem value="zh">中文</MenuItem>
      </Select>
    </FormControl>
  );
}
