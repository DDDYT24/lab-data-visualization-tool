"use client";

import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import { Alert, Button } from "@mui/material";
import { useEffect, useState } from "react";

const STORAGE_KEY = "labviz:dismissed-tips:v1";
const RESET_EVENT = "labviz:tips-reset";

function readDismissedTips(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]");
    return Array.isArray(value) && value.every((item) => typeof item === "string") ? value : [];
  } catch {
    return [];
  }
}

function writeDismissedTips(tips: string[]) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(tips));
}

export function resetDismissedTips() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
  window.dispatchEvent(new Event(RESET_EVENT));
}

export function ContextualTip({
  body,
  dismissLabel,
  id,
}: {
  body: string;
  dismissLabel: string;
  id: string;
}) {
  const [dismissed, setDismissed] = useState<boolean | null>(null);

  useEffect(() => {
    const refresh = () => setDismissed(readDismissedTips().includes(id));
    refresh();
    window.addEventListener(RESET_EVENT, refresh);
    return () => window.removeEventListener(RESET_EVENT, refresh);
  }, [id]);

  if (dismissed !== false) return null;

  return (
    <Alert
      action={
        <Button
          aria-label={dismissLabel}
          color="inherit"
          onClick={() => {
            writeDismissedTips([...new Set([...readDismissedTips(), id])]);
            setDismissed(true);
          }}
          size="small"
          startIcon={<CloseRoundedIcon />}
        >
          {dismissLabel}
        </Button>
      }
      severity="info"
    >
      {body}
    </Alert>
  );
}
