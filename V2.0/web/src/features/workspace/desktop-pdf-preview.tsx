"use client";
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, Typography } from "@mui/material";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { useLocale } from "next-intl";

export function DesktopPdfPreview({ href }: { href: string }) {
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  const [preview, setPreview] = useState<{ href: string; url?: string; error?: boolean } | null>(null);
  const desktop = useSyncExternalStore(() => () => {}, () => !!window.labvizDesktopTransport, () => false);
  const zh = useLocale().startsWith("zh");
  useEffect(() => {
    if (!open || !desktop) return;
    const close = () => setOpen(false);
    // Native PDF controls receive Escape in their own browser document.
    window.addEventListener("labviz:desktop-escape", close);
    return () => window.removeEventListener("labviz:desktop-escape", close);
  }, [open, desktop]);
  useEffect(() => {
    if (!open || !desktop) return;
    const controller = new AbortController();
    let cancelled = false;
    let objectUrl: string | undefined;
    void (async () => {
      try {
        const source = new URL(href, window.location.href);
        if (source.origin !== window.location.origin || !source.pathname.startsWith("/api/v1/projects/")) throw new Error("Invalid PDF source");
        const response = await fetch(source, { signal: controller.signal, cache: "no-store", credentials: "same-origin" });
        if (!response.ok || response.headers.get("content-type")?.split(";")[0].trim().toLowerCase() !== "application/pdf") throw new Error("PDF unavailable");
        const blob = await response.blob();
        if (await blob.slice(0, 5).text() !== "%PDF-") throw new Error("Invalid PDF content");
        if (cancelled) return;
        // Keep the authenticated API's anti-framing headers. Only the validated
        // PDF bytes receive a temporary document URL for the native PDF viewer.
        objectUrl = URL.createObjectURL(blob);
        setPreview({ href, url: objectUrl });
      } catch {
        if (!cancelled) setPreview({ href, error: true });
      }
    })();
    return () => { cancelled = true; controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [open, href, desktop]);
  if (!desktop) return null;
  const current = preview?.href === href ? preview : null;
  return <>
    <Button ref={trigger} onClick={() => { setPreview(null); setOpen(true); }}>{zh ? "预览 PDF" : "Preview PDF"}</Button>
    <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="lg" aria-labelledby="desktop-pdf-title"
      slotProps={{ transition: { onExited: () => trigger.current?.focus() } }}>
      <DialogTitle id="desktop-pdf-title">{zh ? "已保存的 PDF" : "Saved PDF"}</DialogTitle>
      <DialogContent sx={{ height: "70vh", p: 0 }}>{open && (current?.error
        ? <Alert severity="error">{zh ? "无法预览已保存的 PDF，请关闭预览后重试。" : "Could not preview the saved PDF. Close this preview and try again."}</Alert>
        : current?.url
          ? <iframe title={zh ? "已保存 PDF 预览" : "Saved PDF preview"} src={current.url} style={{ border: 0, width: "100%", height: "100%" }} />
          : <Typography role="status" sx={{ p: 2 }}>{zh ? "正在读取已保存的 PDF……" : "Loading saved PDF…"}</Typography>)}</DialogContent>
      <DialogActions><Button onClick={() => setOpen(false)}>{zh ? "返回工作区" : "Return to workspace"}</Button></DialogActions>
    </Dialog>
  </>;
}
