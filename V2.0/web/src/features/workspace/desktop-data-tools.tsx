"use client";
import { Alert, Button, MenuItem, Stack, TextField, Typography } from "@mui/material";
import { useEffect, useState } from "react";
import { useLocale } from "next-intl";
type Operation = { state: string; action: string; message?: string; backupDirectory?: string; copyId?: string; restoredProfileRoot?: string };
type Profiles = { activeCopyId: string | null; nextCopyId: string | null; copies: { copyId: string; profileId: string }[]; unavailableCopies: number };
type Status = { operation: Operation | null; backups: { backupId: string; createdUtcSeconds: number }[]; profiles?: Profiles };
async function call(route: string, value?: object) {
  const bridge = window.labvizDesktopTransport!;
  const response = await fetch(new URL(route, bridge.url), { method: value ? "POST" : "GET",
    headers: { "Content-Type": "application/json", "X-LabViz-Desktop": bridge.token },
    body: value ? JSON.stringify(value) : undefined, signal: AbortSignal.timeout(30000) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.message || "The data operation could not start.");
  return result;
}
export function DesktopDataTools() {
  const zh = useLocale().startsWith("zh");
  const t = (en: string, cn: string) => zh ? cn : en;
  const [status, setStatus] = useState<Status>({ operation: null, backups: [] });
  const [backupId, setBackupId] = useState("");
  const [copyId, setCopyId] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    let mounted = true;
    const refresh = async () => {
      try { const value = await call("/maintenance/status") as Status; if (mounted) { setStatus(value); setError(""); } }
      catch (reason) { if (mounted) setError(String(reason)); }
    };
    void refresh(); const timer = setInterval(() => void refresh(), 1500);
    return () => { mounted = false; clearInterval(timer); };
  }, []);
  async function submit(action: string) {
    setBusy(true); setError(""); setMessage("");
    try { const operation = await call("/maintenance/submit", { action, ...(backupId ? { backupId } : {}) }); setStatus(previous => ({ ...previous, operation })); }
    catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  }
  async function selectCopy(value: string | null) {
    setBusy(true); setError(""); setMessage("");
    try {
      await call("/maintenance/select-copy", { copyId: value });
      setStatus(await call("/maintenance/status") as Status);
      setMessage(t("Exit LabViz and reopen it to use the selected profile. Your current profile is retained.", "退出并重新打开 LabViz 后将使用所选资料，当前资料会保留。"));
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  }
  const waiting = status.operation?.state === "waiting";
  const disabled = busy || waiting;
  const copyLabel = (value: string | null) => value ? `${t("Restored copy", "恢复副本")} ${value.slice(0, 8)}` : t("Original profile", "原资料");
  return <Stack spacing={2} data-testid="desktop-data-tools">
    <Typography component="h3" variant="subtitle1">{t("Desktop data and drafts", "桌面数据与草稿")}</Typography>
    <Typography>{t("Backup saves the database, object files, preferences and draft sources. The app saves your edits and briefly restarts local services. Finish uploads, calculations and downloads first.", "备份包含数据库、对象文件、偏好和草稿来源。应用会保存编辑内容并短暂重启本地服务，请先完成上传、计算和下载。")}</Typography>
    <Button disabled={busy || waiting} variant="outlined" onClick={() => void submit("backup")}>{t("Back up data and drafts", "备份数据与草稿")}</Button>
    <TextField select label={t("Desktop backup", "桌面备份")} value={backupId} onChange={event => setBackupId(event.target.value)}>
      <MenuItem value="">{t("Choose a backup", "选择备份")}</MenuItem>
      {status.backups.map(item => <MenuItem key={item.backupId} value={item.backupId}>{new Date(item.createdUtcSeconds * 1000).toLocaleString()} · {item.backupId.slice(0, 8)}</MenuItem>)}
    </TextField>
    <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
      <Button disabled={busy || waiting || !backupId} onClick={() => void submit("verify")}>{t("Verify backup", "校验备份")}</Button>
      <Button disabled={busy || waiting || !backupId} onClick={() => void submit("restore")}>{t("Restore an independent copy", "恢复为独立副本")}</Button>
    </Stack>
    <Typography>{t("Clear the calculation cache to rebuild it on next use. Saved data, exports, drafts, backups and browser storage are kept. Local services briefly restart.", "清理计算缓存后，下次使用时会重新生成。已保存数据、导出、草稿、备份和浏览器存储均保留，本地服务会短暂重启。")}</Typography>
    <Button disabled={busy || waiting} onClick={() => void submit("clear-cache")}>{t("Clear calculation cache", "清理计算缓存")}</Button>
    {waiting && <Alert severity="info">{t("Waiting for current work to finish, then saving your workspace before maintenance…", "等待当前任务结束，保存工作区后执行维护……")}</Alert>}
    {status.operation?.state === "completed" && <Alert severity="success">{t("Data operation completed.", "数据操作已完成。")}{status.operation.backupDirectory || status.operation.restoredProfileRoot}</Alert>}
    {status.operation?.state === "failed" && <Alert severity="warning">{status.operation.message}</Alert>}
    {status.profiles && <>
      <Typography data-testid="desktop-active-profile">{t("Current profile", "当前资料")}: {copyLabel(status.profiles.activeCopyId)}</Typography>
      <Typography role="status" data-testid="desktop-next-profile">{t("Next launch", "下次启动")}: {copyLabel(status.profiles.nextCopyId)}</Typography>
      <TextField select label={t("Restored profile", "恢复资料副本")} value={copyId} onChange={event => setCopyId(event.target.value)} disabled={disabled}>
        <MenuItem value="">{t("Choose a restored copy", "选择恢复副本")}</MenuItem>
        {status.profiles.copies.map(item => <MenuItem key={item.copyId} value={item.copyId}>{copyLabel(item.copyId)}</MenuItem>)}
      </TextField>
      <Button disabled={disabled || !copyId} onClick={() => void selectCopy(copyId)}>{t("Use selected copy on next launch", "下次启动使用所选副本")}</Button>
      {status.profiles.unavailableCopies > 0 && <Alert severity="warning">{t("Some restored copies could not be verified and cannot be selected. Their files are retained.", "部分恢复副本未通过校验，暂不可选择，其文件已保留。")}</Alert>}
    </>}
    {status.operation?.copyId && <Button disabled={disabled} onClick={() => void selectCopy(status.operation!.copyId!)}>{t("Use restored copy on next launch", "下次启动使用恢复副本")}</Button>}
    <Button disabled={disabled} onClick={() => void selectCopy(null)}>{t("Use original profile on next launch", "下次启动使用原资料")}</Button>
    {error && <Alert severity="error">{error}</Alert>}{message && <Typography role="status">{message}</Typography>}
  </Stack>;
}
