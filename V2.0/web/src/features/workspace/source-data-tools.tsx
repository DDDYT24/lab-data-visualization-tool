"use client";

import { Alert, Button, Divider, Stack, TextField, Typography } from "@mui/material";
import { useLocale } from "next-intl";
import { useEffect, useState } from "react";
import { z } from "zod";
import { request } from "@/lib/api/labviz-api";

const schema = z.discriminatedUnion("available", [
  z.object({ available: z.literal(false), reason: z.string() }),
  z.object({ available: z.literal(true), database: z.string(), objects: z.string(),
    dataDirectory: z.string(), backupDirectory: z.string(), tool: z.string(),
    mode: z.enum(["source", "installed"]).optional(), profileRoot: z.string().optional() }),
]);
const quote = (value: string) => "'" + value.replaceAll("'", "''") + "'";

export function SourceDataTools() {
  const zh = useLocale().startsWith("zh");
  const text = (en: string, cn: string) => zh ? cn : en;
  const [info, setInfo] = useState<z.infer<typeof schema> | null>(null);
  const [backup, setBackup] = useState("");
  const [selectedCommand, setSelectedCommand] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const installed = info?.available && info.mode === "installed";
  const profileArgument = info?.available && info.profileRoot ? ` -ProfileRoot ${quote(info.profileRoot)}` : "";
  useEffect(() => {
    let active = true;
    request("/local/source-data", schema).then((value) => { if (active) { setInfo(value); setError(""); } })
      .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : String(reason)); });
    return () => { active = false; };
  }, [retry]);
  async function copy(command: string) {
    setSelectedCommand(command);
    try { await navigator.clipboard.writeText(command); setMessage(text("Command copied", "命令已复制")); }
    catch { setMessage(text("Clipboard unavailable. Select and copy the command below.", "无法访问剪贴板，请选中下方命令复制。")); }
  }
  return <Stack spacing={2} data-testid="source-data-tools" sx={{ "& .MuiInputLabel-root:not(.Mui-focused)": { color: "text.primary" } }}>
    <Divider />
    <Typography component="h3" variant="subtitle1">{installed ? text("Installed application data", "安装版应用数据") : text("Source workspace data", "源码工作区数据")}</Typography>
    {error && <Alert severity="error" action={<Button onClick={() => setRetry((n) => n + 1)}>{text("Retry", "重试")}</Button>}>{error}</Alert>}
    {!info && !error && <Typography role="status">{text("Loading data location…", "正在读取数据位置…")}</Typography>}
    {info && !info.available && <Alert severity="info">{text("These tools require the private V2.3 local source launcher.", "这些工具仅适用于 V2.3 本地源码启动器。")}</Alert>}
    {info?.available && <>
      <TextField label={text("SQLite database", "SQLite 数据库")} value={info.database} slotProps={{ input: { readOnly: true } }} />
      <TextField label={text("Object files", "对象文件")} value={info.objects} slotProps={{ input: { readOnly: true } }} />
      <TextField label={text("Backup destination", "备份位置")} value={info.backupDirectory} slotProps={{ input: { readOnly: true } }} />
      <Alert severity="info">{installed ? text("Stop this installed instance using its Stop shortcut before running the command. Backup includes the database and all object files; Restore creates an independent profile and keeps the current data.", "先使用安装版的退出快捷方式停止此实例，再执行命令。备份包含数据库及全部对象文件；恢复创建独立资料副本，保留当前数据。") : text("Stop this source launcher with Ctrl+C, then run the copied command in PowerShell. Backup includes the complete database and all object files. Verification checks hashes and SQLite integrity. Restore creates a separate copy and prints its start command.", "先用 Ctrl+C 停止此源码启动器，再在 PowerShell 执行复制的命令。备份包含完整数据库及全部对象文件，校验检查哈希及 SQLite 完整性。恢复创建独立副本，并输出该副本的启动命令。")}</Alert>
      <Button variant="outlined" onClick={() => void copy(`& ${quote(info.tool)} -Action Backup${profileArgument} -DataDirectory ${quote(info.dataDirectory)}`)}>{text("Copy backup command", "复制备份命令")}</Button>
      <TextField label={text("Existing backup folder", "已有备份文件夹")} value={backup} onChange={(event) => { setBackup(event.target.value); setMessage(""); setSelectedCommand(""); }} helperText={text("Paste the backupDirectory returned by a completed backup.", "填写备份成功后返回的 backupDirectory。")}/>
      <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
        <Button disabled={!backup.trim()} onClick={() => void copy(`& ${quote(info.tool)} -Action Verify${profileArgument} -BackupDirectory ${quote(backup.trim())}`)}>{text("Copy verification command", "复制校验命令")}</Button>
        <Button disabled={!backup.trim()} onClick={() => void copy(`& ${quote(info.tool)} -Action Restore${profileArgument} -BackupDirectory ${quote(backup.trim())}`)}>{text("Copy restore command", "复制恢复命令")}</Button>
      </Stack>
      {selectedCommand && <TextField multiline label={text("PowerShell command", "PowerShell 命令")} value={selectedCommand} slotProps={{ input: { readOnly: true } }} />}
      {message && <Typography role="status">{message}</Typography>}
      <Typography variant="caption">{text("Research output ZIPs contain selected outputs; use these tools for a complete data backup. The active data directory is never replaced by Restore.", "研究输出 ZIP 只包含选定输出，完整数据备份请使用这些工具。恢复不会覆盖当前数据目录。")}</Typography>
    </>}
  </Stack>;
}
