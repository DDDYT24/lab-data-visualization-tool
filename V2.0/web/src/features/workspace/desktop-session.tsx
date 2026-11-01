"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Alert, Box, Button, Stack, Typography } from "@mui/material";
import { useWorkspaceStore } from "./workspace-store";
import { useLocale } from "next-intl";

type Base = Partial<Record<"project" | "research" | "figure", string | null>>;
type Entry = { version: 1; projectId: string | null; base: Base; value: unknown };
type Snapshot = { format: "labviz-desktop-checkpoint-v1"; profileId: string; route: string; entries: Record<string, Entry>; checkpointId: string };
type Transport = { url: string; token: string; session?: { initial: Snapshot | null; warning: string | null } };
declare global {
  interface Window {
    labvizDesktop?: { profileId: string; apiBase: string; automaticSuspension: boolean };
    labvizDesktopTransport?: Transport;
    labvizDesktopCheckpoint?: (identifier: string) => Promise<void>;
    labvizDesktopPending?: () => number;
    labvizDesktopFlushPreferences?: () => Promise<void>;
  }
}

const records = new Map<string, { entry: Entry; ready: boolean; busy: boolean }>();
const pending = new Map<string, Entry>();
const conditions = new Map<string, { ready: boolean; busy: boolean }>();
const issues = new Map<string, string>();
let restoringUntil = 0;
let rootReady = false;
let capturing = false;
let activeProfile: string | null = null;
const notify = () => window.dispatchEvent(new Event("labviz:desktop-session"));
const enabled = () => typeof window !== "undefined" && !!window.labvizDesktopTransport?.session;
export function desktopDraftBase(key: string) { return pending.get(key)?.base; }
export function desktopDraftValue(key: string) { return pending.get(key)?.value; }

/** Called only after the user-confirmed server deletion succeeds. */
export function forgetDesktopProject(projectId: string) {
  for (const key of new Set([...records.keys(), ...pending.keys()])) {
    if (records.get(key)?.entry.projectId === projectId || pending.get(key)?.projectId === projectId) {
      records.delete(key); pending.delete(key); conditions.delete(key); issues.delete(key);
    }
  }
  notify();
}

async function transport(path: string, body?: BodyInit, binary = false) {
  const bridge = window.labvizDesktopTransport!;
  const response = await fetch(new URL(path, bridge.url), {
    method: body === undefined ? "GET" : "POST", body,
    headers: { "X-LabViz-Desktop": bridge.token, ...(binary ? { "Content-Type": "application/octet-stream" } : { "Content-Type": "application/json" }) },
    signal: AbortSignal.timeout(120000),
  });
  if (!response.ok) {
    const value = await response.json().catch(() => ({}));
    throw new Error(value.message || "Desktop draft storage is unavailable. Keep this workspace open.");
  }
  return response;
}

async function encode(value: unknown, files: Map<File, unknown>): Promise<unknown> {
  if (value instanceof File) {
    if (files.has(value)) return files.get(value);
    const stored = await (await transport("/session/files", value, true)).json();
    const reference = { __labvizFile: stored.id, size: value.size, name: value.name, type: value.type, lastModified: value.lastModified };
    files.set(value, reference);
    return reference;
  }
  if (value === undefined) return null;
  if (Array.isArray(value)) return Promise.all(value.map(item => encode(item, files)));
  if (value && typeof value === "object") {
    if (Object.getPrototypeOf(value) !== Object.prototype) throw new Error("An unsaved object cannot be checkpointed. Keep the workspace open.");
    return Object.fromEntries(await Promise.all(Object.entries(value).map(async ([key, item]) => [key, await encode(item, files)])));
  }
  if (["function", "symbol", "bigint"].includes(typeof value)) throw new Error("Unsupported desktop draft value");
  return value;
}

async function decode(value: unknown): Promise<unknown> {
  if (Array.isArray(value)) return Promise.all(value.map(decode));
  if (value && typeof value === "object") {
    const node = value as Record<string, unknown>;
    if (typeof node.__labvizFile === "string") {
      const data = await (await transport("/session/files/" + node.__labvizFile)).arrayBuffer();
      const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", data))].map(n => n.toString(16).padStart(2, "0")).join("");
      if (hash !== node.__labvizFile || data.byteLength !== node.size) throw new Error("A staged source file differs from its checkpoint. The original was retained.");
      return new File([data], String(node.name), { type: String(node.type), lastModified: Number(node.lastModified) });
    }
    return Object.fromEntries(await Promise.all(Object.entries(node).map(async ([key, item]) => [key, await decode(item)])));
  }
  return value;
}

/** Register editable state only. Reload result caches and never restore preview tokens. */
export function useDesktopDraft<T>(key: string, projectId: string | null, base: Base, value: T, restore: (draft: T) => void, ready = true, busy = false) {
  const restored = useRef<string | null>(null);
  useEffect(() => () => {
    conditions.delete(key);
    if (key.endsWith(":new")) { records.delete(key); pending.delete(key); }
    else {
      // A route change unmounts the editor, not its unsaved workspace. Keep
      // an existing conflict pending until the user resolves it explicitly.
      const current = records.get(key);
      if (current && !pending.has(key)) pending.set(key, current.entry);
      records.delete(key);
    }
  }, [key]);
  useEffect(() => {
    if (!enabled()) return;
    conditions.set(key, { ready, busy });
    const saved = pending.get(key);
    if (saved && ready && restored.current !== key) {
      restored.current = key;
      if (saved.projectId !== projectId || Object.keys(saved.base).length !== Object.keys(base).length || Object.entries(saved.base).some(([scope, revision]) => base[scope as keyof Base] !== revision)) {
        records.set(key, { entry: { version: 1, projectId, base, value }, ready, busy });
        issues.set(key, "A saved revision changed or the project is unavailable. The previous draft is retained; it has not been applied.");
        notify();
        return;
      }
      pending.delete(key);
      issues.delete(key);
      records.set(key, { entry: saved, ready, busy });
      restoringUntil = Date.now() + 250;
      restore(saved.value as T);
      notify();
      return;
    }
    if (saved) return;
    if (!ready) return;
    records.set(key, { entry: { version: 1, projectId, base, value }, ready, busy });
  });
}

function clientActivity() {
  return { ready: rootReady && Date.now() > restoringUntil && issues.size === 0,
    busy: capturing || [...conditions.values()].some(record => record.busy || !record.ready),
    pendingRequests: window.labvizDesktopPending?.() ?? 0 };
}

async function checkpoint(identifier: string) {
  if (capturing) return;
  capturing = true;
  try {
    if (!rootReady || issues.size || Date.now() < restoringUntil || [...conditions.values()].some(record => record.busy || !record.ready) || (window.labvizDesktopPending?.() ?? 0)) throw new Error("The editor is still busy. Its state and services were retained.");
    await window.labvizDesktopFlushPreferences?.();
    const entries = Object.fromEntries([...pending, ...[...records].map(([key, record]) => [key, record.entry] as const)]);
    const route = window.location.pathname + window.location.search;
    const value: Snapshot = { format: "labviz-desktop-checkpoint-v1", profileId: window.labvizDesktop!.profileId, route, entries: await encode(entries, new Map()) as Record<string, Entry>, checkpointId: identifier };
    const result = await (await transport("/session/checkpoint", JSON.stringify(value))).json();
    if (result.checkpointId !== identifier || result.saved !== true) throw new Error("Draft storage did not acknowledge this checkpoint.");
  } catch (error) {
    await transport("/session/checkpoint-error", JSON.stringify({ checkpointId: identifier, message: String(error) })).catch(() => {});
  } finally {
    capturing = false;
  }
}

export function DesktopSession({ children, localRuntime = false }: { children: ReactNode; localRuntime?: boolean }) {
  const zh = useLocale().startsWith('zh');
  const [ready, setReady] = useState(!localRuntime);
  const [messages, setMessages] = useState<[string, string][]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => {
    let disposed = false;
    const sync = () => setMessages([...issues]);
    window.addEventListener("labviz:desktop-session", sync);
    async function initialize() {
      if (!enabled()) { setReady(true); return; }
      const session = window.labvizDesktopTransport!.session!;
      try {
        if (activeProfile !== window.labvizDesktop!.profileId) {
          records.clear(); pending.clear(); conditions.clear(); issues.clear();
          restoringUntil = 0;
          activeProfile = window.labvizDesktop!.profileId;
        }
        if (session.warning) issues.set("checkpoint", session.warning);
        if (session.initial) {
          if (session.initial.profileId !== window.labvizDesktop!.profileId || session.initial.format !== "labviz-desktop-checkpoint-v1") throw new Error("Desktop checkpoint identity differs");
          const entries = await decode(session.initial.entries) as Record<string, Entry>;
          if (disposed) return;
          for (const [key, entry] of Object.entries(entries)) pending.set(key, entry);
          // An unsubmitted import needs its File objects before the upload hook mounts.
          const fresh = entries["workspace:new"];
          if (fresh && fresh.projectId === null && window.location.pathname === "/workspace/new") {
            const draft = fresh.value as Partial<ReturnType<typeof useWorkspaceStore.getState>>;
            useWorkspaceStore.setState(draft);
          }
        }
      } catch (error) { issues.set("checkpoint", String(error)); }
      if (!disposed) { rootReady = true; setReady(true); sync(); }
    }
    void initialize();
    window.labvizDesktopCheckpoint = checkpoint;
    const timer = window.setInterval(() => {
      if (!enabled() || !rootReady) return;
      void transport("/session/activity", JSON.stringify(clientActivity())).then(r => r.json()).then(value => { if (!disposed) setNotice(value.error ?? null); }).catch(() => {});
    }, 750);
    return () => { disposed = true; rootReady = false; clearInterval(timer); window.removeEventListener("labviz:desktop-session", sync); delete window.labvizDesktopCheckpoint; };
  }, []);
  async function continueSavedState(key: string) {
    if (key === "checkpoint") pending.clear();
    else pending.delete(key);
    issues.delete(key);
    await transport("/session/acknowledge-warning", "{}");
    setMessages([...issues]);
  }
  return <>
    {messages.length > 0 && <Stack spacing={1} sx={{ p: 2 }} data-testid="desktop-recovery-warning">
      {messages.map(([key, message]) => <Alert key={key} severity="warning" action={<Button onClick={() => void continueSavedState(key)}>{zh ? '使用已保存内容' : 'Use current saved state'}</Button>}>{zh ? key === 'checkpoint' ? '上次草稿暂时无法恢复，原文件已保留。请确认后再使用已保存内容。' : '项目版本已经变化，草稿已保留，尚未自动应用。' : message}</Alert>)}
    </Stack>}
    {notice && <Alert severity="warning" data-testid="desktop-suspend-warning">{zh ? '当前尚不能安全暂停，窗口和后台服务会继续保留。请检查后再试。' : notice}</Alert>}
    {ready ? children : <Box sx={{ p: 4 }}><Typography>Restoring local workspace… / 正在恢复本地工作区…</Typography></Box>}
  </>;
}
