"use client";

import { useQuery } from "@tanstack/react-query";
import { useLocale } from "next-intl";
import { Alert, Button, Stack, Typography } from "@mui/material";
import { labvizApi } from "@/lib/api/labviz-api";

export function ResearchProjectList({ projectId }: { projectId: string }) {
  const zh = useLocale().startsWith("zh");
  const t = (en: string, cn: string) => zh ? cn : en;
  const projects = useQuery({ queryKey: ["projects"], queryFn: ({ signal }) => labvizApi.listProjects(signal) });
  return <Stack spacing={1} sx={{ minWidth: 0 }}>
    <Typography component="h2" variant="subtitle1">{t("Projects", "项目")}</Typography>
    <Typography variant="caption">{t("Open another project in a new tab to retain this draft.", "其他项目在新标签页打开，保留当前草稿。")}</Typography>
    {projects.isPending && <Typography role="status">{t("Loading projects…", "正在加载项目…")}</Typography>}
    {projects.error && <Alert severity="error">{t("Project list unavailable.", "项目列表暂不可用。")}</Alert>}
    {projects.data?.projects.slice(0, 12).map((project) => <Button key={project.id} component="a"
      href={`/workspace/${project.id}`} target="_blank" rel="noopener"
      aria-current={project.id === projectId ? "page" : undefined}
      sx={{ justifyContent: "flex-start", textAlign: "left", overflowWrap: "anywhere", minHeight: 44 }}>
      {project.title}{project.id === projectId ? t(" · Current", " · 当前") : ""}
    </Button>)}
    <Button component="a" href="/history" target="_blank" rel="noopener">{t("All project history", "全部项目历史")}</Button>
  </Stack>;
}
