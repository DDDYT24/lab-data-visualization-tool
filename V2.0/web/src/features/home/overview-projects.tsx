"use client";

import AutoGraphOutlinedIcon from "@mui/icons-material/AutoGraphOutlined";
import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
import { Alert, Box, Button, ButtonBase, Paper, Stack, Typography } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useLocale } from "next-intl";
import Link from "next/link";
import { labvizApi } from "@/lib/api/labviz-api";

export function OverviewProjects({ research = false }: { research?: boolean }) {
  const zh = useLocale().startsWith("zh");
  const text = (en: string, cn: string) => zh ? cn : en;
  const auth = useQuery({ queryKey: ["auth-state"], queryFn: () => labvizApi.getAuthState() });
  const projects = useQuery({
    queryKey: ["projects"], queryFn: ({ signal }) => labvizApi.listProjects(signal),
    enabled: auth.data?.authenticated === true, retry: false,
  });
  const recent = [...(projects.data?.projects ?? [])].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).slice(0, 3);
  return <Box component="section" aria-labelledby="overview-recent-title" data-testid="overview-projects" sx={{ mt: 4, width: "100%" }}>
    <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 2, gap: 1 }}>
      <Typography component="h2" id="overview-recent-title" variant="h3">{text("Recent projects", "最近的项目")}</Typography>
      <Button component={Link} href="/history" prefetch={false} endIcon={<ArrowForwardRoundedIcon />}>
        {text("View all projects", "查看全部项目")}
      </Button>
    </Stack>
    {auth.isPending || (auth.data?.authenticated && projects.isPending)
      ? <Typography role="status">{text("Loading projects…", "正在加载项目…")}</Typography>
      : projects.error || auth.error
        ? <Alert severity="warning" action={<Button onClick={() => { if (auth.data?.authenticated) void projects.refetch(); else void auth.refetch(); }}>{text("Retry", "重试")}</Button>}>
          {text("Saved projects are unavailable. You can still import a file.", "暂时无法读取已保存项目，仍可导入新文件。")}
        </Alert>
        : !auth.data?.authenticated
          ? <Paper variant="outlined" sx={{ p: 3 }}><Typography color="text.secondary">{text("Saved projects appear here when this session has access.", "当前会话有访问权限后，这里会显示已保存项目。")}</Typography></Paper>
          : !recent.length
            ? <Paper variant="outlined" sx={{ p: 3 }}><Typography color="text.secondary">{text("Import your first file to create a project.", "导入第一份文件后，这里会显示真实项目。")}</Typography></Paper>
            : <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(3,minmax(0,1fr))" }, gap: 2 }}>
              {recent.map((project) => <Paper key={project.id} component="article" variant="outlined" sx={{ minWidth: 0, borderRadius: "18px", overflow: "hidden" }}>
                <ButtonBase component={Link} prefetch={false}
                  href={`/workspace/${encodeURIComponent(project.id)}${research ? "?workspace=research&researchTask=data" : ""}`}
                  data-testid="overview-project-card" sx={{ width: "100%", display: "block", textAlign: "left", p: 2.5, "&:focus-visible": { outline: "3px solid", outlineColor: "primary.main", outlineOffset: -3 } }}>
                  <Typography component="h3" sx={{ fontSize: 16, fontWeight: 700, overflowWrap: "anywhere" }}>{project.title}</Typography>
                  <Typography color="text.secondary" variant="caption" sx={{ display: "block", mt: 0.5, overflowWrap: "anywhere" }}>{project.sourceName}</Typography>
                  <Box sx={{ height: 150, mt: 1.5, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "background.default", borderRadius: "12px" }}>
                    {project.thumbnailUrl
                      // eslint-disable-next-line @next/next/no-img-element
                      ? <img src={project.thumbnailUrl} alt={text(`Saved figure preview: ${project.title}`, `已保存图形预览：${project.title}`)} loading="lazy" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} />
                      : <Stack sx={{ alignItems: "center" }} spacing={1}><AutoGraphOutlinedIcon sx={{ fontSize: 38, color: "secondary.main" }} /><Typography variant="caption" color="text.secondary">{text("Open project", "打开项目")}</Typography></Stack>}
                  </Box>
                  <Typography component="time" dateTime={project.updatedAt} variant="caption" color="text.secondary" sx={{ display: "block", mt: 1.5 }}>
                    {text("Updated", "更新于")} {new Date(project.updatedAt).toLocaleDateString(zh ? "zh-CN" : "en-CA")}
                  </Typography>
                </ButtonBase>
              </Paper>)}
            </Box>}
  </Box>;
}
