"use client";

import { Alert, Box, Link as MuiLink, Paper, Stack, Typography } from "@mui/material";
import { useLocale, useTranslations } from "next-intl";
import Link from "next/link";
import { useEffect, useState } from "react";

type MarkdownBlock =
  | { kind: "heading"; level: 2 | 3; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "list"; items: string[] }
  | { kind: "image"; alt: string; src: string }
  | { kind: "divider" };

function parseMarkdown(source: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = [];
  let paragraph: string[] = [];
  let list: string[] = [];
  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ kind: "paragraph", text: paragraph.join(" ") });
    paragraph = [];
  };
  const flushList = () => {
    if (list.length) blocks.push({ kind: "list", items: list });
    list = [];
  };

  for (const rawLine of source.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }
    const heading = line.match(/^(#{2,3})\s+(.+)$/);
    const image = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "heading", level: heading[1].length as 2 | 3, text: heading[2] });
    } else if (image) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "image", alt: image[1], src: image[2] });
    } else if (/^---+$/.test(line)) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "divider" });
    } else if (line.startsWith("- ")) {
      flushParagraph();
      list.push(line.slice(2));
    } else {
      flushList();
      paragraph.push(line);
    }
  }
  flushParagraph();
  flushList();
  return blocks;
}

function MarkdownText({ text }: { text: string }) {
  const parts = text.split(/(\[[^\]]+\]\([^)]+\))/g);
  return (
    <>
      {parts.map((part, index) => {
        const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (!link) return part;
        const [, label, href] = link;
        if (href.startsWith("mailto:")) return <MuiLink key={index} href={href}>{label}</MuiLink>;
        if (href.startsWith("/")) return <Link key={index} href={href}>{label}</Link>;
        return label;
      })}
    </>
  );
}

function AboutMarkdown({ source }: { source: string }) {
  return (
    <Stack spacing={2.5}>
      {parseMarkdown(source).map((block, index) => {
        if (block.kind === "heading") {
          return (
            <Typography
              component={`h${block.level}` as "h2" | "h3"}
              key={index}
              sx={{ fontSize: block.level === 2 ? { xs: 23, md: 27 } : 19, fontWeight: 700, mt: 1.5 }}
            >
              {block.text}
            </Typography>
          );
        }
        if (block.kind === "paragraph") {
          return (
            <Typography color="text.secondary" key={index} sx={{ lineHeight: 1.8 }}>
              <MarkdownText text={block.text} />
            </Typography>
          );
        }
        if (block.kind === "list") {
          return (
            <Box component="ul" key={index} sx={{ color: "text.secondary", m: 0, pl: 3 }}>
              {block.items.map((item) => (
                <Box component="li" key={item} sx={{ lineHeight: 1.8, mb: 0.5 }}>
                  <MarkdownText text={item} />
                </Box>
              ))}
            </Box>
          );
        }
        if (block.kind === "image") {
          return (
            <Paper component="figure" key={index} variant="outlined" sx={{ m: 0, overflow: "hidden", p: 1 }}>
              <Box
                alt={block.alt}
                component="img"
                loading="lazy"
                src={block.src}
                sx={{ display: "block", height: "auto", maxHeight: 540, maxWidth: "100%", mx: "auto", objectFit: "contain" }}
              />
              <Typography component="figcaption" color="text.secondary" sx={{ px: 1, py: 0.5 }} variant="caption">
                {block.alt}
              </Typography>
            </Paper>
          );
        }
        return <Box component="hr" key={index} sx={{ border: 0, borderTop: 1, borderColor: "divider", width: "100%" }} />;
      })}
    </Stack>
  );
}

export function AboutScreen() {
  const t = useTranslations("about");
  const locale = useLocale();
  const markdownLocale = locale.startsWith("zh") ? "zh" : "en";
  const [content, setContent] = useState<{
    locale: string;
    source: string | null;
    failed: boolean;
  }>({ locale: "", source: null, failed: false });

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/about/about.${markdownLocale}.md`, {
      signal: controller.signal,
      headers: { Accept: "text/markdown, text/plain" },
    })
      .then((response) => {
        if (!response.ok) throw new Error("About content could not be loaded.");
        return response.text();
      })
      .then((source) => setContent({ locale: markdownLocale, source, failed: false }))
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setContent({ locale: markdownLocale, source: null, failed: true });
        }
      });
    return () => controller.abort();
  }, [markdownLocale]);

  const currentContent = content.locale === markdownLocale ? content : null;

  return (
    <Stack spacing={3} sx={{ maxWidth: 1080, mx: "auto", py: { xs: 2, md: 4 } }}>
      <Typography component="h1" sx={{ fontSize: { xs: 32, md: 40 }, fontWeight: 750 }}>
        {t("title")}
      </Typography>
      {currentContent?.failed ? (
        <Alert severity="error">{t("loadFailed")}</Alert>
      ) : currentContent?.source ? (
        <AboutMarkdown source={currentContent.source} />
      ) : (
        <Typography color="text.secondary" role="status">{t("loading")}</Typography>
      )}
    </Stack>
  );
}
