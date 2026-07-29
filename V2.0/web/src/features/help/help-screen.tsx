"use client";

import AutoGraphRoundedIcon from "@mui/icons-material/AutoGraphRounded";
import CleaningServicesOutlinedIcon from "@mui/icons-material/CleaningServicesOutlined";
import FileUploadOutlinedIcon from "@mui/icons-material/FileUploadOutlined";
import HelpOutlineRoundedIcon from "@mui/icons-material/HelpOutlineRounded";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  InputAdornment,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import ExpandMoreRoundedIcon from "@mui/icons-material/ExpandMoreRounded";
import { useLocale, useTranslations } from "next-intl";
import { useMemo, useState } from "react";

import { ApiStatePanel } from "@/components/common/api-state-panel";

const guidesEn = [
  {
    title: "Import an experiment file",
    body: "Choose XLSX, CSV, TSV, TXT, or JSON. Confirm the detected sheet, header row, numeric columns, and units before continuing.",
    icon: FileUploadOutlinedIcon,
  },
  {
    title: "Review suspicious values",
    body: "LabViz explains why a value was flagged. You decide whether to keep it, exclude it from the chart, or remove it only from the cleaned copy.",
    icon: CleaningServicesOutlinedIcon,
  },
  {
    title: "Create a publication figure",
    body: "Start with a recommended chart, then adjust labels, units, colors, fitting, uncertainty, dimensions, DPI, and output format.",
    icon: AutoGraphRoundedIcon,
  },
];

const guidesZh = [
  {
    title: "导入实验文件",
    body: "选择 XLSX、CSV、TSV、TXT 或 JSON。在继续前确认识别出的工作表、表头行、数值列和单位。",
    icon: FileUploadOutlinedIcon,
  },
  {
    title: "检查可疑数值",
    body: "LabViz 会解释标记原因。你可以保留数值、仅从图表排除，或只从清洗副本中删除。",
    icon: CleaningServicesOutlinedIcon,
  },
  {
    title: "创建论文图表",
    body: "从推荐图表开始，再调整标签、单位、颜色、拟合、不确定性、尺寸、DPI 和输出格式。",
    icon: AutoGraphRoundedIcon,
  },
];

const questionsEn = [
  {
    title: "What is a confidence interval?",
    body: "It is a range calculated from the sample and selected confidence level. A 99% interval is normally wider than a 95% interval; it is not automatically a better analysis.",
  },
  {
    title: "Does LabViz change the original file?",
    body: "No. Cleaning decisions apply to the project and cleaned copy. The uploaded source remains unchanged and is deleted after cloud parsing.",
  },
  {
    title: "Why is my chart preview sampled?",
    body: "For large datasets, LabViz uses evenly distributed points to keep interaction responsive. Cleaning and final data export still operate on the complete dataset.",
  },
  {
    title: "When should I use polynomial fitting?",
    body: "Use it only when the experiment and residual pattern support it. LabViz defaults to a maximum order of three to reduce visually convincing but unreliable overfitting.",
  },
];

const questionsZh = [
  {
    title: "什么是置信区间？",
    body: "它是根据样本和所选置信水平计算出的范围。99% 区间通常比 95% 更宽，但并不自动代表分析更好。",
  },
  {
    title: "LabViz 会修改原始文件吗？",
    body: "不会。清洗决定只作用于项目和清洗副本。上传的源文件保持不变，并会在云端解析后删除。",
  },
  {
    title: "为什么图表预览经过抽样？",
    body: "面对大型数据集，LabViz 会均匀选取数据点来保持交互流畅；清洗和最终数据导出仍处理完整数据集。",
  },
  {
    title: "什么时候应该使用多项式拟合？",
    body: "只有实验原理和残差模式支持时才应使用。LabViz 默认最高三阶，避免得到看似贴合、实际不可靠的曲线。",
  },
];

export function HelpScreen() {
  const t = useTranslations("help");
  const locale = useLocale();
  const [search, setSearch] = useState("");
  const guides = locale === "zh" ? guidesZh : guidesEn;
  const questions = locale === "zh" ? questionsZh : questionsEn;
  const matches = useMemo(() => {
    const normalized = search.trim().toLowerCase();
    if (!normalized) return questions;
    return questions.filter((question) =>
      `${question.title} ${question.body}`.toLowerCase().includes(normalized),
    );
  }, [questions, search]);

  return (
    <Stack spacing={3} sx={{ maxWidth: 1060 }}>
      <Box>
        <Typography component="h1" sx={{ fontSize: 28, fontWeight: 750 }}>
          {t("title")}
        </Typography>
        <Typography color="text.secondary" sx={{ mt: 0.5 }} variant="body2">
          {t("description")}
        </Typography>
      </Box>

      <TextField
        onChange={(event) => setSearch(event.target.value)}
        placeholder={t("search")}
        size="small"
        slotProps={{
          input: {
            startAdornment: (
              <InputAdornment position="start">
                <SearchRoundedIcon fontSize="small" />
              </InputAdornment>
            ),
          },
        }}
        sx={{ maxWidth: 520 }}
        value={search}
      />

      {!search ? (
        <Box
          sx={{
            display: "grid",
            gap: 2,
            gridTemplateColumns: {
              xs: "1fr",
              md: "repeat(3, minmax(0, 1fr))",
            },
          }}
        >
          {guides.map((guide) => {
            const Icon = guide.icon;
            return (
              <Paper
                key={guide.title}
                sx={{ border: 1, borderColor: "divider", p: 3 }}
              >
                <Stack spacing={2}>
                  <Box
                    sx={{
                      alignItems: "center",
                      bgcolor: "primary.light",
                      borderRadius: 2,
                      color: "primary.main",
                      display: "flex",
                      height: 42,
                      justifyContent: "center",
                      width: 42,
                    }}
                  >
                    <Icon />
                  </Box>
                  <Typography component="h2" variant="h3">
                    {guide.title}
                  </Typography>
                  <Typography color="text.secondary" variant="body2">
                    {guide.body}
                  </Typography>
                </Stack>
              </Paper>
            );
          })}
        </Box>
      ) : null}

      <Paper sx={{ border: 1, borderColor: "divider", overflow: "hidden" }}>
        <Stack
          direction="row"
          spacing={1}
          sx={{ alignItems: "center", px: 3, py: 2.5 }}
        >
          <HelpOutlineRoundedIcon color="secondary" />
          <Typography component="h2" sx={{ fontWeight: 750 }}>
            {t("commonQuestions")}
          </Typography>
        </Stack>
        {matches.length === 0 ? (
          <ApiStatePanel
            compact
            description={t("noMatchDescription")}
            kind="empty"
            title={t("noMatchTitle")}
          />
        ) : (
          matches.map((question) => (
            <Accordion disableGutters elevation={0} key={question.title}>
              <AccordionSummary expandIcon={<ExpandMoreRoundedIcon />}>
                <Typography sx={{ fontWeight: 650 }}>
                  {question.title}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography color="text.secondary" variant="body2">
                  {question.body}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))
        )}
      </Paper>
    </Stack>
  );
}
