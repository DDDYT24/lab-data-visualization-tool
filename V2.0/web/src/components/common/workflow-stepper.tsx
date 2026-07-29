"use client";

import CheckRoundedIcon from "@mui/icons-material/CheckRounded";
import { Box, ButtonBase, Stack, Typography } from "@mui/material";
import { useTranslations } from "next-intl";

import {
  type WorkflowStep,
  workflowSteps,
} from "@/features/workspace/workspace-store";

type WorkflowStepperProps = {
  activeStep: WorkflowStep;
  onChange: (step: WorkflowStep) => void;
};

export function WorkflowStepper({ activeStep, onChange }: WorkflowStepperProps) {
  const t = useTranslations("workspace.steps");
  const activeIndex = workflowSteps.indexOf(activeStep);

  return (
    <Stack
      component="nav"
      direction={{ xs: "row", md: "column" }}
      spacing={1}
      sx={{
        maxWidth: "100%",
        overflowX: "auto",
        p: { xs: 1.5, md: 2 },
        scrollbarWidth: "none",
        "&::-webkit-scrollbar": { display: "none" },
      }}
    >
      {workflowSteps.map((step, index) => {
        const active = step === activeStep;
        const complete = index < activeIndex;
        return (
          <ButtonBase
            key={step}
            aria-current={active ? "step" : undefined}
            onClick={() => onChange(step)}
            sx={{
              alignItems: "center",
              borderRadius: 2,
              color: active ? "primary.main" : "text.secondary",
              display: "flex",
              flex: { xs: "0 0 auto", md: "0 0 auto" },
              gap: 1.25,
              justifyContent: "flex-start",
              minHeight: 48,
              px: 1.25,
              textAlign: "left",
              width: { xs: "auto", md: "100%" },
              ...(active && { bgcolor: "primary.light" }),
            }}
          >
            <Box
              sx={{
                alignItems: "center",
                bgcolor: active ? "primary.main" : complete ? "success.light" : "background.paper",
                border: 1,
                borderColor: active ? "primary.main" : complete ? "success.main" : "divider",
                borderRadius: "50%",
                color: active ? "primary.contrastText" : complete ? "success.main" : "text.secondary",
                display: "flex",
                flex: "0 0 auto",
                fontSize: 13,
                fontWeight: 700,
                height: 28,
                justifyContent: "center",
                width: 28,
              }}
            >
              {complete ? <CheckRoundedIcon sx={{ fontSize: 17 }} /> : index + 1}
            </Box>
            <Typography sx={{ fontWeight: active ? 700 : 600 }} variant="body2">
              {t(step)}
            </Typography>
          </ButtonBase>
        );
      })}
    </Stack>
  );
}
