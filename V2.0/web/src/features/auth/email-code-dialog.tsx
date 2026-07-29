"use client";

import CheckCircleOutlineRoundedIcon from "@mui/icons-material/CheckCircleOutlineRounded";
import EmailOutlinedIcon from "@mui/icons-material/EmailOutlined";
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { LabVizApiError, labvizApi } from "@/lib/api/labviz-api";

type EmailCodeDialogProps = {
  open: boolean;
  onClose: () => void;
  onAuthenticated?: () => void;
};

export function EmailCodeDialog({ open, onAuthenticated, onClose }: EmailCodeDialogProps) {
  const t = useTranslations("auth");
  const queryClient = useQueryClient();
  const [step, setStep] = useState<"email" | "code" | "success">("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [challengeId, setChallengeId] = useState("");
  const [resendSeconds, setResendSeconds] = useState(0);
  const [deliveryMode, setDeliveryMode] = useState<"console" | "email">("email");

  const requestCode = useMutation({
    mutationFn: () => labvizApi.requestEmailCode(email),
    onSuccess: (result) => {
      setChallengeId(result.challengeId);
      setResendSeconds(result.resendAfterSeconds);
      setDeliveryMode(result.deliveryMode);
      setStep("code");
    },
  });

  const verifyCode = useMutation({
    mutationFn: () => labvizApi.verifyEmailCode(challengeId, code),
    onSuccess: () => {
      setStep("success");
      void queryClient.invalidateQueries({ queryKey: ["auth-state"] });
      onAuthenticated?.();
    },
  });

  useEffect(() => {
    if (step !== "code" || resendSeconds <= 0) return;
    const timer = window.setTimeout(
      () => setResendSeconds((seconds) => seconds - 1),
      1_000,
    );
    return () => window.clearTimeout(timer);
  }, [resendSeconds, step]);

  const handleClose = () => {
    setStep("email");
    setCode("");
    setChallengeId("");
    setResendSeconds(0);
    requestCode.reset();
    verifyCode.reset();
    onClose();
  };

  const error = requestCode.error ?? verifyCode.error;
  const errorText =
    error instanceof LabVizApiError
      ? error.message
      : error
        ? t("genericError")
        : null;

  return (
    <Dialog
      fullWidth
      maxWidth="xs"
      onClose={handleClose}
      open={open}
      slotProps={{ paper: { sx: { borderRadius: 2, p: 0.5 } } }}
    >
      <DialogTitle>{step === "success" ? t("successTitle") : t("title")}</DialogTitle>
      <DialogContent>
        {step === "success" ? (
          <Stack spacing={2} sx={{ alignItems: "center", py: 3, textAlign: "center" }}>
            <Box
              sx={{
                alignItems: "center",
                bgcolor: "success.light",
                borderRadius: "50%",
                color: "success.main",
                display: "flex",
                height: 56,
                justifyContent: "center",
                width: 56,
              }}
            >
              <CheckCircleOutlineRoundedIcon />
            </Box>
            <Typography color="text.secondary" variant="body2">
              {t("successBody")}
            </Typography>
          </Stack>
        ) : (
          <Stack spacing={2.25} sx={{ pt: 1 }}>
            <Typography color="text.secondary" variant="body2">
              {step === "email" ? t("reason") : t("codeSent", { email })}
            </Typography>
            {step === "code" && deliveryMode === "console" ? (
              <Alert severity="warning">{t("consoleDelivery")}</Alert>
            ) : null}
            {step === "email" ? (
              <TextField
                autoComplete="email"
                autoFocus
                fullWidth
                label={t("email")}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="name@example.com"
                type="email"
                value={email}
              />
            ) : (
              <TextField
                autoComplete="one-time-code"
                autoFocus
                fullWidth
                label={t("code")}
                onChange={(event) =>
                  setCode(event.target.value.replace(/\D/g, "").slice(0, 6))
                }
                placeholder="000000"
                slotProps={{
                  htmlInput: {
                    inputMode: "numeric",
                    maxLength: 6,
                    style: {
                      fontSize: 24,
                      letterSpacing: "0.32em",
                      textAlign: "center",
                    },
                  },
                }}
                value={code}
              />
            )}
            {errorText ? <Alert severity="error">{errorText}</Alert> : null}
            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <EmailOutlinedIcon color="secondary" fontSize="small" />
              <Typography color="text.secondary" variant="caption">
                {t("privacy")}
              </Typography>
            </Stack>
          </Stack>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5 }}>
        {step === "email" ? (
          <Button
            disabled={!email.includes("@")}
            loading={requestCode.isPending}
            onClick={() => requestCode.mutate()}
            variant="contained"
          >
            {t("send")}
          </Button>
        ) : step === "code" ? (
          <>
            <Button onClick={() => setStep("email")}>{t("changeEmail")}</Button>
            <Button
              disabled={resendSeconds > 0 || requestCode.isPending}
              onClick={() => requestCode.mutate()}
            >
              {resendSeconds > 0
                ? t("resendIn", { seconds: resendSeconds })
                : t("resend")}
            </Button>
            <Button
              disabled={code.length !== 6}
              loading={verifyCode.isPending}
              onClick={() => verifyCode.mutate()}
              variant="contained"
            >
              {t("verify")}
            </Button>
          </>
        ) : (
          <Button onClick={handleClose} variant="contained">
            {t("continue")}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
