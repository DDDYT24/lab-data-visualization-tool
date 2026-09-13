import { APP_VERSION } from "@/lib/app-version";

export type FeedbackCategory = "bug" | "usability" | "scientific" | "feature";

export type FeedbackDiagnosticInput = {
  category: FeedbackCategory;
  description: string;
  pathname?: string;
  userAgent?: string;
};

function platformClass(userAgent: string): string {
  if (/Windows/i.test(userAgent)) return "Windows";
  if (/Android/i.test(userAgent)) return "Android";
  if (/iPhone|iPad|iPod/i.test(userAgent)) return "iOS";
  if (/Mac OS X|Macintosh/i.test(userAgent)) return "macOS";
  if (/Linux/i.test(userAgent)) return "Linux";
  return "Other";
}

function browserClass(userAgent: string): string {
  if (/Edg\//i.test(userAgent)) return "Edge";
  if (/Firefox\//i.test(userAgent)) return "Firefox";
  if (/Chrome\//i.test(userAgent)) return "Chrome";
  if (/Safari\//i.test(userAgent)) return "Safari";
  return "Other";
}

function screenClass(pathname: string): string {
  if (pathname.startsWith("/workspace/")) return "workspace";
  if (pathname.startsWith("/share/")) return "shared-chart";
  if (pathname === "/") return "home";
  return pathname.replace(/^\//, "").split("/")[0] || "home";
}

export function buildFeedbackDiagnostic({
  category,
  description,
  pathname = "",
  userAgent = "",
}: FeedbackDiagnosticInput) {
  return {
    appVersion: APP_VERSION,
    category,
    platform: platformClass(userAgent),
    browser: browserClass(userAgent),
    screen: screenClass(pathname),
    description: description.trim(),
  };
}

export function serializeFeedbackDiagnostic(input: FeedbackDiagnosticInput): string {
  return JSON.stringify(buildFeedbackDiagnostic(input), null, 2);
}
