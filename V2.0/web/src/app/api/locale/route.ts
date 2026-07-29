import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const body: unknown = await request.json();
  const locale =
    typeof body === "object" &&
    body !== null &&
    "locale" in body &&
    body.locale === "zh"
      ? "zh"
      : "en";

  const response = NextResponse.json({ locale });
  response.cookies.set("labviz-locale", locale, {
    httpOnly: false,
    sameSite: "lax",
    maxAge: 60 * 60 * 24 * 365,
    path: "/",
  });
  return response;
}
