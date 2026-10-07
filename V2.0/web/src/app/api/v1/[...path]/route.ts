import { proxyLocalApiRequest } from "@/lib/api/local-api-proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const forward = (request: Request) => proxyLocalApiRequest(request);

export const GET = forward;
export const HEAD = forward;
export const POST = forward;
export const PUT = forward;
export const PATCH = forward;
export const DELETE = forward;
export const OPTIONS = forward;
