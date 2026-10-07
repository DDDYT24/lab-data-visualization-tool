const HOP_BY_HOP_HEADERS = [
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
];

export function resolveLocalApiBaseUrl(
  value = process.env.LABVIZ_API_PROXY_TARGET ?? "http://127.0.0.1:8000",
): URL {
  const target = new URL(value);
  const port = Number(target.port || "80");
  if (
    target.protocol !== "http:" ||
    target.hostname !== "127.0.0.1" ||
    target.username !== "" ||
    target.password !== "" ||
    target.pathname !== "/" ||
    target.search !== "" ||
    target.hash !== "" ||
    !Number.isInteger(port) ||
    port < 1 ||
    port > 65_535
  ) {
    throw new TypeError("The local API target must be an HTTP loopback URL with a valid port.");
  }
  return target;
}

export async function proxyLocalApiRequest(
  request: Request,
  fetcher: typeof fetch = fetch,
): Promise<Response> {
  let apiBase: URL;
  try {
    apiBase = resolveLocalApiBaseUrl();
  } catch {
    return Response.json(
      { detail: "The local API proxy is not configured safely." },
      { status: 500 },
    );
  }

  const incoming = new URL(request.url);
  const target = new URL(`${incoming.pathname}${incoming.search}`, apiBase);
  const headers = new Headers(request.headers);
  for (const name of HOP_BY_HOP_HEADERS) headers.delete(name);

  const method = request.method.toUpperCase();
  const hasBody = method !== "GET" && method !== "HEAD" && request.body !== null;
  const init = {
    method,
    headers,
    cache: "no-store",
    redirect: "manual",
    ...(hasBody ? { body: request.body, duplex: "half" } : {}),
  } as RequestInit;

  let upstream: Response;
  try {
    upstream = await fetcher(target, init);
  } catch {
    return Response.json({ detail: "The local API is unavailable." }, { status: 502 });
  }

  const responseHeaders = new Headers(upstream.headers);
  for (const name of HOP_BY_HOP_HEADERS) responseHeaders.delete(name);
  const body =
    method === "HEAD" || upstream.status === 204 || upstream.status === 304
      ? null
      : upstream.body;
  return new Response(body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
}
