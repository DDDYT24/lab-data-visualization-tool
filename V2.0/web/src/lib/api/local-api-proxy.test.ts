import { describe, expect, it, vi } from "vitest";

import { proxyLocalApiRequest, resolveLocalApiBaseUrl } from "./local-api-proxy";

describe("runtime local API proxy", () => {
  it("accepts only an HTTP 127.0.0.1 target with a valid port", () => {
    expect(resolveLocalApiBaseUrl("http://127.0.0.1:8420").origin).toBe(
      "http://127.0.0.1:8420",
    );
    expect(() => resolveLocalApiBaseUrl("https://127.0.0.1:8420")).toThrow();
    expect(() => resolveLocalApiBaseUrl("http://localhost:8420")).toThrow();
    expect(() => resolveLocalApiBaseUrl("http://127.0.0.1:8420/private")).toThrow();
    expect(() => resolveLocalApiBaseUrl("http://user@127.0.0.1:8420")).toThrow();
  });

  it("forwards method, query, credentials and request body to the runtime port", async () => {
    const fetcher = vi.fn<typeof fetch>(async (_input, init) => {
      const body = await new Response(init?.body as BodyInit).text();
      expect(body).toBe('{"sample":true}');
      expect((init as (RequestInit & { duplex?: string }) | undefined)?.duplex).toBe(
        "half",
      );
      return new Response("accepted", { status: 201 });
    });
    const request = new Request("http://127.0.0.1:3371/api/v1/projects?limit=2", {
      method: "POST",
      headers: {
        authorization: "Bearer local-session",
        "content-type": "application/json",
      },
      body: '{"sample":true}',
    });

    const response = await proxyLocalApiRequest(request, fetcher);
    const [target, init] = fetcher.mock.calls[0] ?? [];

    expect(target).toBeInstanceOf(URL);
    expect((target as URL).href).toBe("http://127.0.0.1:8000/api/v1/projects?limit=2");
    expect(new Headers(init?.headers).get("authorization")).toBe("Bearer local-session");
    expect(response.status).toBe(201);
    expect(await response.text()).toBe("accepted");
  });

  it("reads the target on each request and rejects unsafe runtime overrides", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response("ok"));
    const request = new Request("http://127.0.0.1:3371/api/v1/health");
    const previous = process.env.LABVIZ_API_PROXY_TARGET;
    try {
      process.env.LABVIZ_API_PROXY_TARGET = "http://127.0.0.1:8420";
      await proxyLocalApiRequest(request, fetcher);
      process.env.LABVIZ_API_PROXY_TARGET = "http://127.0.0.1:8421";
      await proxyLocalApiRequest(request, fetcher);
      expect(fetcher.mock.calls.map(([url]) => (url as URL).port)).toEqual(["8420", "8421"]);

      process.env.LABVIZ_API_PROXY_TARGET = "http://example.com:8422";
      const response = await proxyLocalApiRequest(request, fetcher);
      expect(response.status).toBe(500);
      expect(await response.json()).toEqual({
        detail: "The local API proxy is not configured safely.",
      });
      expect(fetcher).toHaveBeenCalledTimes(2);
    } finally {
      if (previous === undefined) delete process.env.LABVIZ_API_PROXY_TARGET;
      else process.env.LABVIZ_API_PROXY_TARGET = previous;
    }
  });

  it("returns a safe gateway error when the local API cannot be reached", async () => {
    const fetcher = vi.fn<typeof fetch>().mockRejectedValue(new Error("connection refused"));
    const response = await proxyLocalApiRequest(
      new Request("http://127.0.0.1:3371/api/v1/health"),
      fetcher,
    );

    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ detail: "The local API is unavailable." });
  });
});
