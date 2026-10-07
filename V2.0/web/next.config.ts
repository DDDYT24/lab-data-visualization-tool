import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  distDir: process.env.LABVIZ_E2E_BUILD === "1" ? ".next-e2e" : ".next",
  output: process.env.LABVIZ_E2E_BUILD === "1" ? undefined : "standalone",
  outputFileTracingRoot: path.resolve(__dirname, ".."),
  turbopack: {
    root: path.resolve(__dirname, ".."),
  },
};

export default nextConfig;
