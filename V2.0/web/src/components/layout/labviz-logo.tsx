"use client";

import { useTheme } from "@mui/material/styles";

export function LabvizLogo({
  height = 34,
  width = 136,
}: {
  height?: number;
  width?: number;
}) {
  const theme = useTheme();
  const wordmarkColor = theme.palette.mode === "dark" ? "#F4F7FB" : "#172033";

  return (
    <svg
      aria-label="LabViz"
      fill="none"
      height={height}
      role="img"
      viewBox="0 0 256 64"
      width={width}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="labviz-flask" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#8DB7FF" />
          <stop offset="1" stopColor="#2563EB" />
        </linearGradient>
      </defs>
      <g strokeLinecap="round" strokeLinejoin="round">
        <path d="M18 8h20M24 8v15L13 48c-2 4 1 8 6 8h18c5 0 8-4 6-8L32 23V8" stroke="#2563EB" strokeWidth="4" />
        <path d="M16 43c6-5 10 4 16-2 5-5 8 1 12-3" stroke="url(#labviz-flask)" strokeWidth="4" />
        <path d="M15 42h27" stroke="#B7D0FF" strokeWidth="2" />
        <path d="M55 45h13l6-12 8 17 9-25 10 20h11" stroke="#0F766E" strokeWidth="3" />
      </g>
      <text
        fill={wordmarkColor}
        fontFamily="Inter, Noto Sans SC, Segoe UI, Arial, sans-serif"
        fontSize="29"
        fontWeight="700"
        letterSpacing="-1"
        x="126"
        y="42"
      >
        LabViz
      </text>
    </svg>
  );
}
