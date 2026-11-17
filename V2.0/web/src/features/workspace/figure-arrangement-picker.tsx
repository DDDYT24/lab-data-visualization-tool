"use client";

import { Stack, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { ARRANGEMENTS, layoutCells, PANEL_LABELS, sameLayout, type FigureLayout } from "@/domain/figure-draft";

type T = (en: string, cn: string) => string;

/** What a sighted user reads off the drawing: the grid, or which panels span. */
export function arrangementName(layout: FigureLayout, count: number, t: T): string {
  if (layout.kind === "grid") return t(`${layout.rows} × ${layout.cols} grid`, `${layout.rows} × ${layout.cols} 网格`);
  const spans = layoutCells(layout, count).flatMap(([, , rows, cols], index) => {
    const label = PANEL_LABELS[index]!;
    if (cols > 1) return [t(`${label} across ${cols} columns`, `${label} 横跨 ${cols} 列`)];
    if (rows > 1) return [t(`${label} down ${rows} rows`, `${label} 纵跨 ${rows} 行`)];
    return [];
  });
  return spans.join(t(", ", "，"));
}

function Thumbnail({ layout, count }: { layout: FigureLayout; count: number }) {
  const width = 44, height = 32, gap = 2;
  const cellW = (width - gap * (layout.cols - 1)) / layout.cols, cellH = (height - gap * (layout.rows - 1)) / layout.rows;
  return <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
    {layoutCells(layout, count).map(([row, col, rows, cols], index) => <g key={index}>
      <rect x={col * (cellW + gap)} y={row * (cellH + gap)} width={cols * cellW + (cols - 1) * gap} height={rows * cellH + (rows - 1) * gap}
        rx={1.5} fill="currentColor" fillOpacity={0.14} stroke="currentColor" strokeOpacity={0.55} />
      <text x={col * (cellW + gap) + 3} y={row * (cellH + gap) + 9} fontSize={8} fill="currentColor">{PANEL_LABELS[index]}</text>
    </g>)}
  </svg>;
}

export function FigureArrangementPicker({ count, value, disabled, onChange, t }: {
  count: number; value: FigureLayout; disabled: boolean; onChange: (layout: FigureLayout) => void; t: T;
}) {
  const options = ARRANGEMENTS[count] ?? [];
  if (options.length < 2) return null;
  const selected = options.findIndex((layout) => sameLayout(layout, value));
  return <Stack spacing={0.75}>
    <Typography id="figure-arrangement-label" variant="body2">{t("Arrangement", "排列")}</Typography>
    <ToggleButtonGroup exclusive size="small" value={selected} disabled={disabled} aria-labelledby="figure-arrangement-label"
      onChange={(_, index: number | null) => { if (index !== null) onChange(options[index]!); }} sx={{ flexWrap: "wrap" }}>
      {options.map((layout, index) => {
        const name = arrangementName(layout, count, t);
        return <ToggleButton key={index} value={index} aria-label={name} title={name} sx={{ p: 0.75, color: "text.secondary", "&.Mui-selected": { color: "text.primary" } }}>
          <Thumbnail layout={layout} count={count} />
        </ToggleButton>;
      })}
    </ToggleButtonGroup>
  </Stack>;
}
