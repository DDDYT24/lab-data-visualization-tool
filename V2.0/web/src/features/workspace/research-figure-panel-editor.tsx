"use client";

import { useState } from "react";
import { Box, Button, IconButton, Menu, MenuItem, Paper, Stack, TextField, Typography } from "@mui/material";
import DeleteOutlineRoundedIcon from "@mui/icons-material/DeleteOutlineRounded";
import { autoLimits, noteDefaults, type AxisDraft, type PanelDraft } from "@/domain/figure-draft";
import type { FigureAnnotation } from "@/domain/figure-spec";
import type { RenderPlan } from "@/domain/render-plan";

type T = (en: string, cn: string) => string;
type PlanAxes = RenderPlan["panels"][number]["axes"];
type PlanAxis = PlanAxes["x"];

const shown = (value: number) => String(Number(value.toPrecision(6)));

/** A number typed freely: the draft changes only when the text parses, and empty means none. */
function NumberField({ label, name, value, onChange, placeholder, required = false, disabled, t }: {
  t: T; label: string; name?: string; value: number | null; onChange: (value: number | null) => void; placeholder?: string; required?: boolean; disabled?: boolean;
}) {
  const [text, setText] = useState(value === null ? "" : String(value));
  const [last, setLast] = useState(value);
  if (value !== last) { setLast(value); setText(value === null ? "" : String(value)); }
  const parsed = text.trim() === "" ? null : Number(text);
  const invalid = (parsed === null && required) || (parsed !== null && !Number.isFinite(parsed));
  // `name` is the accessible name when the visible label is short ("From" under "X axis").
  return <TextField size="small" label={label} value={text} placeholder={placeholder} disabled={disabled} error={invalid}
    helperText={invalid ? (parsed === null ? t("Required", "必填") : t("Enter a number", "请输入数字")) : undefined}
    slotProps={{ htmlInput: { inputMode: "decimal", ...(name ? { "aria-label": name } : {}) } }} sx={{ minWidth: 0 }}
    onChange={(event) => {
      setText(event.target.value);
      const next = event.target.value.trim() === "" ? null : Number(event.target.value);
      if (next === null ? !required : Number.isFinite(next)) { setLast(next); onChange(next); }
    }} />;
}

/** A category axis places by group: its fields pick a group instead of taking a number. */
function PositionField({ label, value, onChange, axis, disabled, t }: {
  t: T; label: string; value: number; onChange: (value: number) => void; axis: PlanAxis | undefined; disabled?: boolean;
}) {
  if (axis?.scale !== "category") return <NumberField t={t} label={label} value={value} required disabled={disabled} onChange={(next) => onChange(next!)} />;
  return <TextField select size="small" label={label} value={value} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))}>
    {axis.ticks.map((tick) => <MenuItem key={tick.value} value={tick.value}>{tick.label.split("\n")[0]}</MenuItem>)}
  </TextField>;
}

function AxisControls({ name, axis, control, shared, disabled, onChange, t }: {
  name: string; axis: PlanAxis | undefined; control: AxisDraft; shared: boolean; disabled: boolean; onChange: (control: AxisDraft) => void; t: T;
}) {
  if (axis?.scale === "category" || shared) {
    return <Stack direction="row" spacing={1} sx={{ alignItems: "baseline" }}>
      <Typography variant="subtitle2">{name}</Typography>
      <Typography variant="body2" color="text.secondary">{shared ? t("Shared across panels", "各面板共享") : t("Categories", "分类")}</Typography>
    </Stack>;
  }
  const auto = autoLimits(axis, control.scale);
  const [low, high] = control.domain ?? [null, null];
  const limits = (next: [number | null, number | null]) => onChange({
    ...control, domain: next[0] === null && next[1] === null ? null : next as [number, number],
  });
  const half = low === null || high === null ? (low === null && high === null ? null : t("Give both limits", "请填写上下限")) : low >= high ? t("Lower limit must be below upper", "下限须小于上限") : null;
  return <Stack spacing={1}>
    <Typography variant="subtitle2">{name}</Typography>
    <Box sx={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 1, "& > :first-of-type": { gridColumn: "1 / -1" } }}>
      <TextField select size="small" label={t(`${name} scale`, `${name}刻度`)} value={control.scale} disabled={disabled}
        onChange={(event) => onChange({ ...control, scale: event.target.value as AxisDraft["scale"], ticks: null })}>
        <MenuItem value="linear">{t("Linear", "线性")}</MenuItem>
        <MenuItem value="log10">{t("Log 10", "对数 10")}</MenuItem>
        <MenuItem value="symlog">{t("Symmetric log", "对称对数")}</MenuItem>
      </TextField>
      <NumberField t={t} label={t("From", "下限")} name={t(`${name} from`, `${name}下限`)} value={low} disabled={disabled} placeholder={auto ? shown(auto[0]) : t("Auto", "自动")} onChange={(value) => limits([value, high])} />
      <NumberField t={t} label={t("To", "上限")} name={t(`${name} to`, `${name}上限`)} value={high} disabled={disabled} placeholder={auto ? shown(auto[1]) : t("Auto", "自动")} onChange={(value) => limits([low, value])} />
    </Box>
    {half && <Typography role="alert" variant="caption" color="error">{half}</Typography>}
  </Stack>;
}

const KINDS: FigureAnnotation["kind"][] = ["text", "hline", "vline", "rect", "bracket"];

function NoteFields({ note, axes, disabled, onChange, t }: {
  note: FigureAnnotation; axes: PlanAxes | null; disabled: boolean; onChange: (note: FigureAnnotation) => void; t: T;
}) {
  const label = <TextField size="small" label={note.kind === "text" ? t("Text", "文字") : t("Label", "标签")} disabled={disabled}
    value={note.kind === "text" ? note.text : note.label} error={note.kind === "text" && !note.text.trim()}
    slotProps={{ htmlInput: { maxLength: 160 } }}
    onChange={(event) => onChange(note.kind === "text" ? { ...note, text: event.target.value } : { ...note, label: event.target.value })} />;
  const pair = (name: string, values: [number, number], axis: PlanAxis | undefined, set: (values: [number, number]) => void) => <>
    <PositionField t={t} label={t(`${name} from`, `${name} 起`)} value={values[0]} axis={axis} disabled={disabled} onChange={(value) => set([value, values[1]])} />
    <PositionField t={t} label={t(`${name} to`, `${name} 止`)} value={values[1]} axis={axis} disabled={disabled} onChange={(value) => set([values[0], value])} />
  </>;
  const fields = (() => {
    switch (note.kind) {
      case "text": return <>
        <PositionField t={t} label="X" value={note.x} axis={axes?.x} disabled={disabled} onChange={(x) => onChange({ ...note, x })} />
        <PositionField t={t} label="Y" value={note.y} axis={axes?.y} disabled={disabled} onChange={(y) => onChange({ ...note, y })} />
      </>;
      case "hline": case "vline": return <PositionField t={t} label={note.kind === "hline" ? "Y" : "X"} value={note.value}
        axis={note.kind === "hline" ? axes?.y : axes?.x} disabled={disabled} onChange={(value) => onChange({ ...note, value })} />;
      case "rect": return <>
        {pair("X", note.x, axes?.x, (x) => onChange({ ...note, x }))}
        {pair("Y", note.y, axes?.y, (y) => onChange({ ...note, y }))}
      </>;
      case "bracket": return <>
        {pair("X", note.x, axes?.x, (x) => onChange({ ...note, x }))}
        <PositionField t={t} label={t("Height", "高度")} value={note.y} axis={axes?.y} disabled={disabled} onChange={(y) => onChange({ ...note, y })} />
      </>;
    }
  })();
  // One column: the inspector is 240–340px wide, and group names must not be cut.
  return <Stack spacing={1}>{label}{fields}</Stack>;
}

/** The selected panel's own axes and annotations. */
export function ResearchFigurePanelEditor({ panel, label, axes, sharedX, sharedY, disabled, onChange, t }: {
  panel: PanelDraft; label: string; axes: PlanAxes | null; sharedX: boolean; sharedY: boolean; disabled: boolean;
  onChange: (panel: PanelDraft) => void; t: T;
}) {
  const [menu, setMenu] = useState<HTMLElement | null>(null);
  const kinds: Record<FigureAnnotation["kind"], string> = {
    text: t("Text", "文字"), hline: t("Horizontal line", "水平线"), vline: t("Vertical line", "竖直线"),
    rect: t("Shaded range", "阴影区间"), bracket: t("Significance bracket", "显著性括线"),
  };
  const setNote = (index: number, note: FigureAnnotation | null) => onChange({
    ...panel, notes: note ? panel.notes.map((item, i) => (i === index ? note : item)) : panel.notes.filter((_, i) => i !== index),
  });
  return <Stack spacing={2} data-testid="figure-panel-editor">
    <AxisControls name={t("X axis", "X 轴")} axis={axes?.x} control={panel.axes.x} shared={sharedX} disabled={disabled} t={t}
      onChange={(x) => onChange({ ...panel, axes: { ...panel.axes, x } })} />
    <AxisControls name={t("Y axis", "Y 轴")} axis={axes?.y} control={panel.axes.y} shared={sharedY} disabled={disabled} t={t}
      onChange={(y) => onChange({ ...panel, axes: { ...panel.axes, y } })} />
    <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between" }}>
      <Typography variant="subtitle2">{t("Annotations", "标注")}</Typography>
      <Button size="small" disabled={disabled || panel.notes.length >= 20} aria-haspopup="menu" onClick={(event) => setMenu(event.currentTarget)}
        aria-label={t(`Add annotation to panel ${label}`, `添加标注到面板 ${label}`)}>
        {t("Add", "添加")}
      </Button>
      <Menu anchorEl={menu} open={!!menu} onClose={() => setMenu(null)}>
        {KINDS.map((kind) => <MenuItem key={kind} onClick={() => {
          setMenu(null);
          const note = noteDefaults(kind, axes, panel.axes);
          onChange({ ...panel, notes: [...panel.notes, note.kind === "text" ? { ...note, text: t("Note", "注释") } : note] });
        }}>{kinds[kind]}</MenuItem>)}
      </Menu>
    </Stack>
    {panel.notes.map((note, index) => <Paper key={index} variant="outlined" sx={{ p: 1.5 }} aria-label={`${kinds[note.kind]} ${index + 1}`} component="section">
      <Stack spacing={1}>
        <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between" }}>
          <Typography variant="body2">{kinds[note.kind]}</Typography>
          <IconButton size="small" disabled={disabled} aria-label={t(`Remove ${kinds[note.kind]} ${index + 1}`, `删除${kinds[note.kind]} ${index + 1}`)}
            onClick={() => setNote(index, null)}><DeleteOutlineRoundedIcon fontSize="small" /></IconButton>
        </Stack>
        <NoteFields note={note} axes={axes} disabled={disabled} t={t} onChange={(next) => setNote(index, next)} />
      </Stack>
    </Paper>)}
  </Stack>;
}
