"""Synthetic datasets for the V2.5 reference set, one per reference figure design (R01-R20).

Every number is invented: each entry copies only the design of a published figure (groups,
pairing, factors, time points, censoring), never its values. Each entry draws from NumPy's
``default_rng`` with its own fixed seed, but uses only the uniform stream and turns it into
normal, log-normal and exponential draws here (Box-Muller, inverse CDF), then rounds. The files
therefore depend on the seed and this code, not on NumPy's distribution algorithms.

Rewrite the files from ``V2.0/api`` with ``python -m scripts.build_v25_reference_fixtures``;
``tests/test_v25_reference_fixtures.py`` checks the committed bytes against ``build()``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "contracts" / "fixtures" / "v25"
BASE_SEED = 2025


class Draws:
    """Random draws for one entry, built from PCG64 uniforms only."""

    def __init__(self, entry: int) -> None:
        self._rng = np.random.default_rng([BASE_SEED, entry])

    def uniform(self) -> float:
        """A draw in (0, 1], so that its logarithm is finite."""
        return 1.0 - float(self._rng.random())

    def normal(self, mean: float, sd: float) -> float:
        radius = math.sqrt(-2.0 * math.log(self.uniform()))
        return mean + sd * radius * math.cos(2.0 * math.pi * self.uniform())

    def lognormal(self, median: float, sigma: float) -> float:
        return median * math.exp(self.normal(0.0, sigma))

    def exponential(self, mean: float) -> float:
        return -mean * math.log(self.uniform())


def num(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"
    # A value that rounds to zero is written without a sign ("-0.000" -> "0.000").
    return text.lstrip("-") if float(text) == 0.0 else text


Rows = list[list[str]]


def r01(d: Draws) -> Rows:
    rows: Rows = []
    for prefix, group, mean, sd in (("C", "Control", 1.0, 0.25), ("T", "Treated", 1.45, 0.45)):
        rows += [[f"{prefix}{i:02d}", group, num(d.normal(mean, sd), 3)] for i in range(1, 9)]
    return rows


def r02(d: Draws) -> Rows:
    rows: Rows = []
    for prefix, group, n, median, sigma in (
        ("C", "Control", 7, 12.0, 0.6),
        ("M", "Mutant", 6, 4.0, 0.8),
    ):
        rows += [
            [f"{prefix}{i:02d}", group, str(round(d.lognormal(median, sigma)))]
            for i in range(1, n + 1)
        ]
    return rows


def r03(d: Draws) -> Rows:
    rows: Rows = []
    for i in range(1, 7):
        before = d.normal(45.0, 8.0)
        after = before - 9.0 + d.normal(0.0, 4.0)
        rows += [[f"s{i:02d}", "Before", num(before, 2)], [f"s{i:02d}", "After", num(after, 2)]]
    return rows


def r04(d: Draws) -> Rows:
    rows: Rows = []
    for i in range(1, 13):
        baseline = d.normal(205.0, 8.0)
        shift = d.lognormal(10.0, 0.8) - 3.0
        rows += [
            [f"m{i:02d}", "Cue absent", num(baseline, 1)],
            [f"m{i:02d}", "Cue present", num(baseline + shift, 1)],
        ]
    return rows


def groups(
    d: Draws, unit: str, spec: tuple[tuple[str, float, float], ...], n: int, digits: int
) -> Rows:
    rows: Rows = []
    for index, (group, mean, sd) in enumerate(spec):
        rows += [
            [f"{unit}{index + 1}-{i:02d}", group, num(d.normal(mean, sd), digits)]
            for i in range(1, n + 1)
        ]
    return rows


def r05(d: Draws) -> Rows:
    spec = (
        ("Untreated", 1.0, 0.4),
        ("Treatment A", 1.3, 0.4),
        ("Treatment B", 4.4, 0.4),
        ("Treatment C", 5.9, 0.4),
    )
    return groups(d, "w", spec, 6, 3)


def r06(d: Draws) -> Rows:
    spec = (
        ("T1", 5.8, 0.3),
        ("T2", 5.7, 0.3),
        ("T3", 5.3, 0.3),
        ("T4", 4.7, 0.3),
        ("T5", 3.3, 0.3),
    )
    return groups(d, "s", spec, 6, 3)


def r07(d: Draws) -> Rows:
    spec = (("Wild type", 18.0, 3.0), ("Variant A", 26.0, 6.0), ("Variant B", 13.0, 2.5))
    return groups(d, "a", spec, 6, 2)


def r08(d: Draws) -> Rows:
    rows: Rows = []
    for index, (group, median) in enumerate(
        (("G1", 60.0), ("G2", 110.0), ("G3", 165.0), ("G4", 300.0))
    ):
        rows += [
            [f"w{index + 1}-{i}", group, num(d.lognormal(median, 0.35), 1)] for i in range(1, 5)
        ]
    return rows


def r09(d: Draws) -> Rows:
    rows: Rows = []
    cells = (
        ("Wild type", "Vehicle", 30.0, 6),
        ("Wild type", "Drug", 25.0, 6),
        ("Knockout", "Vehicle", 55.0, 6),
        ("Knockout", "Drug", 38.0, 5),
    )
    for index, (genotype, treatment, mean, n) in enumerate(cells):
        rows += [
            [f"a{index + 1}-{i:02d}", genotype, treatment, num(d.normal(mean, 5.0), 2)]
            for i in range(1, n + 1)
        ]
    return rows


def r10(d: Draws) -> Rows:
    rows: Rows = []
    for prefix, group, mean in (("c", "Control", 1.4), ("m", "Mutant", 0.9)):
        for animal in range(1, 6):
            animal_mean = mean + d.normal(0.0, 0.25)
            rows += [
                [f"{prefix}{animal}", group, str(repeat), num(animal_mean + d.normal(0.0, 0.1), 3)]
                for repeat in range(1, 5)
            ]
    return rows


def r11(d: Draws) -> Rows:
    rows: Rows = []
    for i in range(1, 61):
        rows.append([f"A{i:02d}", "Group A", num(d.normal(2.0, 0.55), 3)])
    for i in range(1, 61):
        rows.append([f"B{i:02d}", "Group B", num(d.normal(1.8, 0.6), 3)])
    for i in range(1, 61):
        value = d.normal(1.1, 0.3) if d.uniform() <= 0.6 else d.normal(2.2, 0.35)
        rows.append([f"C{i:02d}", "Group C", num(value, 3)])
    return rows


R12_TIMES = (0, 15, 30, 60, 90, 120)


def r12(d: Draws) -> Rows:
    rows: Rows = []
    for index, (group, amplitude) in enumerate(
        (("Control", 180.0), ("Diet", 260.0), ("Diet + drug", 210.0))
    ):
        for animal in range(1, 9):
            baseline = d.normal(120.0, 10.0)
            scale = d.normal(1.0, 0.12)
            for minute in R12_TIMES:
                shape = (minute / 30.0) * math.exp(1.0 - minute / 30.0)
                value = baseline + amplitude * scale * shape + d.normal(0.0, 12.0)
                rows.append([f"g{index + 1}-{animal}", group, str(minute), num(value, 1)])
    return rows


R13_CONCENTRATIONS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def r13(d: Draws) -> Rows:
    rows: Rows = []
    for compound, ec50 in (("Compound A", 0.08), ("Compound B", 0.6)):
        for concentration in R13_CONCENTRATIONS:
            mean = 1.0 + 4.0 / (1.0 + (ec50 / concentration) ** 1.2)
            rows += [
                [f"r{rep}", compound, f"{concentration:g}", num(mean + d.normal(0.0, 0.18), 3)]
                for rep in range(1, 4)
            ]
    return rows


def r14(d: Draws) -> Rows:
    rows: Rows = []
    for substrate in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 12.5):
        mean = 1500.0 * substrate / (2.0 + substrate)
        rows += [
            [f"r{rep}", f"{substrate:g}", num(mean + d.normal(0.0, 0.04 * mean + 15.0), 1)]
            for rep in range(1, 4)
        ]
    return rows


def r15(d: Draws) -> Rows:
    rate = math.log(2.0) / 0.25
    rows: Rows = []
    for hours in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        mean = 0.38 + 0.62 * math.exp(-rate * hours)
        rows += [
            [f"r{rep}", f"{hours:g}", num(mean + d.normal(0.0, 0.03), 3)] for rep in range(1, 4)
        ]
    return rows


R16_FOLLOW_UP = 80


def r16(d: Draws) -> Rows:
    rows: Rows = []
    for index, (arm, mean) in enumerate(
        (("Control", 55.0), ("Treatment A", 50.0), ("Treatment B", 120.0))
    ):
        for mouse in range(1, 11):
            event_time = d.exponential(mean)
            lost = d.uniform() <= 0.1
            time, event = (event_time * d.uniform(), 0) if lost else (event_time, 1)
            if time >= R16_FOLLOW_UP:
                time, event = float(R16_FOLLOW_UP), 0
            rows.append(
                [f"k{index + 1}-{mouse:02d}", arm, str(max(1, math.ceil(time))), str(event)]
            )
    return rows


def r17(_: Draws) -> Rows:
    counts = (("Condition 1", 26, 7), ("Condition 2", 20, 28), ("Condition 3", 14, 43))
    rows: Rows = []
    for condition, first, second in counts:
        rows += [[condition, "Class A", str(first)], [condition, "Class B", str(second)]]
    return rows


def r18(d: Draws) -> Rows:
    rows: Rows = []
    for i in range(1, 13):
        x = 0.6 + 0.65 * d.uniform()
        rows.append([f"p{i:02d}", num(x, 3), num(-9.0 + 10.0 * x + d.normal(0.0, 0.35), 3)])
    return rows


def r19(d: Draws) -> Rows:
    rows: Rows = []
    for blot in range(1, 5):
        control_target = d.normal(1000.0, 80.0)
        rows.append(
            [f"b{blot}", "Control", num(control_target, 1), num(d.normal(2000.0, 100.0), 1)]
        )
        treated_target = control_target * 2.6 * d.lognormal(1.0, 0.12)
        rows.append(
            [f"b{blot}", "Treated", num(treated_target, 1), num(d.normal(2000.0, 100.0), 1)]
        )
    return rows


@dataclass(frozen=True)
class Entry:
    id: str
    slug: str
    design: str
    analysis: str
    chart: str
    milestones: tuple[str, ...]
    columns: tuple[str, ...]
    model: str
    build: Callable[[Draws], Rows] | None = None
    panels: tuple[str, ...] = ()


ENTRIES = (
    Entry(
        "R01",
        "two_groups_unpaired",
        "two groups, unpaired",
        "Welch t",
        "dots, mean ± SD",
        ("M4", "M5"),
        ("animal", "group", "value"),
        "8 per group; normal, means 1.0 and 1.45, SD 0.25 and 0.45",
        r01,
    ),
    Entry(
        "R02",
        "two_groups_skewed",
        "two groups, small n, skewed",
        "Mann-Whitney",
        "dots, median with IQR",
        ("M4", "M5"),
        ("embryo", "group", "count"),
        "7 and 6 per group; rounded log-normal counts, medians 12 and 4",
        r02,
    ),
    Entry(
        "R03",
        "paired",
        "before / after, paired",
        "paired t",
        "paired lines",
        ("M4",),
        ("slice", "condition", "value"),
        "6 slices; before normal 45 (SD 8); after = before - 9 + normal (SD 4)",
        r03,
    ),
    Entry(
        "R04",
        "paired_nonnormal",
        "paired, non-normal",
        "Wilcoxon signed-rank",
        "paired lines",
        ("M4",),
        ("mouse", "condition", "value"),
        "12 mice; shift = log-normal (median 10, sigma 0.8) - 3",
        r04,
    ),
    Entry(
        "R05",
        "doses_vs_control",
        "treatments against control",
        "one-way ANOVA, Dunnett",
        "dots, mean ± SEM",
        ("M4", "M5"),
        ("well", "treatment", "value"),
        "6 per group; normal, SD 0.4, means 1.0, 1.3, 4.4, 5.9",
        r05,
    ),
    Entry(
        "R06",
        "all_pairs",
        "several treatments, all pairs",
        "one-way ANOVA, Tukey",
        "bar with points",
        ("M4", "M5"),
        ("sample", "treatment", "value"),
        "6 per group; normal, SD 0.3, means 5.8, 5.7, 5.3, 4.7, 3.3",
        r06,
    ),
    Entry(
        "R07",
        "unequal_variances",
        "unequal variances",
        "Welch ANOVA, Games-Howell",
        "dots, mean ± SD",
        ("M4", "M5"),
        ("animal", "group", "value"),
        "6 per group; normal, means 18, 26, 13, SD 3, 6, 2.5",
        r07,
    ),
    Entry(
        "R08",
        "groups_nonnormal",
        "several groups, non-normal",
        "Kruskal-Wallis, Dunn (Holm)",
        "box with points",
        ("M4", "M5"),
        ("well", "group", "value"),
        "4 per group; log-normal, sigma 0.35, medians 60, 110, 165, 300",
        r08,
    ),
    Entry(
        "R09",
        "two_factor",
        "genotype × treatment",
        "two-way ANOVA, post-hoc (Holm)",
        "grouped bar with points",
        ("M3", "M4", "M5"),
        ("animal", "genotype", "treatment", "value"),
        "cells of 6, 6, 6 and 5; normal, SD 5, means 30, 25, 55, 38 (interaction)",
        r09,
    ),
    Entry(
        "R10",
        "technical_repeats",
        "technical repeats within animals",
        "t on unit means, repeat warning",
        "dots, units as n",
        ("M4",),
        ("animal", "group", "repeat", "value"),
        "5 animals per group, 4 repeats each; animal SD 0.25, repeat SD 0.1, means 1.4 and 0.9",
        r10,
    ),
    Entry(
        "R11",
        "distributions",
        "distribution comparison",
        "test of choice",
        "violin with points",
        ("M5",),
        ("cell", "group", "value"),
        "60 cells per group; normal 2.0 (0.55), normal 1.8 (0.6), and a 0.6/0.4 mixture of "
        "normal 1.1 (0.3) and 2.2 (0.35)",
        r11,
    ),
    Entry(
        "R12",
        "time_course",
        "time course, several groups",
        "two-way repeated-measures ANOVA (Geisser-Greenhouse), Šidák per time point; area under "
        "the curve per animal",
        "XY mean ± SEM per time",
        ("M3", "M4", "M6"),
        ("animal", "group", "time_min", "glucose"),
        "8 animals per group at 0-120 min; baseline 120 (SD 10) + amplitude 180, 260 or 210 "
        "× animal scale (SD 0.12) × (t/30)·exp(1 - t/30) + noise (SD 12)",
        r12,
    ),
    Entry(
        "R13",
        "dose_response",
        "dose-response, two compounds",
        "4PL, EC50 with interval, comparison",
        "log-x curve",
        ("M6",),
        ("replicate", "compound", "concentration_uM", "response"),
        "3 replicates at 9 concentrations; 4PL bottom 1, top 5, Hill 1.2, EC50 0.08 and 0.6; "
        "noise SD 0.18",
        r13,
    ),
    Entry(
        "R14",
        "enzyme_kinetics",
        "enzyme kinetics",
        "Michaelis-Menten (Vmax, Km)",
        "XY with curve",
        ("M6",),
        ("replicate", "substrate_uM", "rate"),
        "3 replicates at 8 concentrations; Vmax 1500, Km 2; noise SD 4% + 15",
        r14,
    ),
    Entry(
        "R15",
        "decay",
        "protein decay",
        "one-phase decay, half-life",
        "XY with curve",
        ("M6",),
        ("replicate", "time_h", "abundance"),
        "3 replicates at 7 times; plateau 0.38, span 0.62, half-life 0.25 h; noise SD 0.03",
        r15,
    ),
    Entry(
        "R16",
        "survival",
        "survival, three arms",
        "Kaplan-Meier, log-rank",
        "step curves, numbers at risk",
        ("M7",),
        ("mouse", "arm", "time_days", "event"),
        "10 per arm; exponential event times, means 55, 50, 120 days; 10% lost early; "
        "censored at 80 days",
        r16,
    ),
    Entry(
        "R17",
        "proportions",
        "proportions",
        "chi-square or Fisher",
        "stacked percentage bars",
        ("M7",),
        ("condition", "outcome", "count"),
        "fixed counts 26/7, 20/28, 14/43",
        r17,
    ),
    Entry(
        "R18",
        "association",
        "association",
        "Pearson r with interval, regression line",
        "research scatter",
        ("M6",),
        ("sample", "predictor", "response"),
        "12 samples; predictor uniform 0.6-1.25; response = -9 + 10 × predictor + noise (SD 0.35)",
        r18,
    ),
    Entry(
        "R19",
        "normalised",
        "values normalised to control",
        "one-sample t against 1",
        "dots, mean ± SEM",
        ("M4",),
        ("blot", "condition", "target_signal", "loading_signal"),
        "4 blots; control target normal 1000 (SD 80), loading normal 2000 (SD 100); treated "
        "target = control × 2.6 × log-normal (sigma 0.12)",
        r19,
    ),
    Entry(
        "R20",
        "composed",
        "composed figure",
        "panels from the rows above",
        "4 to 6 panels, journal width",
        ("M8",),
        (),
        "no data of its own",
        panels=("R01", "R05", "R09", "R12"),
    ),
)


def write_csv(columns: tuple[str, ...], rows: Rows) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue()


def build() -> dict[str, str]:
    files: dict[str, str] = {}
    entries = []
    for index, entry in enumerate(ENTRIES, start=1):
        record: dict[str, object] = {
            "id": entry.id,
            "design": entry.design,
            "analysis": entry.analysis,
            "chart": entry.chart,
            "milestones": list(entry.milestones),
            "model": entry.model,
        }
        if entry.build is None:
            record.update(file=None, panels=list(entry.panels))
        else:
            rows = entry.build(Draws(index))
            name = f"{entry.id}_{entry.slug}.csv"
            files[name] = write_csv(entry.columns, rows)
            record.update(
                file=name,
                seed=[BASE_SEED, index],
                columns=list(entry.columns),
                rows=len(rows),
                sha256=hashlib.sha256(files[name].encode()).hexdigest(),
            )
        entries.append(record)
    manifest = {
        "schemaVersion": 1,
        "synthetic": True,
        "generator": "build_v25_reference_fixtures.py",
        "entries": entries,
    }
    files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    return files


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, content in build().items():
        (ROOT / name).write_bytes(content.encode())
        print(name)


if __name__ == "__main__":
    main()
