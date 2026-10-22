# -*- coding: utf-8 -*-
"""
Machine-check the numbers printed in the public README against what the scoring
tools compute.

    python tools/verify_reported_numbers.py

Parses the v2.0 leaderboard columns in README.md and compares
them with `data/eval_results/all_models/{summary.csv,extra_metrics.csv}`. Writes
`data/eval_results/all_models/reproduction_check.md`.

Why: the tables in `README.md` are hand-assembled from the eval output.
A silent edit or a stale run would not be noticed otherwise, and a benchmark
whose headline numbers cannot be re-derived from its own tools is not a
benchmark.
"""
from __future__ import annotations

import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
OUT = os.path.join(ROOT, "data", "eval_results", "all_models")

# column in the published table -> (source csv, column in that csv)
# v2.0 leaderboard cell indexes after the model and paradigm columns.
COLUMNS = [
    ("valid", "extra_metrics.csv", "organic_valid_%", 2),
    ("exact", "summary.csv", "organic_exact_%", 3),
    ("connectivity", "summary.csv", "organic_connectivity_%", 5),
    ("chirality", "extra_metrics.csv", "organic_weighted_chirality_%", 8),
    ("tautomer", "summary.csv", "organic_tautomer_%", 9),
    ("tanimoto", "extra_metrics.csv", "organic_image_tanimoto_%", 10),
]

# Table 3 (image-level error decomposition) -> column in extra_metrics.csv
DECOMPOSITION = [
    ("exact", "image_exact_%"),
    ("ez_loss", "image_ez_loss_%"),
    ("tetrahedral_loss", "image_tetrahedral_or_both_%"),
    ("skeleton_error", "image_skeleton_error_%"),
    ("unparseable", "image_unparseable_%"),
    ("stereo_precision", "conditional_stereo_precision_%"),
    ("skeleton_error_tanimoto", "skeleton_error_mean_tanimoto_%"),
]

FIRST_FLOAT = re.compile(r"-?\d+(?:\.\d+)?")


def load_csv(name):
    path = os.path.join(OUT, name)
    with open(path, encoding="utf-8") as fh:
        return {row["model"]: row for row in csv.DictReader(fh)}


def parse_table(text):
    """Rows of the leaderboard table: model name -> list of numeric cells."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("| Model | Paradigm & Backbone"))
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        name = cells[0].replace("*", "").split(" (")[0].strip()
        numbers = []
        for cell in cells[2:]:
            match = FIRST_FLOAT.search(cell.replace("*", "").replace(",", ""))
            numbers.append(float(match.group()) if match else None)
        rows[name] = numbers
    return rows


def parse_decomposition(text):
    """Rows of the Table 3 decomposition: model -> 7 numeric cells."""
    lines = text.splitlines()
    # the header was relabelled once the denominators were written down; accept both
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("| Model | Exact (image level) %")
                 or line.startswith("| Model | Exact Match %"))
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        name = cells[0].replace("*", "").strip()
        numbers = []
        for cell in cells[1:8]:
            match = FIRST_FLOAT.search(cell.replace("*", "").replace(",", ""))
            numbers.append(float(match.group()) if match else None)
        rows[name] = numbers
    return rows


def main():
    text = open(README, encoding="utf-8").read()
    published = parse_table(text)
    sources = {name: load_csv(name) for _, name, _ in COLUMNS}

    lines = ["# Reproduction check: published table vs scoring tools", "",
             "Every numeric column of the leaderboard in `publish/README.md` is "
             "compared with `data/eval_results/all_models/`, which is produced by "
             "`tools/collect_results.py` and `tools/eval_extra.py`. Tolerance 0.5 pp "
             "(the published table is rounded to one decimal).", "",
             "| model | column | published | computed | delta (pp) | result |",
             "|---|---|---:|---:|---:|---|"]
    failures = 0
    checked = 0
    for model, numbers in published.items():
        for label, csv_name, column, index in COLUMNS:
            value = numbers[index - 2] if 0 <= index - 2 < len(numbers) else None
            computed_row = sources[csv_name].get(model)
            if computed_row is None or value is None:
                lines.append(f"| {model} | {label} | {value} | — | — | not found |")
                failures += 1
                continue
            computed = float(computed_row[column])
            delta = computed - value
            checked += 1
            ok = abs(delta) <= 0.5
            if not ok:
                failures += 1
            lines.append(f"| {model} | {label} | {value:.1f} | {computed:.2f} | "
                         f"{delta:+.2f} | {'ok' if ok else '**MISMATCH**'} |")

    decomposition = parse_decomposition(text)
    extra = load_csv("extra_metrics.csv")
    lines += ["", "## Table 3 (error decomposition, image level)", "",
              "| model | bucket | published | computed | delta (pp) | result |",
              "|---|---|---:|---:|---:|---|"]
    for model, numbers in decomposition.items():
        for (label, column), value in zip(DECOMPOSITION, numbers):
            row = extra.get(model)
            if row is None or value is None:
                continue
            computed = float(row[column])
            delta = computed - value
            checked += 1
            ok = abs(delta) <= 0.5
            if not ok:
                failures += 1
            lines.append(f"| {model} | {label} | {value:.1f} | {computed:.2f} | "
                         f"{delta:+.2f} | {'ok' if ok else '**MISMATCH**'} |")

    lines += ["", f"Checked {checked} cells: "
              f"**{checked - failures} match, {failures} mismatch**."]
    if failures == 0:
        lines.append("")
        lines.append("The published leaderboard reproduces from the repository tools.")
    path = os.path.join(OUT, "reproduction_check.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"checked {checked} cells, {failures} mismatches -> {path}")
    for line in lines[-3:]:
        if line:
            print(line)


if __name__ == "__main__":
    main()
