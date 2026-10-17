# -*- coding: utf-8 -*-
"""
One place that says which scored run an image's prediction comes from.

Four models (OCSRGlyph, TinyChemVL, MolSight, MolSight-stereo) were run before
the coordination complexes were redrawn in v1.1, so their full-grid prediction
file still describes the old drawings. Their coordination column was re-run on
the redrawn complexes and lives in a separate `*_coord` run. Every table the
project publishes uses the re-run for those four - this module applies that rule
when it loads rows, so `collect_results.py` and `stats_ci.py` cannot drift apart
from the README.

A model without a coordination re-run keeps its single run for both subsets.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.join(ROOT, "data", "eval_results")

# label, run used for the organic subset, coordination re-run ("" = none)
MODELS = [
    ("DECIMER", "v11_seed42_decimer", ""),
    ("MolNexTR", "v11_seed42_molnextr", ""),
    ("MolScribe", "v11_seed42_molscribe", ""),
    ("COMO", "v11_seed42_como", ""),
    ("MolParser-Mobile-V2", "v11_seed42_molparser_mobile_v2", ""),
    ("MolSight", "v11_seed42_molsight", "v11_seed42_molsight_coord"),
    ("MolSight-stereo", "v11_seed42_molsight_stereo", "v11_seed42_molsight_stereo_coord"),
    ("OCSRGlyph", "v11_seed42_ocsrglyph", "v11_seed42_ocsrglyph_coord"),
    ("TinyChemVL", "v11_seed42_tinychemvl", "v11_seed42_tinychemvl_coord"),
    ("MolGrapher", "v11_seed42_molgrapher", ""),
    ("MarkushGlyph", "v11_seed42_markushglyph", ""),
]

SCORE_FIELDS = ["valid", "exact", "tautomer", "connectivity",
                "inchikey_skel", "inchikey_full", "parts"]
TAGS = ["has_bridged_ring", "has_spiro", "has_macrocycle", "has_cage",
        "is_stereo_dense", "is_large", "is_organometallic"]


def _load_run(run):
    path = os.path.join(EVAL_ROOT, run, "per_image.jsonl")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if "valid" in row:
                    rows.append(row)
    return rows


def load_model_rows(label):
    """Scored rows for one model: organic from its run, coordination from the re-run."""
    entry = next(m for m in MODELS if m[0] == label)
    _, organic_run, coordination_run = entry
    rows = {row["image_id"]: row for row in _load_run(organic_run)}
    if coordination_run:
        for row in _load_run(coordination_run):
            rows[row["image_id"]] = row
    return list(rows.values())


def load_reference_rows():
    """The union of every model's rows, as a source of metadata."""
    merged = {}
    for label, _, _ in MODELS:
        for row in load_model_rows(label):
            merged.setdefault(row["image_id"], row)
    return list(merged.values())


def compound_table(rows, subset, field, unit_weight=False):
    """Per-compound mean of `field`, plus the weight that compound carries.

    Mirrors `compound_block` in eval_ocsr.py: a compound contributes its mean
    per-image score and takes the weight of its first row (0.25 for compact
    organics, 1 otherwise).
    """
    per_compound = {}
    for row in rows:
        if row["subset"] != subset:
            continue
        compound = row["compound_id"]
        entry = per_compound.get(compound)
        if entry is None:
            entry = per_compound[compound] = {
                "hits": 0, "n": 0,
                "weight": 1.0 if unit_weight else float(row.get("weight") or 0.0),
            }
        entry["hits"] += 1 if row.get(field) else 0
        entry["n"] += 1
    compounds = sorted(per_compound)
    rates = [per_compound[c]["hits"] / per_compound[c]["n"] for c in compounds]
    weights = [per_compound[c]["weight"] for c in compounds]
    return compounds, rates, weights


def block(rows, subset, fields=SCORE_FIELDS, unit_weight=False):
    """Weighted mean over compounds for each field, in one pass."""
    out = {}
    for field in fields:
        compounds, rates, weights = compound_table(rows, subset, field, unit_weight)
        total = sum(weights)
        out[field] = sum(r * w for r, w in zip(rates, weights)) / total if total else 0.0
    out["n"] = sum(1 for w in
                   compound_table(rows, subset, fields[0], unit_weight)[2] if w > 0)
    return out
