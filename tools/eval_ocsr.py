# -*- coding: utf-8 -*-
"""
Scoring harness for the OCSR-complex benchmark.

Takes a prediction file and reports recognition accuracy with the breakdowns
that matter for this dataset: by subset (organic vs coordination), by drawing
template, by geometric variant, and by structural difficulty tag
(bridged / spiro / macrocycle / cage / stereo-dense / large).

Prediction file format (JSONL or CSV), one row per image:
    {"image_id": "...", "prediction": "<SMILES>"}
or  {"image_path": "organic/images/acs1996/paclitaxel/v0_orig.png",
     "prediction": "..."}

Metrics
-------
  valid          prediction parses as a molecule
  exact          canonical isomeric SMILES identical to the ground truth
  connectivity   canonical SMILES ignoring stereochemistry
  inchikey_skel  first 14-character InChIKey block (skeleton) matches
  inchikey_full  full InChIKey matches (skeleton + stereo + protonation)

Usage
-----
  python tools/eval_ocsr.py --predictions preds.jsonl
  python tools/eval_ocsr.py --predictions preds.csv --out results/
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ["has_bridged_ring", "has_spiro", "has_macrocycle", "has_cage",
        "is_stereo_dense", "is_large", "is_organometallic"]

# v1.1 headline: organic compounds, with compact structures down-weighted.
# Coordination is reported on its own and does not enter this score.
COMPACT_WEIGHT = 0.25
COMPACT_MAX_HEAVY = 20
COMPACT_MAX_STEREO = 2
_TAUT = rdMolStandardize.TautomerEnumerator()
_TAUT_CACHE = {}
_PARTS = None


def _coord_parts():
    """Depiction keys for redrawn complexes. Missing file -> no parts score."""
    global _PARTS
    if _PARTS is None:
        path = os.path.join(ROOT, "data", "coord_parts.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                _PARTS = json.load(fh)
        else:
            _PARTS = {}
    return _PARTS


def _parts_key(parts):
    if not parts:
        return None
    return (tuple(parts.get("metals") or ()), tuple(parts.get("ligands") or ()))


def load_rows(path):
    rows = []
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    else:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def canon(smi, iso=True):
    """Canonical SMILES, or None when the input is empty / unparseable.

    Note: RDKit happily parses the empty string into a molecule with zero atoms,
    which would otherwise be counted as a *valid* prediction when a model
    answers with prose instead of a structure.
    """
    if not (smi or "").strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=iso)


def is_compact(row):
    """Small organic structures. They stay in the headline at COMPACT_WEIGHT."""
    if row.get("subset") != "organic" or row.get("is_organometallic"):
        return False
    heavy, stereo = row.get("n_heavy_atoms"), row.get("n_stereo_assigned")
    if heavy is None or stereo is None:
        return False
    return int(heavy) <= COMPACT_MAX_HEAVY and int(stereo) <= COMPACT_MAX_STEREO


def organic_weight(row):
    if row.get("subset") != "organic":
        return 0.0
    return COMPACT_WEIGHT if is_compact(row) else 1.0


def tautomer_key(smi):
    """Canonical-tautomer InChIKey. Empty or huge inputs are not enumerated."""
    if smi in _TAUT_CACHE:
        return _TAUT_CACHE[smi]
    key = None
    m = Chem.MolFromSmiles(smi or "")
    if m is not None and 0 < m.GetNumAtoms() <= 80:
        try:
            parent = _TAUT.Canonicalize(m)
            key = Chem.MolToInchiKey(parent)
        except Exception:
            key = None
    _TAUT_CACHE[smi] = key
    return key


def inchikey(smi):
    if not (smi or "").strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "benchmark",
                                                       "manifest_all.jsonl"))
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval_results"))
    args = ap.parse_args()

    results, summary, unmasked = score_predictions(args.manifest, args.predictions)
    scored = [r for r in results if "valid" in r]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_predictions": len(unmasked), "n_scored": len(scored),
                   "unmatched": len(unmasked) - len(scored), "breakdown": summary},
                  fh, indent=2)
    with open(os.path.join(args.out, "per_image.jsonl"), "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_summary(summary)
    print(f"\nwrote {args.out}/summary.json and per_image.jsonl")


def load_manifest_index(path):
    gt = {}
    for r in open(path, "r", encoding="utf-8"):
        if r.strip():
            row = json.loads(r)
            gt[row["image_id"]] = row
            gt[row["image_path"]] = row
    return gt


def score_predictions(manifest_path, predictions_path):
    """Score a prediction file. Returns (per_image, breakdown, raw_prediction_rows)."""

    def rate(rows, field):
        n = sum(1 for r in rows if r.get(field))
        d = sum(1 for r in rows if "valid" in r)
        return (n / d) if d else 0.0

    def block(name, rows):
        return {
            "group": name, "n": sum(1 for r in rows if "valid" in r),
            "valid": rate(rows, "valid"), "exact": rate(rows, "exact"),
            "tautomer": rate(rows, "tautomer"),
            "connectivity": rate(rows, "connectivity"),
            "inchikey_skel": rate(rows, "inchikey_skel"),
            "inchikey_full": rate(rows, "inchikey_full"),
        }

    gt = load_manifest_index(manifest_path)
    preds = load_rows(predictions_path)
    results = []
    for p in preds:
        key = p.get("image_id") or p.get("image_path")
        ref = gt.get(key)
        if ref is None:
            results.append({"key": key, "matched_row": False})
            continue
        pred = (p.get("prediction") or p.get("smiles") or "").strip()
        truth = ref["smiles_isomeric"]
        c_pred, c_true = canon(pred), canon(truth)
        ik_p, ik_t = inchikey(pred), inchikey(truth)
        tk_p, tk_t = tautomer_key(pred), tautomer_key(truth)
        gt_parts = _coord_parts().get(ref["compound_id"])
        pred_parts = None
        if gt_parts:
            from coord_depict import parts_from_smiles
            pred_parts = parts_from_smiles(pred)
        row = {
            "image_id": ref["image_id"],
            "image_path": ref["image_path"],
            "compound_id": ref["compound_id"],
            "subset": ref["subset"],
            "template": ref["template"],
            "variant": ref["variant"],
            "n_heavy_atoms": ref.get("n_heavy_atoms"),
            "n_stereo_assigned": ref.get("n_stereo_assigned"),
            "valid": c_pred is not None,
            "exact": bool(c_pred and c_true and c_pred == c_true),
            "tautomer": bool(tk_p and tk_t and tk_p == tk_t),
            "connectivity": bool(canon(pred, False) and canon(truth, False)
                                 and canon(pred, False) == canon(truth, False)),
            "inchikey_full": bool(ik_p and ik_t and ik_p == ik_t),
            "inchikey_skel": bool(ik_p and ik_t and ik_p[:14] == ik_t[:14]),
            "parts": bool(gt_parts and _parts_key(pred_parts) == _parts_key(gt_parts)),
            "prediction": pred,
            **{t: ref.get(t) for t in TAGS},
        }
        row["is_compact"] = is_compact(row)
        row["weight"] = organic_weight(row)
        results.append(row)

    scored = [r for r in results if "valid" in r]
    summary = [block("ALL", scored)]
    for field in ("subset", "template", "variant"):
        for val in sorted({r[field] for r in scored}):
            summary.append(block(f"{field}={val}", [r for r in scored if r[field] == val]))
    for tag in TAGS:
        sub = [r for r in scored if r.get(tag)]
        if sub:
            summary.append(block(f"tag={tag}", sub))
    hard = [r for r in scored if r.get("has_bridged_ring") and r.get("is_stereo_dense")]
    if hard:
        summary.append(block("tag=bridged+stereo_dense", hard))
    organic = [r for r in scored if r.get("subset") == "organic"]
    summary.insert(0, compound_block("organic_weighted", organic, organic_weight))
    summary.insert(1, compound_block(
        "organic_compact", [r for r in organic if r.get("is_compact")],
        lambda r: 1.0))
    summary.insert(2, compound_block(
        "coordination", [r for r in scored if r.get("subset") == "coordination"],
        lambda r: 1.0))
    return results, summary, preds


def compound_block(name, rows, weight_fn):
    """One rate per compound, then a weighted mean. `n` is the compound count."""
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["compound_id"]].append(row)
    buckets = {k: [] for k in ("valid", "exact", "tautomer", "connectivity",
                               "inchikey_skel", "inchikey_full", "parts")}
    weights = []
    for items in grouped.values():
        weight = float(weight_fn(items[0]))
        if weight <= 0:
            continue
        weights.append(weight)
        for field in buckets:
            buckets[field].append(sum(bool(i.get(field)) for i in items) / len(items))
    total = sum(weights) or 1.0

    def avg(vals):
        return sum(w * v for w, v in zip(weights, vals)) / total if weights else 0.0

    out = {field: avg(vals) for field, vals in buckets.items()}
    out.update({"group": name, "n": len(weights), "unit": "compound",
                "weight_sum": round(sum(weights), 3)})
    return out


def print_summary(summary):
    w = "{:<32} {:>7} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}"
    print(w.format("group", "n", "valid", "exact", "taut", "conn", "ikey_skel", "ikey_full"))
    print("-" * 100)
    for b in summary:
        unit = " cmp" if b.get("unit") == "compound" else ""
        print(w.format(
            b["group"], str(b["n"]) + unit, f"{b['valid']:.3f}", f"{b['exact']:.3f}",
            f"{b.get('tautomer', 0):.3f}", f"{b['connectivity']:.3f}",
            f"{b['inchikey_skel']:.3f}", f"{b['inchikey_full']:.3f}"))
        if b.get("weight_sum") is not None and b["group"] == "organic_weighted":
            print(f"  weights: compact {COMPACT_WEIGHT}, others 1; "
                  f"weight sum {b['weight_sum']}")


if __name__ == "__main__":
    main()
