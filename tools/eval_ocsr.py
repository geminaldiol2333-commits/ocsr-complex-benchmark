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

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ["has_bridged_ring", "has_spiro", "has_macrocycle", "has_cage",
        "is_stereo_dense", "is_large", "is_organometallic"]


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
        results.append({
            "image_id": ref["image_id"],
            "image_path": ref["image_path"],
            "compound_id": ref["compound_id"],
            "subset": ref["subset"],
            "template": ref["template"],
            "variant": ref["variant"],
            "valid": c_pred is not None,
            "exact": bool(c_pred and c_true and c_pred == c_true),
            "connectivity": bool(canon(pred, False) and canon(truth, False)
                                 and canon(pred, False) == canon(truth, False)),
            "inchikey_full": bool(ik_p and ik_t and ik_p == ik_t),
            "inchikey_skel": bool(ik_p and ik_t and ik_p[:14] == ik_t[:14]),
            "prediction": pred,
            **{t: ref.get(t) for t in TAGS},
        })

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
    return results, summary, preds


def print_summary(summary):
    w = "{:<32} {:>7} {:>8} {:>8} {:>8} {:>10} {:>10}"
    print(w.format("group", "n", "valid", "exact", "conn", "ikey_skel", "ikey_full"))
    print("-" * 88)
    for b in summary:
        print(w.format(b["group"], b["n"], f"{b['valid']:.3f}", f"{b['exact']:.3f}",
                       f"{b['connectivity']:.3f}", f"{b['inchikey_skel']:.3f}",
                       f"{b['inchikey_full']:.3f}"))


if __name__ == "__main__":
    main()
