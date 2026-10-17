# -*- coding: utf-8 -*-
"""
Extra metrics: chirality-kept (E/Z-blind) match, E/Z-only match, ECFP4 Tanimoto,
and the error decomposition that separates alkene E/Z loss from tetrahedral loss.

    python tools/eval_extra.py

Writes ``data/eval_results/all_models/``:

    extra_metrics.csv   per model: weighted organic rates for every metric, plus
                        the image-level error decomposition
    extra_metrics.md    the same as two tables

Definitions
-----------
* ``exact``          canonical isomeric SMILES identity (as `eval_ocsr.py`).
* ``chirality_kept`` canonical isomeric identity after clearing double-bond
                     stereo on both sides: tetrahedral ``@``/``@@`` must match,
                     alkene E/Z is ignored. This is the "E/Z-blind" column.
* ``ez_only``        the mirror image: clear tetrahedral chirality on both
                     sides and require the E/Z assignment to match.
* ``tanimoto``       Morgan (radius 2, 2048 bits) Tanimoto between prediction and
                     ground truth; 0 for unparseable output.
* decomposition      over organic images, split every non-exact image into
                     unparseable / skeleton error / E/Z-only stereo loss /
                     tetrahedral-only stereo loss / both stereo losses, and
                     report the conditional stereo precision
                     ``exact / connectivity`` and its E/Z-blind analogue
                     ``chirality_kept / connectivity``.

Rates that mirror the published headline use the same weighting as
`eval_ocsr.py` (compound mean, 0.25 for compact organics). The decomposition is
image level, which is what makes the five buckets add up to 100 %.
"""
from __future__ import annotations

import collections
import json
import os
import sys

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import results_source as source  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.join(ROOT, "data", "eval_results")
OUT_DIR = os.path.join(EVAL_ROOT, "all_models")

_GT_CACHE = {}


def parse(smiles):
    if not (smiles or "").strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def strip_ez(mol):
    """Copy with double-bond stereo cleared, tetrahedral chirality kept."""
    copy = Chem.Mol(mol)
    for bond in copy.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    return copy


def strip_tetrahedral(mol):
    """Copy with atom chirality cleared, double-bond stereo kept."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return copy


def smiles_of(mol, isomeric=True):
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def gt_variants(smiles):
    """Canonical forms of the ground truth, cached per compound."""
    cached = _GT_CACHE.get(smiles)
    if cached is not None:
        return cached
    mol = parse(smiles)
    if mol is None:
        cached = {"exact": None, "chirality": None, "ez": None, "conn": None,
                  "fp": None}
    else:
        cached = {
            "exact": smiles_of(mol),
            "chirality": smiles_of(strip_ez(mol)),
            "ez": smiles_of(strip_tetrahedral(mol)),
            "conn": smiles_of(mol, isomeric=False),
            "fp": fingerprint(mol),
        }
    _GT_CACHE[smiles] = cached
    return cached


def score_image(prediction, truth_smiles):
    truth = gt_variants(truth_smiles)
    mol = parse(prediction)
    if mol is None:
        return {"valid": False, "exact": False, "chirality_kept": False,
                "ez_only": False, "connectivity": False, "tanimoto": 0.0}
    return {
        "valid": True,
        "exact": smiles_of(mol) == truth["exact"],
        "chirality_kept": smiles_of(strip_ez(mol)) == truth["chirality"],
        "ez_only": smiles_of(strip_tetrahedral(mol)) == truth["ez"],
        "connectivity": smiles_of(mol, isomeric=False) == truth["conn"],
        "tanimoto": DataStructs.TanimotoSimilarity(fingerprint(mol), truth["fp"]),
    }


def weighted_rate(rows, field):
    compounds, rates, weights = source.compound_table(rows, "organic", field)
    total = sum(weights)
    return sum(r * w for r, w in zip(rates, weights)) / total if total else 0.0


def weighted_mean(rows, field):
    """Same weighting as `weighted_rate`, for a value that is not a 0/1 hit.

    `compound_table` treats every field as a boolean, which is right for the
    match metrics and wrong for Tanimoto, so averages are taken here instead.
    """
    per_compound = {}
    for row in rows:
        compound = row["compound_id"]
        entry = per_compound.get(compound)
        if entry is None:
            entry = per_compound[compound] = {
                "sum": 0.0, "n": 0,
                "weight": float(row.get("weight") or 0.0),
            }
        entry["sum"] += float(row.get(field) or 0.0)
        entry["n"] += 1
    total = sum(entry["weight"] for entry in per_compound.values())
    if not total:
        return 0.0
    return sum(entry["weight"] * entry["sum"] / entry["n"]
               for entry in per_compound.values()) / total


def main():
    labels = [label for label, _, _ in source.MODELS]
    manifest = {json.loads(l)["image_id"]: json.loads(l)
                for l in open(os.path.join(ROOT, "benchmark", "random_seed42",
                                           "manifest.jsonl"), encoding="utf-8")
                if l.strip()}

    rows_out = []
    for label in labels:
        rows = [row for row in source.load_model_rows(label)
                if row["subset"] == "organic"]
        for row in rows:
            truth = manifest[row["image_id"]]["smiles_isomeric"]
            row.update(score_image(row.get("prediction") or "", truth))

        weighted = {field: weighted_rate(rows, field)
                    for field in ("valid", "exact", "chirality_kept", "ez_only",
                                  "connectivity")}
        weighted["tanimoto"] = weighted_mean(rows, "tanimoto")
        image_tanimoto = sum(row["tanimoto"] for row in rows) / len(rows)

        # image-level decomposition
        n = len(rows)
        buckets = collections.Counter()
        for row in rows:
            if row["exact"]:
                buckets["exact"] += 1
            elif not row["valid"]:
                buckets["unparseable"] += 1
            elif not row["connectivity"]:
                buckets["skeleton_error"] += 1
            elif row["chirality_kept"]:
                buckets["ez_loss"] += 1
            elif row["ez_only"]:
                buckets["tetrahedral_loss"] += 1
            else:
                buckets["both_stereo_loss"] += 1

        connectivity = buckets["exact"] + buckets["ez_loss"] + \
            buckets["tetrahedral_loss"] + buckets["both_stereo_loss"]
        chirality_matched = buckets["exact"] + buckets["ez_loss"]
        tanimoto_skeleton = [row["tanimoto"] for row in rows
                             if row["valid"] and not row["connectivity"]]

        rows_out.append({
            "model": label,
            "organic_weighted_exact_%": round(100 * weighted["exact"], 2),
            "organic_weighted_chirality_%": round(100 * weighted["chirality_kept"], 2),
            "organic_weighted_ez_only_%": round(100 * weighted["ez_only"], 2),
            "organic_weighted_connectivity_%": round(100 * weighted["connectivity"], 2),
            "organic_weighted_tanimoto_%": round(100 * weighted["tanimoto"], 2),
            "organic_image_tanimoto_%": round(100 * image_tanimoto, 2),
            "organic_valid_%": round(100 * weighted["valid"], 2),
            "image_exact_%": round(100 * buckets["exact"] / n, 2),
            "image_ez_loss_%": round(100 * buckets["ez_loss"] / n, 2),
            "image_tetrahedral_loss_%": round(100 * buckets["tetrahedral_loss"] / n, 2),
            "image_both_stereo_loss_%": round(100 * buckets["both_stereo_loss"] / n, 2),
            "image_tetrahedral_or_both_%":
                round(100 * (buckets["tetrahedral_loss"] + buckets["both_stereo_loss"]) / n, 2),
            "image_skeleton_error_%": round(100 * buckets["skeleton_error"] / n, 2),
            "image_unparseable_%": round(100 * buckets["unparseable"] / n, 2),
            "image_connectivity_%": round(100 * connectivity / n, 2),
            "conditional_stereo_precision_%":
                round(100 * buckets["exact"] / connectivity, 2) if connectivity else 0.0,
            "conditional_no_ez_precision_%":
                round(100 * chirality_matched / connectivity, 2) if connectivity else 0.0,
            "skeleton_error_mean_tanimoto_%":
                round(100 * sum(tanimoto_skeleton) / len(tanimoto_skeleton), 2)
                if tanimoto_skeleton else 0.0,
        })
        print(f"  {label:22s} exact {rows_out[-1]['organic_weighted_exact_%']:5.1f} "
              f"chirality {rows_out[-1]['organic_weighted_chirality_%']:5.1f} "
              f"tanimoto {rows_out[-1]['organic_weighted_tanimoto_%']:5.1f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    fields = list(rows_out[0])
    with open(os.path.join(OUT_DIR, "extra_metrics.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(fields) + "\n")
        for row in rows_out:
            fh.write(",".join(str(row[field]) for field in fields) + "\n")

    order = sorted(rows_out, key=lambda row: -row["organic_weighted_exact_%"])
    lines = ["# Extra metrics (chirality-kept, E/Z-only, ECFP4, error decomposition)", "",
             "Computed by `python tools/eval_extra.py`; every rate uses the same weighting "
             "as the headline (compound mean, 0.25 for compact organics).", "",
             "## Stereo-aware matching", "",
             "| model | exact % | chirality-kept (E/Z-blind) % | E/Z-only kept % | "
             "connectivity % | ECFP4 Tanimoto % (image mean) | ECFP4 (compound-weighted) % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in order:
        lines.append(
            f"| **{row['model']}** | {row['organic_weighted_exact_%']:.1f} | "
            f"{row['organic_weighted_chirality_%']:.1f} | "
            f"{row['organic_weighted_ez_only_%']:.1f} | "
            f"{row['organic_weighted_connectivity_%']:.1f} | "
            f"{row['organic_image_tanimoto_%']:.1f} | "
            f"{row['organic_weighted_tanimoto_%']:.1f} |")

    lines += ["", "## Error decomposition (image level, organic)", "",
              "The five loss buckets plus `exact` add up to 100 %.", "",
              "| model | exact % | E/Z loss % | tetrahedral loss % | both lost % | "
              "tetra+only-tetra % | skeleton error % | unparseable % | stereo precision % | "
              "no-E/Z precision % | skeleton-error Tanimoto % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in order:
        lines.append(
            f"| **{row['model']}** | {row['image_exact_%']:.1f} | "
            f"{row['image_ez_loss_%']:.1f} | {row['image_tetrahedral_loss_%']:.1f} | "
            f"{row['image_both_stereo_loss_%']:.1f} | "
            f"{row['image_tetrahedral_or_both_%']:.1f} | "
            f"{row['image_skeleton_error_%']:.1f} | "
            f"{row['image_unparseable_%']:.1f} | "
            f"{row['conditional_stereo_precision_%']:.1f} | "
            f"{row['conditional_no_ez_precision_%']:.1f} | "
            f"{row['skeleton_error_mean_tanimoto_%']:.1f} |")

    lines += ["", "`stereo precision` is `exact / connectivity` (of the predictions with the "
              "right skeleton, how many also have the right stereo); `no-E/Z precision` is "
              "`chirality-kept / connectivity`. The E/Z loss bucket is the set of images "
              "where the skeleton and the tetrahedral centres are right but the alkene "
              "geometry is not."]
    with open(os.path.join(OUT_DIR, "extra_metrics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_DIR}/extra_metrics.csv and extra_metrics.md")


if __name__ == "__main__":
    main()
