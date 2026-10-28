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
  chiral_match   tetrahedral stereo only; blank when the compound has none
  e_z_match      alkene E/Z only; blank when the compound has no E/Z bond
  tautomer       canonical-tautomer InChIKey; blank above 80 heavy atoms
  inchikey_skel  first 14-character InChIKey block (skeleton) matches
  inchikey_full  full InChIKey matches (skeleton + stereo + protonation)

The number to quote is the compound-level ``organic_weighted`` block on the
full manifest: each compound's images are averaged first, then compact
organics (≤20 heavy atoms and ≤2 assigned stereocentres) are weighted 0.25.
``--template acs1996`` is the cheaper headline. v1.0/v1.1 found journal style
moves exact-match by only a few points, so the five ACS variants stand in for
the 7×5 grid. Coordination stays in its own block and is scored with the part
key. Pass ``--manifest benchmark/random_seed42/manifest.jsonl`` only when
reproducing the v1.1 slice; that slice is not the v2.0 headline.

Usage
-----
  python tools/eval_ocsr.py --predictions preds.jsonl
  python tools/eval_ocsr.py --predictions preds.jsonl --template acs1996
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
from rdkit import rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ["has_bridged_ring", "has_spiro", "has_macrocycle", "has_cage",
        "is_stereo_dense", "is_large", "is_organometallic", "has_e_z"]

# Full-manifest headline: one rate per organic compound, then a weighted mean.
# Compact structures stay in the mean at a lower weight so they do not dominate.
# Coordination is reported on its own and does not enter this score.
COMPACT_WEIGHT = 0.25
COMPACT_MAX_HEAVY = 20
COMPACT_MAX_STEREO = 2
TAUTOMER_MAX_HEAVY = 80
_TAUT = rdMolStandardize.TautomerEnumerator()
_TAUT_CACHE = {}
_STEREO_CACHE = {}
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


def stereo_variants(smi):
    """(tetrahedral-only, E/Z-only) canonical SMILES for one structure.

    `tetrahedral-only` clears double-bond stereo and keeps `@`/`@@`;
    `E/Z-only` clears atom chirality and keeps the alkene configuration. They are
    what the `chiral_match` and `e_z_match` columns compare, so that a wrong
    answer can be attributed to the tetrahedral centres or to the alkene
    geometry instead of being lumped into one "stereo wrong" bucket.
    """
    if smi in _STEREO_CACHE:
        return _STEREO_CACHE[smi]
    mol = Chem.MolFromSmiles(smi or "")
    if mol is None or mol.GetNumAtoms() == 0:
        _STEREO_CACHE[smi] = (None, None)
        return _STEREO_CACHE[smi]
    tetrahedral = Chem.Mol(mol)
    for bond in tetrahedral.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    alkene = Chem.Mol(mol)
    for atom in alkene.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    _STEREO_CACHE[smi] = (Chem.MolToSmiles(tetrahedral), Chem.MolToSmiles(alkene))
    return _STEREO_CACHE[smi]


def tautomer_key(smi):
    """Canonical-tautomer InChIKey. Empty or huge inputs are not enumerated."""
    if smi in _TAUT_CACHE:
        return _TAUT_CACHE[smi]
    key = None
    m = Chem.MolFromSmiles(smi or "")
    if m is not None and 0 < m.GetNumAtoms() <= TAUTOMER_MAX_HEAVY:
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
    ap.add_argument("--template", default="",
                    help="score only this drawing template; acs1996 across all pose "
                         "variants is the v2.0 organic headline")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval_results"))
    args = ap.parse_args()

    results, summary, unmasked = score_predictions(
        args.manifest, args.predictions, template=args.template)
    scored = [r for r in results if "valid" in r]

    os.makedirs(args.out, exist_ok=True)
    # Version fingerprint: canonical SMILES and the tautomer enumerator both move
    # between RDKit releases, so a score is only comparable to one taken with the
    # same build. The pinned baseline lives in requirements.txt.
    env = {
        "python": sys.version.split()[0],
        "rdkit": rdBase.rdkitVersion,
        "manifest": os.path.basename(args.manifest),
        "template": args.template or "all",
        "predictions": args.predictions,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_predictions": len(unmasked), "n_scored": len(scored),
                   "unmatched": len(unmasked) - len(scored), "env": env,
                   "breakdown": summary},
                  fh, indent=2)
    with open(os.path.join(args.out, "per_image.jsonl"), "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_summary(summary)
    print(f"\nrdkit {env['rdkit']} / python {env['python']}")
    print(f"wrote {args.out}/summary.json and per_image.jsonl")


def load_manifest_index(path):
    gt = {}
    for r in open(path, "r", encoding="utf-8"):
        if r.strip():
            row = json.loads(r)
            gt[row["image_id"]] = row
            gt[row["image_path"]] = row
    return gt


def score_predictions(manifest_path, predictions_path, template=""):
    """Score a prediction file. Returns (per_image, breakdown, raw_prediction_rows).

    ``template`` restricts the score to one drawing style. Empty means every
    template in the manifest. Filtering happens before any average, so a full
    prediction file can be rescored as ACS-only without being rewritten.
    """

    def rate(rows, field):
        defined = [r for r in rows if "valid" in r and r.get(field) is not None]
        if not defined:
            return None
        return sum(1 for r in defined if r.get(field)) / len(defined)

    def image_block(name, rows):
        scored_rows = [r for r in rows if "valid" in r]
        out = {
            "group": name, "n": len(scored_rows), "unit": "image",
            "valid": rate(rows, "valid"), "exact": rate(rows, "exact"),
            "tautomer": rate(rows, "tautomer"),
            "connectivity": rate(rows, "connectivity"),
            "chiral_match": rate(rows, "chiral_match"),
            "e_z_match": rate(rows, "e_z_match"),
            "inchikey_skel": rate(rows, "inchikey_skel"),
            "inchikey_full": rate(rows, "inchikey_full"),
        }
        return out

    gt = load_manifest_index(manifest_path)
    preds = load_rows(predictions_path)
    results = []
    for p in preds:
        key = p.get("image_id") or p.get("image_path")
        ref = gt.get(key)
        if ref is None:
            results.append({"key": key, "matched_row": False})
            continue
        if template and ref.get("template") != template:
            continue
        pred = (p.get("prediction") or p.get("smiles") or "").strip()
        truth = ref["smiles_isomeric"]
        heavy = int(ref.get("n_heavy_atoms") or 0)
        n_stereo = int(ref.get("n_stereo_assigned") or 0)
        has_ez = bool(ref.get("has_e_z"))
        c_pred, c_true = canon(pred), canon(truth)
        ik_p, ik_t = inchikey(pred), inchikey(truth)
        if heavy > TAUTOMER_MAX_HEAVY:
            tautomer = None
        else:
            tk_p, tk_t = tautomer_key(pred), tautomer_key(truth)
            tautomer = bool(tk_p and tk_t and tk_p == tk_t)
        tet_p, ez_p = stereo_variants(pred)
        tet_t, ez_t = stereo_variants(truth)
        gt_parts = _coord_parts().get(ref["compound_id"])
        pred_parts = None
        if gt_parts and ref.get("subset") == "coordination":
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
            "tautomer": tautomer,
            "connectivity": bool(canon(pred, False) and canon(truth, False)
                                 and canon(pred, False) == canon(truth, False)),
            "chiral_match": (bool(tet_p and tet_t and tet_p == tet_t)
                             if n_stereo else None),
            "e_z_match": (bool(ez_p and ez_t and ez_p == ez_t) if has_ez else None),
            "inchikey_full": bool(ik_p and ik_t and ik_p == ik_t),
            "inchikey_skel": bool(ik_p and ik_t and ik_p[:14] == ik_t[:14]),
            "parts": (bool(_parts_key(pred_parts) == _parts_key(gt_parts))
                      if gt_parts and ref.get("subset") == "coordination" else None),
            "prediction": pred,
            **{t: ref.get(t) for t in TAGS},
        }
        row["is_compact"] = is_compact(row)
        row["weight"] = organic_weight(row)
        results.append(row)

    scored = [r for r in results if "valid" in r]
    organic = [r for r in scored if r.get("subset") == "organic"]
    headline = ("organic_weighted_" + template) if template else "organic_weighted"
    summary = [
        compound_block(headline, organic, organic_weight),
        compound_block("organic_compact",
                       [r for r in organic if r.get("is_compact")], lambda r: 1.0),
        compound_block("coordination",
                       [r for r in scored if r.get("subset") == "coordination"],
                       lambda r: 1.0),
        image_block("ALL", scored),
    ]
    for field in ("template", "variant"):
        for val in sorted({r[field] for r in organic}):
            summary.append(compound_block(
                f"{field}={val}", [r for r in organic if r[field] == val],
                organic_weight))
    for tag in TAGS:
        sub = [r for r in organic if r.get(tag)]
        if sub:
            summary.append(compound_block(f"tag={tag}", sub, organic_weight))
    hard = [r for r in organic
            if r.get("has_bridged_ring") and r.get("is_stereo_dense")]
    if hard:
        summary.append(compound_block(
            "tag=bridged+stereo_dense", hard, organic_weight))
    return results, summary, preds


SCORE_FIELDS = ("valid", "exact", "tautomer", "connectivity", "chiral_match",
                 "e_z_match", "inchikey_skel", "inchikey_full", "parts")


def compound_block(name, rows, weight_fn):
    """One rate per compound, then a weighted mean.

    A metric that is null on an image (no E/Z bond, no tetrahedral centre, or a
    molecule too large to enumerate tautomers) is left out of that compound's
    rate. A compound with no defined image for a metric is left out of that
    metric's mean, so the denominator is the compounds the metric applies to.
    """
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["compound_id"]].append(row)
    values = {field: [] for field in SCORE_FIELDS}
    weights = {field: [] for field in SCORE_FIELDS}
    n_compounds = 0
    for items in grouped.values():
        weight = float(weight_fn(items[0]))
        if weight <= 0:
            continue
        n_compounds += 1
        for field in SCORE_FIELDS:
            defined = [item.get(field) for item in items if item.get(field) is not None]
            if not defined:
                continue
            values[field].append(sum(bool(item) for item in defined) / len(defined))
            weights[field].append(weight)

    def avg(field):
        pair = weights[field]
        if not pair:
            return None
        total = sum(pair)
        return sum(w * v for w, v in zip(pair, values[field])) / total

    out = {field: avg(field) for field in SCORE_FIELDS}
    out.update({
        "group": name,
        "n": n_compounds,
        "unit": "compound",
        "weight_sum": round(sum(weights["exact"]), 3),
        "n_chiral": len(weights["chiral_match"]),
        "n_e_z": len(weights["e_z_match"]),
        "n_tautomer": len(weights["tautomer"]),
        "n_parts": len(weights["parts"]),
    })
    return out


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(summary):
    headline = next((b for b in summary if str(b.get("group", "")).startswith(
        "organic_weighted")), None)
    if headline:
        scope = headline["group"].replace("organic_weighted_", "").replace(
            "organic_weighted", "all templates")
        print(f"headline: {headline['group']}  "
              f"(compound mean, compact weight {COMPACT_WEIGHT}; scope {scope})")
        print("quote this row, not the image-level ALL row")
    w = "{:<32} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(w.format("group", "n", "exact", "conn", "chiral", "e/z", "taut"))
    print("-" * 90)
    for b in summary:
        unit = " cmp" if b.get("unit") == "compound" else " img"
        print(w.format(
            b["group"], str(b["n"]) + unit, _fmt(b.get("exact")),
            _fmt(b.get("connectivity")), _fmt(b.get("chiral_match")),
            _fmt(b.get("e_z_match")), _fmt(b.get("tautomer"))))
        if b.get("unit") == "compound" and b["group"].startswith("organic_weighted"):
            print(f"  chiral compounds {b.get('n_chiral')}  "
                  f"E/Z compounds {b.get('n_e_z')}  "
                  f"tautomer defined {b.get('n_tautomer')} "
                  f"(skipped {b['n'] - b.get('n_tautomer', 0)} over "
                  f"{TAUTOMER_MAX_HEAVY} heavy atoms)")
            print(f"  weights: compact {COMPACT_WEIGHT}, others 1; "
                  f"weight sum {b['weight_sum']}")
        elif b.get("group") == "coordination" and b.get("n_parts"):
            print(f"  part-key { _fmt(b.get('parts')) } over {b['n_parts']} compounds")


if __name__ == "__main__":
    main()
