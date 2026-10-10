# -*- coding: utf-8 -*-
"""
Build a randomised evaluation slice ("random scoring mode") of the benchmark.

Sampling contract
-----------------
For every compound:
  * 3 of the 7 drawing templates are selected (semi-random, see below)
  * for each selected template, 2 of the 5 geometric variants are drawn

Template selection is deliberately *not* uniform.  ``acs1996`` is the single
most common style in the literature, so it is assigned a target share of the
total template slots (``--acs-share``, default 0.30).  With 3 slots per compound
that means exactly ``round(0.30 * n_compounds * 3)`` compounds carry acs1996 and
the remaining slots are spread over the other six templates.

Exactly which compounds carry acs1996, and every variant draw, is decided by a
seeded PRNG, so the slice is reproducible.  The full selection is written to
``split.json`` next to the manifest for auditing.

Default output for the shipped benchmark:
    309 compounds x 3 templates x 2 images = 1854 images

Usage
-----
  python tools/make_random_split.py                     # -> benchmark/random_seed42
  python tools/make_random_split.py --seed 7 --out benchmark/random_seed7
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "benchmark", "manifest_all.jsonl")
DEFAULT_OUT = os.path.join(ROOT, "benchmark", "random_seed42")


def jl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--templates-per-compound", type=int, default=3)
    ap.add_argument("--images-per-template", type=int, default=2)
    ap.add_argument("--acs-template", default="acs1996")
    ap.add_argument("--acs-share", type=float, default=0.30,
                    help="target share of ALL template slots taken by acs-template")
    ap.add_argument("--split-name", default="",
                    help="label written into each row (default: random_seed<seed>). "
                         "Kept independent of --out so the manifest content is a "
                         "pure function of the sampling parameters.")
    args = ap.parse_args()

    rows = jl(args.manifest)
    by_comp = collections.defaultdict(list)
    for r in rows:
        by_comp[r["compound_id"]].append(r)

    templates = sorted({r["template"] for r in rows})
    if args.acs_template not in templates:
        raise SystemExit(f"{args.acs_template!r} not among {templates}")
    others = [t for t in templates if t != args.acs_template]

    n_comp = len(by_comp)
    k = args.templates_per_compound
    n_slots = n_comp * k
    # how many compounds must carry the special template to hit the target share
    n_acs = int(round(args.acs_share * n_slots))
    n_acs = max(0, min(n_acs, n_comp))
    if k - 1 > len(others):
        raise SystemExit("templates-per-compound too large for the template pool")

    rng = random.Random(args.seed)
    comp_ids = sorted(by_comp)
    acs_bearers = set(rng.sample(comp_ids, n_acs))

    selected = {}          # compound_id -> [(template, [variant, ...]), ...]
    for cid in comp_ids:
        if cid in acs_bearers:
            picks = [args.acs_template] + rng.sample(others, k - 1)
            rng.shuffle(picks)
        else:
            picks = rng.sample(others, k)
        per_tmpl = []
        for t in picks:
            variants = sorted({r["variant"] for r in by_comp[cid] if r["template"] == t})
            take = min(args.images_per_template, len(variants))
            per_tmpl.append((t, rng.sample(variants, take)))
        selected[cid] = per_tmpl

    # ---- materialise the slice ---------------------------------------
    index = {(r["compound_id"], r["template"], r["variant"]): r for r in rows}
    split_name = args.split_name or f"random_seed{args.seed}"
    out_rows = []
    for cid in comp_ids:
        for tmpl, variants in selected[cid]:
            for v in variants:
                src = index[(cid, tmpl, v)]
                row = dict(src)
                row["split"] = split_name
                row["split_seed"] = args.seed
                out_rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.jsonl"), "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    tmpl_counts = collections.Counter(r["template"] for r in out_rows)
    var_counts = collections.Counter(r["variant"] for r in out_rows)
    sub_counts = collections.Counter(r["subset"] for r in out_rows)
    slot_counts = collections.Counter(t for cid in comp_ids
                                      for t, _ in selected[cid])
    split = {
        "split_name": split_name,
        "seed": args.seed,
        "source_manifest": os.path.relpath(args.manifest, ROOT).replace("\\", "/"),
        "templates_per_compound": k,
        "images_per_template": args.images_per_template,
        "acs_template": args.acs_template,
        "acs_share_target": args.acs_share,
        "n_compounds": n_comp,
        "n_template_slots": n_slots,
        "n_compounds_with_acs": len(acs_bearers),
        "acs_share_actual": len(acs_bearers) / n_slots,
        "n_images": len(out_rows),
        "images_per_template": dict(sorted(tmpl_counts.items())),
        "template_slots": dict(sorted(slot_counts.items())),
        "variant_counts": dict(sorted(var_counts.items())),
        "subset_counts": dict(sorted(sub_counts.items())),
        "selection": {cid: [[t, vs] for t, vs in sel] for cid, sel in selected.items()},
    }
    with open(os.path.join(args.out, "split.json"), "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2, ensure_ascii=False)

    print(f"compounds            : {n_comp}")
    print(f"template slots       : {n_slots}  ({k} per compound)")
    print(f"compounds with {args.acs_template:8s}: {len(acs_bearers)} "
          f"({len(acs_bearers)/n_slots:.1%} of slots, target {args.acs_share:.0%})")
    print(f"images               : {len(out_rows)}")
    print(f"  per template       : {dict(sorted(tmpl_counts.items()))}")
    print(f"  per variant        : {dict(sorted(var_counts.items()))}")
    print(f"  per subset         : {dict(sorted(sub_counts.items()))}")
    print(f"\nwrote {args.out}/manifest.jsonl, manifest.csv, split.json")


if __name__ == "__main__":
    main()
