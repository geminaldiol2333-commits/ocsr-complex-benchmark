# -*- coding: utf-8 -*-
"""
Side-by-side comparison of OCSR models on the OCSR-complex benchmark.

    python tools/compare_models.py \
        --preds ocsrglyph=predictions/ocsrglyph.jsonl \
                tinychemvl=predictions/tinychemvl.jsonl \
        --out data/eval_results/comparison.md

By default the comparison is restricted to the images every model actually
predicted (the intersecting set), so models evaluated on different subsets are
still compared fairly; pass --no-common for per-model absolute numbers instead.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ocsr import score_predictions, load_rows  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "benchmark", "manifest_all.jsonl")
TAGS = ["has_bridged_ring", "has_spiro", "has_macrocycle", "has_cage",
        "is_stereo_dense", "is_large", "is_organometallic"]


def pct(x):
    return f"{100 * x:.1f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def metrics(rows):
    n = len(rows)
    if not n:
        return dict(n=0, valid=0, exact=0, connectivity=0, inchikey_full=0)
    return {
        "n": n,
        "valid": sum(r["valid"] for r in rows) / n,
        "exact": sum(r["exact"] for r in rows) / n,
        "connectivity": sum(r["connectivity"] for r in rows) / n,
        "inchikey_full": sum(r["inchikey_full"] for r in rows) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--preds", nargs="+", required=True,
                    help="name=path[,name=path...]")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval_results",
                                                  "comparison.md"))
    ap.add_argument("--no-common", action="store_true",
                    help="do not restrict to the intersection of image ids")
    args = ap.parse_args()

    specs = []
    for item in args.preds:
        for part in item.split(","):
            if part.strip():
                name, path = part.split("=", 1)
                specs.append((name.strip(), path.strip()))

    per_model = {}
    for name, path in specs:
        results, _, preds = score_predictions(args.manifest, path)
        scored = [r for r in results if "valid" in r]
        per_model[name] = {r["image_id"]: r for r in scored}
        print(f"{name}: {len(scored)} scored rows from {path}")

    common = None
    for name in per_model:
        ids = set(per_model[name])
        common = ids if common is None else (common & ids)
    print(f"common images across models: {len(common)}")

    def rows_for(name):
        if args.no_common:
            return list(per_model[name].values())
        return [per_model[name][i] for i in common]

    names = [n for n, _ in specs]
    lines = ["# OCSR-complex — model comparison", ""]
    if not args.no_common:
        lines.append(f"Restricted to the **{len(common)} images** predicted by every "
                     f"model. Metrics are computed over that intersection.")
    else:
        lines.append("Per-model absolute numbers (models may cover different subsets).")
    lines.append("")

    # ---- headline ----------------------------------------------------
    rows = []
    for name in names:
        m = metrics(rows_for(name))
        rows.append([name, m["n"], pct(m["valid"]), pct(m["exact"]),
                     pct(m["connectivity"]), pct(m["inchikey_full"])])
    lines += ["## Overall", "",
              table(["model", "n", "valid %", "exact %", "connectivity %",
                     "InChIKey %"], rows), ""]

    # ---- breakdowns --------------------------------------------------
    for field, title in (("template", "By drawing template"),
                         ("variant", "By geometric variant"),
                         ("subset", "By subset")):
        keys = sorted({r[field] for r in rows_for(names[0])})
        rows = []
        for k in keys:
            row = [k]
            for name in names:
                sub = [r for r in rows_for(name) if r[field] == k]
                m = metrics(sub)
                row.append(f'{pct(m["exact"])} (n={m["n"]})')
            rows.append(row)
        lines += [f"## {title}", "", table(["value"] + names, rows), ""]

    # ---- difficulty tags --------------------------------------------
    tag_rows = []
    for tag in TAGS:
        row = [f"`{tag}`"]
        for name in names:
            sub = [r for r in rows_for(name) if r.get(tag)]
            m = metrics(sub)
            row.append(f'{pct(m["exact"])} (n={m["n"]})')
        tag_rows.append(row)
    row = ["`bridged+stereo_dense`"]
    for name in names:
        sub = [r for r in rows_for(name)
               if r.get("has_bridged_ring") and r.get("is_stereo_dense")]
        m = metrics(sub)
        row.append(f'{pct(m["exact"])} (n={m["n"]})')
    tag_rows.append(row)
    lines += ["## By difficulty tag (exact-match %)", "",
              table(["tag"] + names, tag_rows), ""]

    # ---- validity (parse rate) --------------------------------------
    tag_rows = []
    for tag in TAGS:
        row = [f"`{tag}`"]
        for name in names:
            sub = [r for r in rows_for(name) if r.get(tag)]
            m = metrics(sub)
            row.append(f'{pct(m["valid"])} (n={m["n"]})')
        tag_rows.append(row)
    lines += ["## By difficulty tag (parseable %)", "",
              table(["tag"] + names, tag_rows), ""]

    # ---- hardest / easiest compounds --------------------------------
    for name in names:
        by_comp = collections.defaultdict(list)
        for r in rows_for(name):
            by_comp[r["compound_id"]].append(r)
        stats = [(c, sum(x["exact"] for x in v) / len(v), len(v))
                 for c, v in by_comp.items()]
        stats.sort(key=lambda t: (t[1], -t[2]))
        hard = [f"`{c}`" for c, e, n in stats[:10] if n >= 5]
        easy = [f"`{c}`" for c, e, n in reversed(stats[-10:]) if n >= 5]
        lines += [f"### {name}", "",
                  f"- **Hardest** (exact-rate 0): {', '.join(hard) or '—'}",
                  f"- **Solved** (exact-rate 1.0): {', '.join(easy) or '—'}", ""]

    # ---- pairwise agreement -----------------------------------------
    if len(names) >= 2:
        a, b = names[0], names[1]
        agree = both = only_a = only_b = neither = 0
        for i in common:
            ea, eb = per_model[a][i]["exact"], per_model[b][i]["exact"]
            if ea and eb:
                both += 1
            elif ea:
                only_a += 1
            elif eb:
                only_b += 1
            else:
                neither += 1
        tot = max(len(common), 1)
        lines += ["## Pairwise agreement (exact match)", "",
                  table(["outcome", "count", "%"],
                        [["both correct", both, pct(both / tot)],
                         [f"only {a}", only_a, pct(only_a / tot)],
                         [f"only {b}", only_b, pct(only_b / tot)],
                         ["neither", neither, pct(neither / tot)]]), ""]

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print("\n" + report)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
