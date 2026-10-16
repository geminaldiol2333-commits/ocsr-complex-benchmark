# -*- coding: utf-8 -*-
"""
Run DECIMER 2.7 over a benchmark manifest.

    python tools/predict_decimer.py --manifest path/to/manifest.jsonl

Needs ``decimer==2.7.1`` (TensorFlow 2.12–2.15; on Windows that build is
CPU-only). The first import downloads the standard and hand-drawn SavedModels
from Zenodo into ``$PYSTOW_HOME/DECIMER-V2`` (pystow's default if unset).
The scored string is ``DECIMER.predict_SMILES`` with the standard model
(SELFIES decoded by the package). The run is resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# tensorflow-intel's oneDNN path gets slower on every later SavedModel call.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def load_manifest(path, args):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.template and row["template"] not in args.template.split(","):
                continue
            if args.variant and row["variant"] not in args.variant.split(","):
                continue
            if args.subset != "all" and row["subset"] != args.subset:
                continue
            if args.ids and row["compound_id"] not in args.ids.split(","):
                continue
            rows.append(row)
    return rows


def image_path(images_root, rel):
    return os.path.join(images_root, *rel.split("/"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="predictions/decimer.jsonl")
    ap.add_argument("--images-root", default="",
                    help="root the manifest's image_path is relative to")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all", choices=["all", "organic", "coordination"])
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    if not args.images_root:
        args.images_root = os.path.dirname(os.path.dirname(os.path.abspath(args.manifest)))

    rows = load_manifest(args.manifest, args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    todo = [row for row in rows if row["image_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"manifest rows {len(rows)}, already done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return

    print("loading DECIMER", flush=True)
    from DECIMER import predict_SMILES

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for row in todo:
            path = image_path(args.images_root, row["image_path"])
            try:
                smiles = predict_SMILES(path) or ""
            except Exception as exc:
                print(f"  skip {row['image_id']} {exc!r}"[:300], flush=True)
                smiles = ""
            smiles = str(smiles).strip()
            out.write(json.dumps({
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "prediction": smiles,
            }, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            if n == 1 or n % 20 == 0 or n == len(todo):
                elapsed = time.time() - t0
                print(f"[{n}/{len(todo)}] {elapsed:.0f}s  {elapsed / n:.2f}s/img", flush=True)
    print(f"done: {n} predictions in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
