# -*- coding: utf-8 -*-
"""
Run OCSRGlyph (EdisonScientific/glyph) over a benchmark manifest and write one
prediction per image.

OCSRGlyph is a Swin-B/384 image encoder plus a 6-layer transformer decoder.
It needs the ``glyph`` package (``pip install -e ".[ocsr]"`` from
https://github.com/EdisonScientific/glyph), preferably inside its own virtualenv.

    python predict_ocsrglyph.py --manifest path/to/manifest.jsonl --out preds.jsonl

Weights are taken from ``--checkpoint``, else ``$OCSRGLYPH_CHECKPOINT``, else the
Hugging Face repo ``EdisonScientific/OCSRGlyph`` (downloaded on first use).

Image paths in the manifest are resolved relative to ``--images-root``, which
defaults to the directory containing the manifest's parent.

The run is resumable: rows already present in the output file are skipped, so it
can be stopped and restarted at any time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

DEFAULT_REPO = "EdisonScientific/OCSRGlyph"


def load_manifest(path, args):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.template and r["template"] not in args.template.split(","):
                continue
            if args.variant and r["variant"] not in args.variant.split(","):
                continue
            if args.subset != "all" and r.get("subset") != args.subset:
                continue
            rows.append(r)
    return rows


def resolve_images_root(manifest_path, explicit):
    if explicit:
        return explicit
    # benchmark/manifest_all.jsonl -> benchmark/
    return os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="predictions/ocsrglyph.jsonl")
    ap.add_argument("--checkpoint", default=os.environ.get("OCSRGLYPH_CHECKPOINT",
                                                           DEFAULT_REPO))
    ap.add_argument("--images-root", default="",
                    help="root the manifest's image_path is relative to")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--no-postprocess", action="store_true")
    args = ap.parse_args()

    root = resolve_images_root(args.manifest, args.images_root)
    rows = load_manifest(args.manifest, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    todo = [r for r in rows if r["image_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"manifest rows {len(rows)}, already done {len(done)}, to run {len(todo)}",
          flush=True)
    if not todo:
        print("nothing to do")
        return

    import torch
    from glyph.ocsr.predict import OCSRPredictor

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    predictor = OCSRPredictor(args.checkpoint)

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            paths = [os.path.join(root, *r["image_path"].split("/")) for r in chunk]
            try:
                preds = predictor.predict_batch(
                    paths, batch_size=args.batch_size,
                    postprocess=not args.no_postprocess)
            except Exception as exc:              # keep the run alive
                preds = [""] * len(chunk)
                print(f"  batch {i//args.batch_size} ERROR {exc!r}"[:200], flush=True)
            for r, p in zip(chunk, preds):
                p = (p or "").strip()
                out.write(json.dumps({"image_id": r["image_id"],
                                      "image_path": r["image_path"],
                                      "prediction": p, "raw": p},
                                     ensure_ascii=False) + "\n")
            out.flush()
            n += len(chunk)
            el = time.time() - t0
            print(f"[{n}/{len(todo)}] {el:.0f}s  {el/max(n,1):.2f}s/img", flush=True)

    print(f"done: {n} predictions in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
