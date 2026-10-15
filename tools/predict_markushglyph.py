# -*- coding: utf-8 -*-
"""
Run MarkushGlyph over a benchmark manifest.

    python tools/predict_markushglyph.py --manifest path/to/manifest.jsonl \
        --checkpoint path/to/MarkushGlyph \
        --base-model path/to/Qwen3.5-2B-Base

The checkpoint is the EdisonScientific/MarkushGlyph LoRA on Qwen3.5-2B-Base.
Point ``$GLYPH_REPO`` at a glyph checkout. Decoding is greedy. The prompt asks
for a base structure with an empty Markush extension. The scored string is
the CXSMILES core before any ``|`` extension.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

GLYPH_REPO = os.environ.get("GLYPH_REPO", "")
if GLYPH_REPO and GLYPH_REPO not in sys.path:
    sys.path.insert(0, GLYPH_REPO)

DEFAULT_CHECKPOINT = os.environ.get("MARKUSHGLYPH_CHECKPOINT", "")
DEFAULT_BASE = os.environ.get("MARKUSH_BASE_MODEL", "")


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


def core_smiles(prediction):
    from glyph.markush.data.pipeline import split_cxsmiles

    text = (prediction.cxsmiles or prediction.cxsmiles_opt or "").strip()
    if not text:
        return ""
    core, _extension = split_cxsmiles(text)
    return (core or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="predictions/markushglyph.jsonl")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--images-root", default="",
                    help="root the manifest's image_path is relative to")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all", choices=["all", "organic", "coordination"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not GLYPH_REPO:
        ap.error("set GLYPH_REPO to a local glyph checkout")
    if not args.checkpoint:
        ap.error("pass --checkpoint or set MARKUSHGLYPH_CHECKPOINT")
    if not os.path.isdir(args.checkpoint):
        ap.error(f"checkpoint dir not found: {args.checkpoint}")
    if not args.base_model:
        ap.error("pass --base-model or set MARKUSH_BASE_MODEL")
    if not os.path.isdir(args.base_model):
        ap.error(f"base model dir not found: {args.base_model}")
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

    import torch
    from glyph.markush.inference import MarkushPredictor, OCSR_NO_EXTENSION_PROMPT

    print("torch", torch.__version__, "device", args.device,
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    print("checkpoint", args.checkpoint, flush=True)
    print("base", args.base_model, flush=True)
    predictor = MarkushPredictor(
        args.checkpoint,
        base_model=args.base_model,
        device=args.device,
        dtype="bfloat16",
        max_new_tokens=args.max_new_tokens,
        prompt=OCSR_NO_EXTENSION_PROMPT,
    )
    predictor.load()

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            paths = [image_path(args.images_root, row["image_path"]) for row in chunk]
            try:
                preds = predictor.predict_many(paths, batch_size=args.batch_size)
            except Exception as exc:
                print(f"  batch {start // args.batch_size} ERROR {exc!r}"[:400], flush=True)
                preds = []
                for one in paths:
                    try:
                        preds.append(predictor.predict_many([one], batch_size=1)[0])
                    except Exception as one_exc:
                        print(f"    skip {one} {one_exc!r}"[:300], flush=True)
                        preds.append(None)
            for row, pred in zip(chunk, preds):
                smiles = core_smiles(pred) if pred is not None else ""
                raw = pred.raw if pred is not None else ""
                out.write(json.dumps({
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "prediction": smiles,
                    "raw": raw,
                }, ensure_ascii=False) + "\n")
            out.flush()
            n += len(chunk)
            elapsed = time.time() - t0
            print(f"[{n}/{len(todo)}] {elapsed:.0f}s  {elapsed / max(n, 1):.2f}s/img", flush=True)
    print(f"done: {n} predictions in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
