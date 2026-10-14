# -*- coding: utf-8 -*-
"""
Run COMO (Keylab/COMO) over a benchmark manifest and write one prediction
per image.

    python predict_como.py --manifest path/to/manifest.jsonl \
        --checkpoint path/to/tanimoto/best.pth --out predictions/como.jsonl

COMO is a Swin-B/384 image encoder plus a 6-layer decoder (about 94M
parameters). Install the package with ``pip install como-ocsr``. Weights are
taken from ``--checkpoint`` or ``$COMO_CHECKPOINT``. The numbers on this page
use the Tanimoto-reward checkpoint ``models/tanimoto/best.pth`` from
https://huggingface.co/Keylab/COMO (weights are CC BY-NC 4.0).

SMILES are COMO's postprocess output, the mode the COMO paper reports. When
that restoration does not yield a canonical SMILES, the row keeps the
decoder-mode canonical SMILES instead. Both modes drop cis-trans markers.

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
    return os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))


def smiles_from_result(result):
    """Postprocess SMILES, else a canonical decoder string."""
    import como
    from como.chemistry import canonicalize_smiles

    post = como._result_to_smiles(result, "postprocess")
    if post:
        return post, "postprocess"
    raw = result.get("decode_smiles") if isinstance(result, dict) else ""
    can, ok = canonicalize_smiles(raw or "", ignore_cistrans=True)
    if ok and can:
        return can, "decoder"
    return "", "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="predictions/como.jsonl")
    ap.add_argument("--checkpoint", default=os.environ.get("COMO_CHECKPOINT", ""))
    ap.add_argument("--images-root", default="",
                    help="root the manifest's image_path is relative to")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.checkpoint:
        ap.error("pass --checkpoint or set $COMO_CHECKPOINT")

    root = resolve_images_root(args.manifest, args.images_root)
    rows = load_manifest(args.manifest, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

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
        return 0

    import torch
    import como

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
          flush=True)
    print("checkpoint", args.checkpoint, flush=True)
    model = como.load_model(args.checkpoint, device=args.device)

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            paths = [os.path.join(root, *r["image_path"].split("/")) for r in chunk]
            try:
                raw = model.predict_batch(
                    paths, beam_size=args.beam_size, max_len=args.max_len,
                    smiles_mode=None)
            except Exception as exc:
                print(f"  batch {i // args.batch_size} ERROR {exc!r}"[:300], flush=True)
                raw = []
                for one_path in paths:
                    try:
                        raw.append(model.predict(
                            one_path, beam_size=args.beam_size,
                            max_len=args.max_len, smiles_mode=None))
                    except Exception as one_exc:
                        print(f"    skip {one_path} {one_exc!r}"[:300], flush=True)
                        raw.append(None)
            for r, result in zip(chunk, raw):
                if result is None:
                    pred, source = "", "error"
                else:
                    pred, source = smiles_from_result(result)
                out.write(json.dumps({
                    "image_id": r["image_id"],
                    "image_path": r["image_path"],
                    "prediction": pred,
                    "raw": pred,
                    "smiles_source": source,
                }, ensure_ascii=False) + "\n")
            out.flush()
            n += len(chunk)
            el = time.time() - t0
            print(f"[{n}/{len(todo)}] {el:.0f}s  {el / max(n, 1):.2f}s/img", flush=True)

    print(f"done: {n} predictions in {time.time() - t0:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
