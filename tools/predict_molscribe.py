# -*- coding: utf-8 -*-
"""
Run MolScribe over a benchmark manifest.

    python tools/predict_molscribe.py --manifest path/to/manifest.jsonl \
        --checkpoint path/to/swin_base_char_aux_1m680k.pth

Checkpoint: yujieq/MolScribe ``swin_base_char_aux_1m680k.pth`` (Swin-B encoder,
graph decoder). Point ``--checkpoint`` or ``$MOLSCRIBE_CHECKPOINT`` at that file.
The scored string is the official graph-to-SMILES output.
The run is resumable: rows already present in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

DEFAULT_CHECKPOINT = os.environ.get("MOLSCRIBE_CHECKPOINT", "")


def patch_import_compat():
    """Keep MolScribe 1.1.1 importable on current timm."""
    import timm.models.helpers as helpers
    import timm.models.vision_transformer as vit

    if not hasattr(helpers, "overlay_external_default_cfg"):
        def overlay_external_default_cfg(default_cfg, kwargs):
            return default_cfg
        helpers.overlay_external_default_cfg = overlay_external_default_cfg
    if not hasattr(vit, "_init_vit_weights"):
        def _init_vit_weights(module, name="", head_bias=0.0, jax_impl=False):
            return None
        vit._init_vit_weights = _init_vit_weights
    if not hasattr(vit, "checkpoint_filter_fn"):
        def checkpoint_filter_fn(state_dict, model):
            return state_dict
        vit.checkpoint_filter_fn = checkpoint_filter_fn


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


def read_rgb(path):
    import cv2
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="predictions/molscribe.jsonl")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--images-root", default="",
                    help="root the manifest's image_path is relative to")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all", choices=["all", "organic", "coordination"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.checkpoint:
        ap.error("pass --checkpoint or set MOLSCRIBE_CHECKPOINT")
    if not os.path.isfile(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")
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

    patch_import_compat()
    import torch
    from molscribe import MolScribe

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("torch", torch.__version__, "device", device,
          torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    print("checkpoint", args.checkpoint, flush=True)
    model = MolScribe(args.checkpoint, device=torch.device(device))

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            paths = [image_path(args.images_root, row["image_path"]) for row in chunk]
            try:
                images = [read_rgb(path) for path in paths]
                preds = model.predict_images(images, batch_size=args.batch_size)
            except Exception as exc:
                print(f"  batch {start // args.batch_size} ERROR {exc!r}"[:400], flush=True)
                preds = []
                for one in paths:
                    try:
                        preds.append(model.predict_images([read_rgb(one)], batch_size=1)[0])
                    except Exception as one_exc:
                        print(f"    skip {one} {one_exc!r}"[:300], flush=True)
                        preds.append({"smiles": ""})
            for row, pred in zip(chunk, preds):
                smiles = (pred.get("smiles") or "").strip()
                if smiles == "<invalid>":
                    smiles = ""
                out.write(json.dumps({
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "prediction": smiles,
                }, ensure_ascii=False) + "\n")
            out.flush()
            n += len(chunk)
            elapsed = time.time() - t0
            print(f"[{n}/{len(todo)}] {elapsed:.0f}s  {elapsed / max(n, 1):.2f}s/img", flush=True)
    print(f"done: {n} predictions in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
