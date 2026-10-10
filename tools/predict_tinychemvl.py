# -*- coding: utf-8 -*-
"""
Run TinyChemVL (InternViT-6B + Qwen2.5-3B) over a benchmark manifest and write
one prediction per image.

    python predict_tinychemvl.py --manifest path/to/manifest.jsonl \
        --model-dir /path/to/TinyChemVL --out predictions/tinychemvl.jsonl

Requires ``torch``, ``transformers``, ``torchvision`` and ``pillow``.

Two quirks of this checkpoint are handled explicitly — both cost hours if
missed, and the published example prompts do not work as-is:

1. The prompt **must** contain the ``<image>`` placeholder. Without it the image
   is never spliced into the token stream, so the model answers from text alone
   (or crashes with a shape mismatch).
2. ``modeling_internvl_chat.chat()`` divides ``num_image_token`` by 16 *in place*,
   so only the very first call after loading succeeds; ``batch_chat`` does not do
   that division. Either way the correct value is
   ``config.num_image_token // 16`` tokens per 448 px tile, and it is set
   explicitly before every batch here.

The run is resumable: rows already present in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import List, Tuple

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MIN_DYNAMIC_PATCH = 1
MAX_DYNAMIC_PATCH = 12

PROMPT = ("<image>\nIdentify the molecule in the image and output its SMILES "
          "representation. Answer with the SMILES string only, no explanation.")

_SMILES_RUN = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$%\\/\.\*:]{5,}")


# --------------------------------------------------------------------------
# InternVL dynamic-resolution preprocessing (448 px tiles + thumbnail)
# --------------------------------------------------------------------------
def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def load_image_and_preprocess(image_path, max_num=MAX_DYNAMIC_PATCH,
                              use_thumbnail=True) -> Tuple["object", int]:
    import torch
    import torchvision.transforms as T
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set()
    for n in range(MIN_DYNAMIC_PATCH, max_num + 1):
        for i in range(1, n + 1):
            if i * (n // i) == n:
                target_ratios.add((i, n // i))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target = _closest_aspect_ratio(aspect_ratio, target_ratios,
                                   orig_width, orig_height, INPUT_SIZE)

    target_width = INPUT_SIZE * target[0]
    target_height = INPUT_SIZE * target[1]
    blocks = target[0] * target[1]
    resized = image.resize((target_width, target_height))

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN,
                                                     std=IMAGENET_STD)])
    tiles = []
    for i in range(blocks):
        col, row = i % (target_width // INPUT_SIZE), i // (target_width // INPUT_SIZE)
        box = (col * INPUT_SIZE, row * INPUT_SIZE,
               col * INPUT_SIZE + INPUT_SIZE, row * INPUT_SIZE + INPUT_SIZE)
        tiles.append(transform(resized.crop(box)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(transform(image.resize((INPUT_SIZE, INPUT_SIZE))))
    pixel_values = torch.stack(tiles)
    return pixel_values, pixel_values.size(0)


# --------------------------------------------------------------------------
# SMILES extraction from a chatty response
# --------------------------------------------------------------------------
def extract_smiles(text):
    if not text:
        return ""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    cands = [f.strip() for f in re.findall(r"```(?:smiles)?\s*([^`]+)```", text, flags=re.I)]
    cands += [m.group(0).strip() for m in _SMILES_RUN.finditer(text)]
    cands.append(text.strip())

    for c in cands:
        c = c.strip().strip(".,;:").replace(" ", "")
        if not c:
            continue
        for cut in range(len(c), 4, -1):
            head = c[:cut]
            if head.count("(") != head.count(")") or head.count("[") != head.count("]"):
                continue
            if Chem.MolFromSmiles(head) is not None:
                return head
    return ""


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model-dir", default=os.environ.get("TINYCHEMVL_MODEL_DIR", ""),
                    help="local path to the TinyChemVL weights (or set "
                         "$TINYCHEMVL_MODEL_DIR)")
    ap.add_argument("--out", default="predictions/tinychemvl.jsonl")
    ap.add_argument("--images-root", default="",
                    help="root the manifest's image_path is relative to")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all")
    args = ap.parse_args()

    if not args.model_dir:
        raise SystemExit("pass --model-dir or set $TINYCHEMVL_MODEL_DIR")

    import torch
    from transformers import AutoModel, AutoTokenizer

    root = args.images_root or os.path.dirname(os.path.dirname(os.path.abspath(args.manifest)))
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

    print(f"loading {args.model_dir} ...", flush=True)
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    tokens_per_tile = model.num_image_token // 16      # quirk 2, see docstring
    print(f"tokens per tile: {tokens_per_tile}", flush=True)

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            paths = [os.path.join(root, *r["image_path"].split("/")) for r in chunk]
            raw = [""] * len(chunk)
            try:
                pvs, npl = [], []
                for p in paths:
                    px, npt = load_image_and_preprocess(p)
                    pvs.append(px)
                    npl.append(npt)
                pv = torch.cat(pvs, 0).to(device=model.device, dtype=model.dtype)
                model.num_image_token = tokens_per_tile
                raw = model.batch_chat(
                    tokenizer=tokenizer, pixel_values=pv,
                    questions=[PROMPT] * len(chunk),
                    generation_config={"max_new_tokens": args.max_new_tokens,
                                       "do_sample": False},
                    num_patches_list=npl)
            except Exception as exc:
                print(f"  batch {i//args.batch_size} ERROR {exc!r}"[:200], flush=True)
            for r, txt in zip(chunk, raw):
                txt = (txt or "").strip()
                out.write(json.dumps({"image_id": r["image_id"],
                                      "image_path": r["image_path"],
                                      "prediction": extract_smiles(txt),
                                      "raw": txt}, ensure_ascii=False) + "\n")
            out.flush()
            n += len(chunk)
            el = time.time() - t0
            print(f"[{n}/{len(todo)}] {el:.0f}s  {el/max(n,1):.2f}s/img", flush=True)

    print(f"done: {n} predictions in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
