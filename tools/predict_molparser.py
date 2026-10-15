# -*- coding: utf-8 -*-
"""
Run MolParser-Mobile-V2 over a benchmark manifest.

    python tools/predict_molparser.py --manifest path/to/manifest.jsonl \
        --model path/to/MolParser-Mobile-V2

The checkpoint is the 10.00M-parameter UniParser/MolParser-Mobile-V2 release
(https://huggingface.co/UniParser/MolParser-Mobile-V2). Point ``--model`` or
``$MOLPARSER_MODEL`` at a local copy of that repository.
Inference follows the model card: FP16, greedy decoding, max length 384.
The scored SMILES is the trunk before an E-SMILES ``<sep>`` or ``|`` suffix.
Raw decoder text is kept in the ``raw`` field.

The run is resumable: rows already present in the output file are skipped.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time

DEFAULT_MODEL = os.environ.get("MOLPARSER_MODEL", "")


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


def load_module(model_dir, name, filename):
    path = os.path.join(model_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trunk(text):
    """SMILES body of an E-SMILES 2.0 string, before <sep> or a pipe suffix."""
    text = (text or "").strip()
    for marker in ("<sep>", "|"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    parts = text.split()
    return parts[0] if parts else ""


def patch_init_context(model_cls):
    """transformers 4.57 initialises on the meta device; timm ViT cannot.

    The checkpoint's own override targets the transformers 5.x signature.
    When the installed library still uses the 4.x signature, allocate real
    tensors and skip random init instead.
    """
    from transformers.modeling_utils import no_init_weights
    from transformers.models.vision_encoder_decoder import (
        modeling_vision_encoder_decoder as ved,
    )

    parent = ved.VisionEncoderDecoderModel.get_init_context
    names = list(inspect.signature(parent).parameters)
    if "dtype" in names:
        return

    @classmethod
    def get_init_context(cls, is_quantized, _is_ds_init_called):
        return [no_init_weights()]

    model_cls.get_init_context = get_init_context


def load_model(model_dir, device, dtype):
    import torch

    image_mod = load_module(model_dir, "image_processing_molparser_mobile",
                            "image_processing_molparser_mobile.py")
    tok_mod = load_module(model_dir, "tokenization_molparser_mobile",
                          "tokenization_molparser_mobile.py")
    model_mod = load_module(model_dir, "modeling_molparser_mobile",
                            "modeling_molparser_mobile.py")
    processor = image_mod.MolParserImageProcessor.from_pretrained(
        model_dir, local_files_only=True)
    tokenizer = tok_mod.MolParserTokenizer.from_pretrained(
        model_dir, local_files_only=True)
    patch_init_context(model_mod.MolParserVisionEncoderDecoderModel)
    config = model_mod.MolParserVisionEncoderDecoderConfig.from_pretrained(
        model_dir, local_files_only=True)
    model_mod.assert_mobile_config(config)
    if getattr(config, "encoder", None) is not None:
        config.encoder.timm_pretrained = False
    model = model_mod.MolParserVisionEncoderDecoderModel.from_pretrained(
        model_dir, config=config, local_files_only=True)
    model = model.to(device=device, dtype=dtype).eval()
    return processor, tokenizer, model


def predict_batch(processor, tokenizer, model, paths, device, dtype, max_length):
    import torch
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in paths]
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.inference_mode():
        output_ids = model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=1,
            do_sample=False,
        )
    return [text.strip() for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="predictions/molparser_mobile_v2.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--images-root", default="",
                    help="root the manifest's image_path is relative to")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all", choices=["all", "organic", "coordination"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model or set MOLPARSER_MODEL to a local MolParser-Mobile-V2 checkout")
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

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("torch", torch.__version__, "device", device,
          torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    print("model", args.model, flush=True)
    processor, tokenizer, model = load_model(args.model, device, dtype)

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            paths = [image_path(args.images_root, row["image_path"]) for row in chunk]
            try:
                raws = predict_batch(processor, tokenizer, model, paths, device, dtype, args.max_length)
            except Exception as exc:
                print(f"  batch {start // args.batch_size} ERROR {exc!r}"[:300], flush=True)
                raws = []
                for one in paths:
                    try:
                        raws.append(predict_batch(
                            processor, tokenizer, model, [one], device, dtype, args.max_length)[0])
                    except Exception as one_exc:
                        print(f"    skip {one} {one_exc!r}"[:300], flush=True)
                        raws.append("")
            for row, raw in zip(chunk, raws):
                pred = trunk(raw)
                out.write(json.dumps({
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "prediction": pred,
                    "raw": raw,
                }, ensure_ascii=False) + "\n")
            out.flush()
            n += len(chunk)
            elapsed = time.time() - t0
            print(f"[{n}/{len(todo)}] {elapsed:.0f}s  {elapsed / max(n, 1):.2f}s/img", flush=True)
    print(f"done: {n} predictions in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
