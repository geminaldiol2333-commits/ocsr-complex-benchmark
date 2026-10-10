# -*- coding: utf-8 -*-
"""
Run MolSight (hustvl/MolSight) over a benchmark manifest and write one
prediction per image.

    python predict_molsight.py --manifest path/to/manifest.jsonl \
        --repo-dir /path/to/MolSight --out predictions/molsight.jsonl

MolSight is an EfficientViT-L1 image encoder plus a 512-dimensional, 6-layer
transformer decoder. It needs ``torch``, ``timm``, ``albumentations==2.0.5``,
``opencv-python``, ``rdkit``, ``datasets`` and ``SmilesPE``, plus a checkout of
the model code itself:

    git clone https://github.com/hustvl/MolSight

Point ``--repo-dir`` at that checkout (or set ``$MOLSIGHT_REPO``); the release
vocabulary is read from ``<repo-dir>/vocab/vocab_chars.json``.

Weights are taken from ``--checkpoint``, else ``$MOLSIGHT_CHECKPOINT``, else the
Hugging Face repo ``Robert-zwr/MolSight`` (downloaded on first use). Four
released checkpoints are exercised with two extra flags:

===============================  ==============================  ================
checkpoint                       predict field                   flags
===============================  ==============================  ================
``pubchem_uspto_smiles_edges_10``  SMILES & edge (base)           --
``pubchem_uspto_smiles_edges_30``  SMILES & edge (extra epochs)   --
``stereo_grpo_10_2``               SMILES only (LoRA, stereo)     ``--lora
                                                                  --formats char``
``pubchem_coords_10_2``            SMILES & edge & coords         ``--formats
                                                                  char,edges,coords``
===============================  ==============================  ================

Image paths in the manifest are recorded relative to the benchmark root, so
``--images-root`` defaults to whichever of the manifest's own directory or its
parent holds the ``organic/`` and ``coordination/`` trees — that covers both
``benchmark/manifest_all.jsonl`` and a slice such as
``benchmark/random_seed42/manifest.jsonl``. The run is resumable: rows already
present in the output file are skipped, so it can be stopped and restarted at
any time.

Two upstream quirks are handled here:

1. ``molsight/dataset.py`` does ``from .indigo import Indigo`` at import time,
   but the bundled Indigo build is Linux-only (``molsight/indigo/lib/Linux``).
   Indigo is used to *render* training images and sits nowhere on the inference
   path, so a stub ``molsight.indigo`` package is installed in ``sys.modules``
   before the import instead of patching upstream.
2. The repository's ``inference.py`` is single-image and reads its checkpoint
   from the working directory. The decoding hyper-parameters it sets on the
   command line are exposed here as configuration, so every released checkpoint
   is selected with ``--lora`` / ``--formats``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

HF_REPO = "Robert-zwr/MolSight"
HF_DEFAULT_FILE = "pubchem_uspto_smiles_edges_10.pth"


def install_indigo_stub():
    """Make ``from .indigo import Indigo`` work without the Linux Indigo build."""
    if "molsight.indigo" in sys.modules:
        return
    pkg = types.ModuleType("molsight.indigo")

    class _Unavailable:  # pragma: no cover - never reached on the inference path
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "molsight.indigo is a Linux-only build; it is only needed to "
                "render training data, not for inference.")

    pkg.Indigo = _Unavailable
    pkg.IndigoRenderer = _Unavailable
    renderer = types.ModuleType("molsight.indigo.renderer")
    renderer.IndigoRenderer = _Unavailable
    pkg.renderer = renderer
    sys.modules["molsight.indigo"] = pkg
    sys.modules["molsight.indigo.renderer"] = renderer


def resolve_checkpoint(explicit):
    if explicit:
        return explicit
    env = os.environ.get("MOLSIGHT_CHECKPOINT", "")
    if env:
        return env
    from huggingface_hub import hf_hub_download
    print(f"downloading {HF_DEFAULT_FILE} from {HF_REPO} ...", flush=True)
    return hf_hub_download(HF_REPO, HF_DEFAULT_FILE)


def resolve_images_root(manifest_path, explicit):
    if explicit:
        return explicit
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    parent = os.path.dirname(manifest_dir)
    for cand in (manifest_dir, parent):
        if any(os.path.isdir(os.path.join(cand, s)) for s in ("organic", "coordination")):
            return cand
    return parent


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


def model_args(opts):
    """The training-time hyper-parameters the released checkpoints were built with."""
    return argparse.Namespace(
        encoder="efficientvit",
        use_checkpoint=False,
        embed_dim=512,
        dec_n_layer=6,
        dec_n_head=8,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        lora=opts.lora,
        regression=False,
        input_size=512,
        formats=opts.formats.split(","),
        vocab_file=os.path.join(opts.repo_dir, "vocab", opts.vocab),
        resume=True,          # also means: do not fetch timm ImageNet weights
        max_len=opts.max_len,
        beam_size=1,
        n_samples=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="predictions/molsight.jsonl")
    ap.add_argument("--repo-dir", default=os.environ.get("MOLSIGHT_REPO", "MolSight"),
                    help="checkout of https://github.com/hustvl/MolSight "
                         "(or set $MOLSIGHT_REPO)")
    ap.add_argument("--checkpoint", default="",
                    help="weights file; defaults to $MOLSIGHT_CHECKPOINT, else the "
                         "Hugging Face repo " + HF_REPO)
    ap.add_argument("--vocab", default="vocab_chars.json")
    ap.add_argument("--formats", default="char,edges",
                    help="char / char,edges / char,edges,coords - must match the checkpoint")
    ap.add_argument("--lora", action="store_true",
                    help="build the decoder with LoRA layers (stereo / Markush checkpoints)")
    ap.add_argument("--images-root", default="",
                    help="root the manifest's image_path is relative to")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all")
    args = ap.parse_args()

    if not os.path.isdir(args.repo_dir):
        raise SystemExit(f"MolSight code not found at {args.repo_dir!r} - clone "
                         "https://github.com/hustvl/MolSight and pass --repo-dir "
                         "(or set $MOLSIGHT_REPO)")
    checkpoint = resolve_checkpoint(args.checkpoint)

    sys.path.insert(0, args.repo_dir)
    install_indigo_stub()

    import cv2
    import numpy as np
    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    from molsight.chemistry import _postprocess_smiles
    from molsight.dataset import get_transforms
    from molsight.model import MolsightModel
    from molsight.tokenizer import CharTokenizer

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

    m_args = model_args(args)
    print(f"checkpoint {os.path.basename(checkpoint)}  formats={m_args.formats} "
          f"lora={m_args.lora}", flush=True)

    tokenizer = CharTokenizer(m_args.vocab_file)
    model = MolsightModel(m_args, tokenizer)
    checkpoint_state = torch.load(checkpoint, map_location="cpu")
    state = checkpoint_state.get("model", checkpoint_state)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys    ({len(missing)}): {missing[:6]}", flush=True)
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:6]}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"device {device}", flush=True)

    transform = get_transforms(m_args, augment=False, rotate=False)

    def imread_rgb(path):
        # go through imdecode so non-ASCII paths work on Windows
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"cannot read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out, torch.no_grad():
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            paths = [os.path.join(root, *r["image_path"].split("/")) for r in chunk]
            raw = [""] * len(chunk)
            preds = [""] * len(chunk)
            try:
                batch = torch.stack([transform(image=imread_rgb(p))["image"]
                                     for p in paths], 0).to(device)
                kv_cache, hooks = model.install_kv_cache_hooks()
                try:
                    batch_preds, _ = model.generate(image=batch, kv_cache=kv_cache)
                finally:
                    for hook in hooks:
                        hook.remove()
                raw = list(batch_preds["smiles"])
                for j, text in enumerate(raw):
                    try:
                        smiles, _, _ = _postprocess_smiles(text)
                        smiles = Chem.CanonSmiles(smiles) if smiles else ""
                    except Exception:
                        smiles = text
                    preds[j] = smiles
            except Exception as exc:
                print(f"  batch {i//args.batch_size} ERROR {exc!r}"[:200], flush=True)
            for r, p, txt in zip(chunk, preds, raw):
                out.write(json.dumps({"image_id": r["image_id"],
                                      "image_path": r["image_path"],
                                      "prediction": p, "raw": txt},
                                     ensure_ascii=False) + "\n")
            out.flush()
            n += len(chunk)
            el = time.time() - t0
            print(f"[{n}/{len(todo)}] {el:.0f}s  {el/max(n,1):.3f}s/img", flush=True)

    print(f"done: {n} predictions in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
