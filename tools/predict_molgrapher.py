# -*- coding: utf-8 -*-
"""
Run MolGrapher over a benchmark manifest, or convert an existing run.

MolGrapher (ICCV 2023, https://github.com/DS4SD/MolGrapher) is driven through
its own command line entry point instead of being imported here, because it
pins an old PyTorch and needs PaddleOCR: run this script with the interpreter
of the MolGrapher environment, or point ``--python`` at it.

    MOLGRAPHER_REPO=~/src/MolGrapher \
    MOLGRAPHER_CHECKPOINTS=~/models/MolGrapher/models \
    python tools/predict_molgrapher.py

The scored string is the SMILES MolGrapher writes for the graph it predicts
from the ``gc_stereo_model`` node classifier with ``--assign-stereo``. The run
is resumable: image ids already present in ``--out`` are skipped. Each chunk
gets its own MolGrapher output folder, so a chunk that dies does not lose the
chunks before it.

To score a run that already happened (MolGrapher writes one ``.mol`` per image
plus a ``smiles.jsonl`` in its save folder), convert it without inference:

    python tools/predict_molgrapher.py --mol-dir path/to/save_folder --out preds.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "benchmark", "random_seed42", "manifest.jsonl")
DEFAULT_REPO = os.environ.get("MOLGRAPHER_REPO", "")
DEFAULT_CHECKPOINTS = os.environ.get("MOLGRAPHER_CHECKPOINTS", "")


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


def done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    return done


def read_annotations(save_folder):
    """MolGrapher output -> {original image path: SMILES}.

    ``file-info.filename`` points at the preprocessed copy MolGrapher made next
    to the input image, so the ``_preprocessed`` suffix is stripped back off to
    recover the path we asked about.
    """
    annotations = {}
    path = os.path.join(save_folder, "smiles.jsonl")
    if not os.path.exists(path):
        return annotations
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            filename = (row.get("file-info") or {}).get("filename", "")
            key = filename.replace("_preprocessed", "")
            annotations[key] = (row.get("smi") or "").strip()
    return annotations


def convert(save_folder, chunk_rows, out):
    """Append one prediction row per requested image to ``out``."""
    annotations = read_annotations(save_folder)
    written = 0
    with open(out, "a", encoding="utf-8") as fh:
        for row in chunk_rows:
            prediction = annotations.get(row["_path"], "")
            fh.write(json.dumps({
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "prediction": prediction,
            }, ensure_ascii=False) + "\n")
            written += 1
    return written


def checkpoints_ready(checkpoints, variant):
    needed = [
        os.path.join(checkpoints, "keypoint_detector", "kd_model.ckpt"),
        os.path.join(checkpoints, "graph_classifier", f"{variant}.ckpt"),
    ]
    missing = [p for p in needed if not os.path.isfile(p)]
    return needed, missing


def run_chunk(repo, python, chunk_path, save_folder, args, log):
    entry = os.path.join(repo, "molgrapher", "scripts", "annotate", "predict_molgrapher.py")
    cmd = [
        python, entry,
        "--input-images-paths", chunk_path,
        "--save-mol-folder", save_folder + os.sep,
        "--node-classifier-variant", args.node_classifier_variant,
        "--num-threads-pytorch", str(args.num_threads_pytorch),
    ]
    if args.assign_stereo:
        cmd.append("--assign-stereo")
    # MolGrapher writes tmp.mol / tmp2.mol into the working directory while it
    # assigns stereo centres. Two runs sharing a directory corrupt each other,
    # so every run gets its own. It must not live inside the save folder, which
    # MolGrapher deletes on start.
    cwd = os.path.join(os.path.dirname(os.path.abspath(chunk_path)), "cwd")
    os.makedirs(cwd, exist_ok=True)
    print("  $", " ".join(cmd), flush=True)
    with open(log, "a", encoding="utf-8") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=os.path.join(ROOT, "predictions", "molgrapher.jsonl"))
    ap.add_argument("--images-root", default=os.path.join(ROOT, "benchmark"))
    ap.add_argument("--work-dir", default=os.path.join(ROOT, "predictions", "molgrapher"))
    ap.add_argument("--repo", default=DEFAULT_REPO, help="MolGrapher checkout ($MOLGRAPHER_REPO)")
    ap.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS,
                    help="folder holding graph_classifier/ and keypoint_detector/ ($MOLGRAPHER_CHECKPOINTS)")
    ap.add_argument("--python", default=sys.executable, help="interpreter of the MolGrapher environment")
    ap.add_argument("--mol-dir", default="", help="convert an existing MolGrapher save folder instead of running")
    ap.add_argument("--variant", default="", help="manifest variant filter (rz_a,rz_b,tilt_x,...)")
    ap.add_argument("--node-classifier-variant", default="gc_stereo_model",
                    choices=["gc_stereo_model", "gc_no_stereo_model", "gc_gcn_model"])
    ap.add_argument("--assign-stereo", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num-threads-pytorch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--shard", default="",
                    help="i/n: take every n-th manifest row starting at i, for parallel runs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--subset", default="all", choices=["all", "organic", "coordination"])
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    rows = load_manifest(args.manifest, args)
    shard_index = shard_count = 0
    if args.shard:
        index, count = args.shard.split("/")
        shard_index, shard_count = int(index), int(count)
        if not 0 <= shard_index < shard_count:
            sys.exit(f"bad --shard {args.shard}")
        rows = rows[shard_index::shard_count]
    for row in rows:
        row["_path"] = image_path(args.images_root, row["image_path"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.mol_dir:
        print(f"converting {args.mol_dir} -> {args.out}", flush=True)
        n = convert(args.mol_dir, rows, args.out)
        print(f"done: {n} rows")
        return

    if not args.repo or not os.path.isdir(args.repo):
        sys.exit("MolGrapher checkout not found; pass --repo or set MOLGRAPHER_REPO")
    needed, missing = checkpoints_ready(args.checkpoints, args.node_classifier_variant)
    if missing:
        sys.exit("missing checkpoints: " + ", ".join(missing))

    done = done_ids(args.out)
    todo = [row for row in rows if row["image_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"manifest rows {len(rows)}, already done {len(done)}, to run {len(todo)}"
          + (f" (shard {shard_index}/{shard_count})" if shard_count else ""), flush=True)
    if not todo:
        print("nothing to do")
        return

    os.makedirs(args.work_dir, exist_ok=True)
    if shard_count:
        args.work_dir = os.path.join(args.work_dir, f"shard{shard_index}")
        os.makedirs(args.work_dir, exist_ok=True)
    log = os.path.join(args.work_dir, f"run-{args.node_classifier_variant}.log")
    print(f"MolGrapher log: {log}", flush=True)

    t0 = time.time()
    n = 0
    missing = 0
    for start in range(0, len(todo), args.chunk_size):
        chunk = todo[start:start + args.chunk_size]
        index = start // args.chunk_size
        chunk_path = os.path.join(args.work_dir, f"input-chunk{index:03d}.jsonl")
        save_folder = os.path.join(args.work_dir, f"out-{args.node_classifier_variant}-chunk{index:03d}")
        with open(chunk_path, "w", encoding="utf-8") as fh:
            for row in chunk:
                fh.write(json.dumps({"path": row["_path"], "image_id": row["image_id"]}) + "\n")
        code = run_chunk(args.repo, args.python, chunk_path, save_folder, args, log)
        if code != 0:
            # Leave the rows out of --out so a re-run picks them up again.
            print(f"  chunk {index} exited with {code}; {len(chunk)} rows left for a re-run",
                  flush=True)
            missing += len(chunk)
            continue
        n += convert(save_folder, chunk, args.out)
        elapsed = time.time() - t0
        print(f"[{n}/{len(todo)}] {elapsed:.0f}s  {elapsed / max(n, 1):.2f}s/img", flush=True)
    print(f"done: {n} predictions in {time.time() - t0:.1f}s -> {args.out}")
    if missing:
        print(f"WARNING {missing} rows were not predicted; re-run the same command")


if __name__ == "__main__":
    main()
