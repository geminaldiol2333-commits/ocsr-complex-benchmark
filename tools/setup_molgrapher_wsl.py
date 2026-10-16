# -*- coding: utf-8 -*-
"""Prepare a native WSL run of MolGrapher on CUDA.

The upstream repository pins PyTorch 1.12 (CUDA 11.7) in ``setup.py`` and its
Dockerfile installs a CPU-only wheel. Neither can drive a Blackwell card, so
this script only patches the two things that hard-code the old device choice:

* ``use_gpu=not force_cpu`` in the PaddleOCR constructors. PaddleOCR runs on
  CPU unless ``MOLGRAPHER_OCR_GPU=1``; the PyTorch models still run on CUDA.
* The ``--visualize`` default, which otherwise writes a matplotlib figure for
  every input image next to the prediction.

Run it inside WSL against a checkout of DS4SD/MolGrapher::

    python3 tools/setup_molgrapher_wsl.py --repo ~/src/MolGrapher
"""
from __future__ import annotations

import argparse
import os
import sys

OCR_FILES = [
    "molgrapher/utils/utils_dataset.py",
    "molgrapher/models/abbreviation_detector.py",
]

HELPER = '''

def _ocr_use_gpu(force_cpu: bool) -> bool:
    """PaddleOCR device. CPU by default; set MOLGRAPHER_OCR_GPU=1 to opt in."""
    return (not force_cpu) and os.environ.get("MOLGRAPHER_OCR_GPU", "0") == "1"
'''


def ensure_os_import(text: str) -> str:
    if "\nimport os\n" in text:
        return text
    return text.replace("\nimport json\n", "\nimport json\nimport os\n", 1)


def patch_ocr(repo: str) -> list[str]:
    changed = []
    for rel in OCR_FILES:
        path = os.path.join(repo, rel)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if "_ocr_use_gpu(" in text:
            continue
        if "use_gpu=not (force_cpu)" not in text:
            print(f"WARN no use_gpu line in {rel}", file=sys.stderr)
            continue
        text = ensure_os_import(text)
        text = text.replace("use_gpu=not (force_cpu)", "use_gpu=_ocr_use_gpu(force_cpu)")
        # Put the helper after the import block of the module.
        lines = text.splitlines(keepends=True)
        last_import = max(
            i for i, line in enumerate(lines) if line.startswith(("import ", "from "))
        )
        lines.insert(last_import + 1, HELPER)
        text = "".join(lines)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        changed.append(rel)
    return changed


def patch_visualize_default(repo: str) -> bool:
    path = os.path.join(repo, "molgrapher/scripts/annotate/predict_molgrapher.py")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    old = 'parser.add_argument("--visualize", type=bool, default=True)'
    new = 'parser.add_argument("--visualize", type=bool, default=False)'
    if old not in text:
        return new in text
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace(old, new, 1))
    return True


GRAPH_TO_RDKIT_OLD = '''            predicted_molecule = graph.to_rdkit(
                abbreviations,
                self.abbreviations_smiles_mapping,
                self.ocr_atoms_classes_mapping,
                self.spelling_corrector,
                assign_stereo=(
                    self.args["assign_stereo"]
                    and (self.args["node_classifier_variant"] == "gc_stereo_model")
                ),
                align_rdkit_output=self.args["align_rdkit_output"],
                postprocessing_flags={},
            )
            predicted_molecules.append(predicted_molecule)
'''

GRAPH_TO_RDKIT_NEW = '''            try:
                predicted_molecule = graph.to_rdkit(
                    abbreviations,
                    self.abbreviations_smiles_mapping,
                    self.ocr_atoms_classes_mapping,
                    self.spelling_corrector,
                    assign_stereo=(
                        self.args["assign_stereo"]
                        and (self.args["node_classifier_variant"] == "gc_stereo_model")
                    ),
                    align_rdkit_output=self.args["align_rdkit_output"],
                    postprocessing_flags={},
                )
            except Exception as exception:
                # Without this one unparsable graph aborts the whole run.
                print(f"Graph creation failed for {p}: {exception!r}")
                predicted_molecule = None
            predicted_molecules.append(predicted_molecule)
'''

SMILES_LOOP_OLD = '''        for i, (predicted_molecule, image_filename) in enumerate(
            zip(predictions["molecules"], images_filenames)
        ):
            if self.args["save_mol_folder"] != "":
'''

SMILES_LOOP_NEW = '''        for i, (predicted_molecule, image_filename) in enumerate(
            zip(predictions["molecules"], images_filenames)
        ):
            if predicted_molecule is None:
                predictions["smiles"].append(None)
                predictions["confidences"][i] = 0
                continue
            if self.args["save_mol_folder"] != "":
'''


def patch_graph_creation(repo: str) -> str:
    """Skip the failing image instead of aborting the batch."""
    path = os.path.join(repo, "molgrapher/models/molgrapher_model.py")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if "Graph creation failed for" in text:
        return "already patched"
    for old, new in ((GRAPH_TO_RDKIT_OLD, GRAPH_TO_RDKIT_NEW),
                     (SMILES_LOOP_OLD, SMILES_LOOP_NEW)):
        if old not in text:
            return "FAILED: pattern not found"
        text = text.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return "patched"


ABBREVIATION_OLD = '''        abbreviations_list = self.abbreviation_detector.mp_run(
            images_filenames, predictions["graphs"], bonds_sizes, filter=False
        )'''

ABBREVIATION_NEW = '''        # Upstream runs PaddleOCR on every graph. Its detection thresholds are so
        # permissive that on a clean drawing it reads ring outlines and bond
        # crossings as labels, and the junk abbreviations then collapse the graph
        # to a single atom. With MOLGRAPHER_ABBREVIATION_FILTER=1 the OCR only
        # runs on graphs that actually contain a node classified as a text label.
        abbreviations_list = self.abbreviation_detector.mp_run(
            images_filenames,
            predictions["graphs"],
            bonds_sizes,
            filter=os.environ.get("MOLGRAPHER_ABBREVIATION_FILTER", "0") == "1",
        )'''


def patch_abbreviation_filter(repo: str) -> str:
    """Make the unconditional abbreviation OCR switchable."""
    path = os.path.join(repo, "molgrapher/models/molgrapher_model.py")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if "MOLGRAPHER_ABBREVIATION_FILTER" in text:
        return "already patched"
    if ABBREVIATION_OLD not in text:
        return "FAILED: pattern not found"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace(ABBREVIATION_OLD, ABBREVIATION_NEW, 1))
    return "patched"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/src/MolGrapher"))
    args = ap.parse_args()

    changed = patch_ocr(args.repo)
    visualize = patch_visualize_default(args.repo)
    graph_creation = patch_graph_creation(args.repo)
    abbreviation = patch_abbreviation_filter(args.repo)
    print(f"ocr patched: {changed}")
    print(f"visualize default patched: {visualize}")
    print(f"graph creation guarded: {graph_creation}")
    print(f"abbreviation OCR filter added: {abbreviation}")


if __name__ == "__main__":
    main()
