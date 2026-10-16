# OCSR-complex — a benchmark for hard Optical Chemical Structure Recognition

A private benchmark of structurally hard chemical depictions for evaluating
Optical Chemical Structure Recognition (OCSR / molecule-from-image) models.
It over-samples drawings whose 2D layout is genuinely ambiguous: bridged
systems, spiro centres, large macrocycles, stereo-dense molecules, and
coordination complexes.

Structures are drawn in ChemDraw across several publication styles, each with
small geometric perturbations. This repository is a public overview only: a
short description, headline scores, and the scoring tools. The image grid,
manifests, and ground truth are not released.

![Sample gallery](samples/gallery.png)

The pictures under `samples/` show the drawing style. They are not the
evaluation set.

---

## Scores

Headline on the held-out slice. The organic number is the mean of per-compound
exact-match rates, with compact molecules down-weighted. Coordination uses a
ligand-skeleton score and is not part of the organic number.

| model | organic exact % | tautomer % | connectivity % | coordination parts % |
|---|---|---|---|---|
| **OCSRGlyph** (Swin-B/384 + transformer decoder) | 18.0 | 20.2 | 42.4 | 30.3 |
| **TinyChemVL** (InternViT-6B + Qwen2.5-3B) | 14.6 | 16.9 | 24.1 | 3.5 |
| **MolSight** (EfficientViT-L1 + 512-d/6-layer decoder) | 20.2 | 21.0 | 40.2 | 27.2 |
| MolSight-stereo (LoRA) | 19.1 | 20.3 | 38.0 | 37.7 |
| **MolParser-Mobile-V2** (ViT-Tiny + 6-layer decoder, 10.00M) | 27.9 | 29.2 | 41.9 | 3.9 |
| **COMO** (Swin-B + 6-layer decoder, Tanimoto) | 35.5 | 38.5 | 55.6 | 5.3 |
| **MolScribe** (Swin-B + graph decoder) | 41.1 | 44.1 | 53.8 | 3.1 |
| **MarkushGlyph** (Qwen3.5-2B + LoRA) | 5.2 | 8.0 | 17.5 | 14.5 |
| **DECIMER** (EfficientNetV2 + transformer) | **58.6** | **58.2** | **71.0** | **40.8** |

The OCSRGlyph row is full-stereo exact match. The same predictions, scored with
that model's chirality-kept convention (tetrahedral chirality kept, cis/trans
ignored), reach **20.9%** organic exact. Connectivity and the coordination
score stay 42.4% and 30.3%.

COMO is the Tanimoto-reward checkpoint `models/tanimoto/best.pth` from
[Keylab/COMO](https://huggingface.co/Keylab/COMO), decoded greedily. The scored
string is its postprocess SMILES, with cis-trans markers dropped before
canonicalization. Those weights are CC BY-NC 4.0.

MolParser-Mobile-V2 is
[UniParser/MolParser-Mobile-V2](https://huggingface.co/UniParser/MolParser-Mobile-V2),
decoded greedily in FP16 with a 384-token limit. The scored string is the
E-SMILES trunk before a `<sep>` or `|` suffix. Those weights are CC BY-NC-SA 4.0.

MolScribe is
[yujieq/MolScribe](https://huggingface.co/yujieq/MolScribe)
checkpoint `swin_base_char_aux_1m680k`, decoded greedily. The scored string is
the graph-to-SMILES output. Those weights are MIT.

MarkushGlyph is the
[EdisonScientific/MarkushGlyph](https://huggingface.co/EdisonScientific/MarkushGlyph)
LoRA on `Qwen/Qwen3.5-2B-Base`, decoded greedily. The prompt asks for an
ordinary molecule and an empty Markush extension. The scored string is the
CXSMILES core before a `|` extension. Those weights are Apache 2.0.

DECIMER is version 2.7.1 from
[Kohulan/DECIMER-Image_Transformer](https://github.com/Kohulan/DECIMER-Image_Transformer),
decoded greedily with the standard model. The scored string is the SMILES
produced by decoding its SELFIES output. Those weights are MIT.

Slice membership, per-compound results, and finer breakdowns are not published.

The same models report much higher figures on drug-like patent sets
(OCSRGlyph 93.8 % exact on USPTO-OCSR-5719, MolSight 91.2 % on USPTO,
COMO 96.1 % on USPTO-10K, MolParser-Mobile-V2 90.9 % accuracy on USPTO).

---

## Tools

```
.
└── tools/
    ├── make_random_split.py  draw an evaluation slice from a local manifest
    ├── eval_ocsr.py          score predictions
    ├── compare_models.py     side-by-side comparison
    ├── predict_ocsrglyph.py  run OCSRGlyph
    ├── predict_tinychemvl.py run TinyChemVL
    ├── predict_molsight.py   run MolSight
    ├── predict_como.py       run COMO
    ├── predict_molparser.py  run MolParser-Mobile-V2
    ├── predict_molscribe.py  run MolScribe
    ├── predict_markushglyph.py  run MarkushGlyph
    └── predict_decimer.py    run DECIMER 2.7
```

The predict scripts read a manifest that is not in this repository.

`predict_como.py` needs `pip install como-ocsr` and a checkpoint via
`--checkpoint` or `$COMO_CHECKPOINT`.

`predict_molparser.py` needs `torch`, `transformers`, `timm`, and
`opencv-python`, and a local checkout via `--model` or `$MOLPARSER_MODEL`.

`predict_molscribe.py` needs `molscribe`, `torch`, `timm`, `opencv-python`,
and `rdkit`, and a checkpoint via `--checkpoint` or `$MOLSCRIBE_CHECKPOINT`.

`predict_markushglyph.py` needs a glyph checkout via `$GLYPH_REPO`, plus the
adapter directory (`--checkpoint` or `$MARKUSHGLYPH_CHECKPOINT`) and
`Qwen3.5-2B-Base` (`--base-model` or `$MARKUSH_BASE_MODEL`).

`predict_decimer.py` needs `decimer==2.7.1`. The first import downloads the
SavedModels into `$PYSTOW_HOME/DECIMER-V2`.

`predict_molsight.py` needs a checkout of
[hustvl/MolSight](https://github.com/hustvl/MolSight) (`--repo-dir` or
`$MOLSIGHT_REPO`). Released weights are fetched from `Robert-zwr/MolSight`
unless `--checkpoint` or `$MOLSIGHT_CHECKPOINT` is set.

```bash
python tools/eval_ocsr.py \
    --manifest path/to/manifest.jsonl \
    --predictions preds.jsonl \
    --out results/

# {"image_id": "<id>", "prediction": "<SMILES>"}
```

Reported metrics: `valid` (parseable), `exact` (isomeric canonical match),
`connectivity` (stereo-blind canonical match), `inchikey_skel`, `inchikey_full`.

---

## Limitations

* Coordination is scored apart from the organic headline. Multi-centre bonding
  has more than one valid SMILES, so that score matches ligand skeletons
  rather than one string.
* Records stored as mixtures of homologues are not included.
* InChI normalises metal bonding poorly, so those drawings are not judged by
  InChIKey.
* Depictions are generated, not copied from published figures. Layouts are
  computed, then styled and rasterised in ChemDraw.
* ChemDraw is commercial software. Code and the example files in this
  repository are MIT — see [LICENSE](LICENSE). Structures used to build the
  private set come from [PubChem](https://pubchem.ncbi.nlm.nih.gov/), with
  [Wikidata](https://www.wikidata.org/) used only as a cross-check.
