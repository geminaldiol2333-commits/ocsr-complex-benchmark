# OCSR-complex — a benchmark for hard Optical Chemical Structure Recognition

A curated benchmark of **structurally hard** chemical entities for training and
evaluating Optical Chemical Structure Recognition (OCSR / molecule-from-image)
models. The corpus deliberately over-samples the cases where 2D depictions are
genuinely ambiguous:

* **bridged polycyclics** — bonds drawn *behind* a bridge: the classic occlusion
  failure mode (strychnine, taxanes, aconitine, ginkgolides, tetrodotoxin …)
* **spiro systems** — quaternary spiro centres (griseofulvin, spironolactone,
  okadaic acid, azaspiracid, diosgenin, ionophore polyethers …)
* **large macrocycles** — ring-closure bonds and degenerate conformers
  (macrolide antibiotics, ansamycins, cyclosporin, zearalenone …)
* **stereo-dense natural products** — dozens of wedge/hash bonds
  (palytoxin, maitotoxin, brevetoxin, vancomycin …)
* **coordination / organometallic complexes** — a separate subset
  (vitamin B12, porphyrins, phthalocyanines, metallocenes, Pt drugs, Grubbs catalysts)

Every structure is drawn by **ChemDraw** in seven publication style templates,
each with five small geometric perturbations. Published here are the
**overview, a sample gallery with ground truth, and the tooling**; the full
image set is not redistributed.

![Sample gallery](samples/gallery.png)

---

## Dataset at a glance

| | |
|---|---|
| unique compounds | **309** (organic 271, coordination 38) |
| images (full grid) | **10,815** = 309 × 7 templates × 5 variants |
| recommended evaluation slice | **1,854** (`random_seed42`, see below) |
| median / max heavy atoms | 38 / **236** (`maitotoxin`) |
| median / max assigned stereocentres | 7 / **98** (`maitotoxin`) |
| structure source | PubChem. Wikidata was checked; where the two skeletons disagreed, PubChem was kept |

| group | compounds |
|---|---|
| bridged / spiro / cage | 72 |
| drugs | 69 |
| toxins | 60 |
| coordination | 38 |
| macrolides | 36 |
| peptide antibiotics | 34 |

| difficulty tag | compounds | share |
|---|---|---|
| `has_bridged_ring` | 157 | 51 % |
| `has_spiro` | 83 | 27 % |
| `has_macrocycle` | 90 | 29 % |
| `has_cage` | 61 | 20 % |
| `is_stereo_dense` (≥6 stereocentres) | 190 | 61 % |
| `is_large` (≥60 heavy atoms) | 68 | 22 % |
| `is_organometallic` | 37 | 12 % |

Largest molecules: `maitotoxin` (236 heavy atoms, 98 stereocentres),
`nisin-a` (229), `palytoxin` (186), `ramoplanin` (158), `gramicidin-a` (136),
`teicoplanin` (132), `adenosylcobalamin` (109).

---

## Drawing style templates

Dimensions come from each publisher's own author instructions — they are not
invented approximations. Bond length and stroke widths in points, bond spacing
as a percentage of the nominal bond length.

| parameter | `chemdraw-default` | `acs1996` | `nature` | `rsc` | `angewandte` | `thieme` | `theme` |
|---|---|---|---|---|---|---|---|
| BondLength | app default (14.4) | 14.4 | 10.80 | 12.20 | 17.00 | 10.12 | 14.4 |
| BondSpacing | 18.5 | 18.0 | 18.0 | 20.0 | 18.0 | 18.0 | 18.0 |
| LineWidth | 0.60 | 0.60 | 0.5953 | 0.50 | 0.75 | 0.5669 | 0.60 |
| BoldWidth | 2.00 | 2.00 | 1.559 | 1.60 | 2.602 | 1.389 | 2.00 |
| MarginWidth | 1.60 | 1.60 | 1.191 | 1.25 | 2.00 | 0.9071 | 1.60 |
| HashSpacing | 2.50 | 2.50 | 1.701 | 1.80 | 2.602 | 1.757 | 2.50 |
| ChainAngle | 120 | 120 | 120 | 120 | 120 | 120 | 120 |
| Label font / size | Arial 10 | Arial 10 | Helvetica 6 | Arial 7 | Arial 12 | Arial 6 | Arial 10 |
| Colour | black | black | black | black | black | black | CPK colours |

* `acs1996` — ACS Document 1996 / JACS, the most common style in the literature.
* `nature` — Nature's published ChemDraw stylesheet.
* `rsc` — Royal Society of Chemistry author instructions.
* `angewandte` — Angewandte Chemie (Wiley) author stationery.
* `thieme` — Synlett / Synthesis, final-size dimensions from Thieme's
  instructions (already converted from cm; **not** to be reduced to 69 % — that
  figure applies only to ChemDraw's bundled Synlett stationery).
* `theme` — ACS geometry with per-atom CPK colours, applied atom-by-atom.

## Geometric variants

| variant | in-plane rotation | out-of-plane tilt |
|---|---|---|
| `v0_orig` | 0° | 0° |
| `rz_a`, `rz_b` | ±2–5° about the paper normal | — |
| `tilt_x` | — | ±2–5° about the horizontal in-paper axis |
| `tilt_y` | — | ±2–5° about the vertical in-paper axis |

**Every angle is within ±5°.** In-plane rotations are applied inside ChemDraw as
true vector geometry. Out-of-plane tilts are rendered as a homography: the
depiction is treated as a flat sheet rotated by φ about a paper axis and
projected through a pinhole camera with a short focal length
(`f = 0.8 × max(width, height)`), which is what makes a ≤5° tilt produce a
visible keystone. Angles are drawn from a seeded PRNG and recorded per image.
All images are trimmed to the drawing plus a uniform 4 % margin, so framing
carries no information about the variant.

---

## Ground truth

One row per image. Fields include:

| field | meaning |
|---|---|
| `image_id` | `<compound_id>__<template>__<variant>` |
| `image_path` | path relative to the benchmark root |
| `compound_id`, `name`, `subset`, `group` | identity and slicing |
| `template`, `variant` | drawing style and geometric variant |
| `rot_z_deg`, `tilt_x_deg`, `tilt_y_deg` | exact transform applied |
| `smiles_isomeric` | **primary ground truth SMILES** |
| `smiles_canonical` | stereo removed |
| `cxsmiles` | CXSMILES (extended / E-SMILES) |
| `inchi`, `inchikey` | cross-format ground truth |
| `smarts` | SMARTS generated from the ground truth |
| `formula`, `mw`, `exact_mass`, `charge` | molecular identifiers |
| `pubchem_cid`, `pubchem_url` | provenance |
| `n_heavy_atoms`, `n_rings_sssr`, `n_bridgehead_atoms`, `n_spiro_atoms`, `n_macrocycles`, `n_stereo_assigned`, `occlusion_score`, … | RDKit structural features |
| `has_bridged_ring`, `has_spiro`, `has_macrocycle`, `has_cage`, `is_stereo_dense`, `is_large`, `is_organometallic` | boolean difficulty tags |

Per compound, the full release also ships `.smi`, `.canonical.smi`,
`.cxsmiles`, `.smarts`, `.inchi`, `.mol` (the ChemDraw layout),
`.cdxml` and `.svg`.

`occlusion_score = n_bridgehead_atoms + n_multi_ring_atoms` is a simple proxy
for *how likely a 2D depiction is to hide bonds behind a bridge* — it exists to
slice an OCSR model's errors on exactly the failure mode this benchmark targets.

Structures were accepted only after automated checking: the SMILES must parse,
the InChI regenerated from it must equal the deposited InChI, and the heavy-atom
count from the molecular formula must match. **Every shipped structure passed all
three**, and each drawing was verified post-render by comparing CIP codes of
mapped stereocentres plus stereo-blind connectivity against the ground truth
(309/309 exact).

---

## Randomised evaluation slice

Scoring all 10,815 images is wasteful once the question is just *"how good is
this model on hard structures?"*. The slice samples the benchmark at **one sixth
the cost** while keeping every compound represented.

**Sampling contract** (`tools/make_random_split.py`):

* every compound contributes **3 of the 7 templates** and **2 of the 5 variants**
  per chosen template;
* template choice is *semi-random*: `acs1996` is assigned a target share of
  **30 % of all template slots** — 278 of 309 compounds carry it, with the
  remaining slots spread over the other six templates — rather than the 43 % a
  uniform draw would give it;
* the selection is a pure function of the parameters (seeded PRNG), so the slice
  is reproducible and a different `--seed` yields a different slice.

The shipped slice `random_seed42` contains **1,854 images** (309 × 3 × 2):
template slots 927 with `acs1996` at 30.0 %, variants roughly even at 356–392
images each.

The v1.1 headline is not that image-micro average. It is the mean of each
organic compound, with compact molecules (≤20 heavy atoms and ≤2 assigned
stereocentres; 22 compounds) at weight 0.25. Coordination is scored separately.
On this slice the organic exact rates are OCSRGlyph 18.0 %, TinyChemVL 14.6 %,
MolSight 20.2 %, MolSight-stereo 19.1 %. Coordination, scored by ligand
skeleton on the redrawn structures, is OCSRGlyph 30.3 %, TinyChemVL 3.5 %,
MolSight 27.2 %, MolSight-stereo 37.7 %. The older 21.3 % / 23.2 % figures
counted every image equally and included the coordination drawings from before
they were redrawn.

---

## Baseline results

The table below is the **v1.0 image-micro** average on the slice (every image
weighted equally, including the coordination drawings from before the redraw).
The v1.1 organic headline is in the previous section.

Three open OCSR models, run end-to-end on the slice (all at full coverage,
1,854 images):

| model | parseable % | exact % | connectivity % | InChIKey % | speed |
|---|---|---|---|---|---|
| **OCSRGlyph** (Swin-B/384 + transformer decoder) | 73.5 | 21.3 | **41.8** | 21.4 | 0.04 s/img |
| **TinyChemVL** (InternViT-6B + Qwen2.5-3B) | **92.9** | 14.5 | 23.0 | 14.6 | 0.97 s/img |
| **MolSight** (EfficientViT-L1 + 512-d/6-layer decoder) | 74.9 | **23.2** | 40.2 | **23.2** | 0.095 s/img |

`parseable` = the model returned something RDKit can read as a molecule.
TinyChemVL often answers with a *description* instead of a SMILES; those count
as parse failures.

**Where they collapse** (exact %, OCSRGlyph / TinyChemVL / MolSight):

| difficulty tag | OCSRGlyph | TinyChemVL | MolSight |
|---|---|---|---|
| all | 21.3 | 14.5 | **23.2** |
| `has_bridged_ring` | **9.4** | 7.9 | 9.1 |
| `has_spiro` | 6.2 | **7.6** | 3.0 |
| `has_macrocycle` | 3.1 | 2.8 | **6.1** |
| `has_cage` | 1.4 | 2.7 | **4.6** |
| `is_stereo_dense` | **8.3** | 6.1 | 7.7 |
| `is_large` (≥60 heavy atoms) | 1.5 | 0.7 | **2.7** |
| `is_organometallic` | 28.4 | 2.3 | **30.2** |
| `bridged + stereo-dense` | 2.9 | 1.7 | **3.2** |

Head-to-head over all 1,854 images: all three correct 5.6 %, exclusively
MolSight 6.6 %, exclusively OCSRGlyph 6.1 %, exclusively TinyChemVL 4.0 %, none
64.9 % — the remaining 12.8 % are read by exactly two of the three. Within
that, the two-model OCSRGlyph/TinyChemVL split is: both correct 7.3 %, only
OCSRGlyph 14.0 %, only TinyChemVL 7.1 %, neither 71.6 %.

What this says:

* **Drawing style is nearly a non-factor.** Exact-match moves only ~4 points
  across the seven templates (0.8 for MolSight) — the models are style-robust.
* **Small perturbations are nearly a non-factor.** ±5° rotations and tilts cost
  ~1.5 points. The difficulty here is chemical, not photographic.
* **Chemical complexity dominates**, falling to 1–6 % on cages, large molecules
  and macrocycles.
* **Stereochemistry is the largest single error source** — for OCSRGlyph
  connectivity is roughly twice the exact-match rate; MolSight narrows that gap
  (40.2 % vs 23.2 %).
* **The models fail differently.** MolSight is the only one of the three to read
  **strychnine** — the benchmark's flagship behind-the-bridge occlusion case:
  25 of its 35 drawings, against 0/35 for the other two. It also comes out ahead
  on macrocycles, cages, large molecules and coordination complexes, while never
  solving `cubane` (0/35, where OCSRGlyph scores 35/35).
* **Specialisation does not transfer.** The same network post-trained on 200k
  stereoisomeric molecules (`stereo_grpo_10_2.pth`, SMILES head only) raises
  parseability (74.9 % → 76.9 %) while losing exact matches (23.2 % → 22.4 %),
  and drops strychnine from 25/35 to 0/35.

All three score far below their published numbers (OCSRGlyph reports 93.8 %
exact on USPTO-OCSR-5719, MolSight 91.2 % on USPTO), which is the intended
signal: those corpora are dominated by drug-like molecules, this one by the
structures 2D OCSR still cannot read.

---

## Repository contents

```
.
├── samples/
│   ├── gallery.png           labelled contact sheet
│   ├── ground_truth.csv      SMILES / InChI / features for exactly those images
│   └── <image_id>.png        24 example drawings
└── tools/
    ├── make_random_split.py  draw a randomised evaluation slice
    ├── eval_ocsr.py          score predictions (exact / connectivity / InChIKey)
    ├── compare_models.py     side-by-side model comparison report
    ├── predict_ocsrglyph.py  run OCSRGlyph over a manifest
    ├── predict_tinychemvl.py run TinyChemVL over a manifest
    └── predict_molsight.py   run MolSight over a manifest
```

`predict_molsight.py` needs a checkout of
[hustvl/MolSight](https://github.com/hustvl/MolSight) (`--repo-dir` or
`$MOLSIGHT_REPO`) and pulls the released weights from the Hugging Face repo
`Robert-zwr/MolSight` unless `--checkpoint`/`$MOLSIGHT_CHECKPOINT` says
otherwise; it also works around the checkout's Linux-only Indigo build, which
only the training-time renderer uses.

Scoring a model:

```bash
python tools/eval_ocsr.py \
    --manifest path/to/manifest.jsonl \
    --predictions preds.jsonl \
    --out results/

# preds.jsonl: one row per image
# {"image_id": "strychnine__acs1996__v0_orig", "prediction": "C1CN2CC3=CCO[C@H]4..."}
```

Reported metrics: `valid` (parseable), `exact` (isomeric canonical match),
`connectivity` (stereo-blind canonical match), `inchikey_skel` (14-char skeleton
block), `inchikey_full`. Results are broken down by subset, template, variant
and difficulty tag.

---

## Known limitations

* **Coordination compounds** are a separate score, not part of the headline.
  Thirty-six of them were redrawn so the metal sits in a sandwich, a cavity,
  or a chelate. Ferrocene is two cyclopentadienyl rings with iron between
  them. Several deposited SMILES were the wrong molecule (manganocene as two
  cyclopentanes, bis(benzene)chromium as a hexaanion) and those records were
  replaced. A sandwich still has more than one valid SMILES, so the
  coordination score matches ligand skeletons rather than one string.
* **Mixtures are excluded.** Records stored as mixtures of homologues
  (ivermectin B1a/B1b, spinosad A/D) were rejected rather than rendered as two
  disconnected molecules.
* **InChI and metals.** InChI normalises metal bonding and porphyrin NH
  tautomers poorly, so structure fidelity is checked by CIP-code comparison, not
  by InChIKey.
* **Depictions are algorithmically generated.** Layouts come from RDKit's
  CoordGen engine and are styled and rasterised by ChemDraw. They are chemically
  correct and journal-styled but not identical to any specific figure in the
  literature — deliberately, to avoid re-using copyrighted artwork.
* **Per-template conclusions need the full grid.** On the 1,854-image slice each
  template has only 196–236 images, so template rankings are noisy
  (OCSRGlyph ranges 14.8–29.1 % there, but 20.4–24.1 % on the full grid).

## Attribution and licence

Chemical structures were retrieved from
[PubChem](https://pubchem.ncbi.nlm.nih.gov/) (NCBI) and cross-checked against
[Wikidata](https://www.wikidata.org/). No third-party copyrighted figures are
redistributed: every image here was rendered by this pipeline. Note that the
depictions were produced with ChemDraw, which is commercial software.

Code and the released sample data are **MIT licensed** — see [LICENSE](LICENSE).
