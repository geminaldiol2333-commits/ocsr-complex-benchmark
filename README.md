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
| structure source | PubChem (100 %), cross-checked with Wikidata |

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

It is representative: OCSRGlyph scores 21.3 % exact on the slice versus 21.6 %
on the full grid.

---

## Baseline results

Two open OCSR models, run end-to-end on the slice (both at full coverage,
1,854 images):

| model | parseable % | exact % | connectivity % | InChIKey % | speed |
|---|---|---|---|---|---|
| **OCSRGlyph** (Swin-B/384 + transformer decoder) | 73.5 | **21.3** | **41.8** | 21.4 | 0.04 s/img |
| **TinyChemVL** (InternViT-6B + Qwen2.5-3B) | 92.9 | 14.5 | 23.0 | 14.6 | 0.97 s/img |

`parseable` = the model returned something RDKit can read as a molecule.
TinyChemVL often answers with a *description* instead of a SMILES; those count
as parse failures.

**Where they collapse** (exact %, OCSRGlyph / TinyChemVL):

| difficulty tag | OCSRGlyph | TinyChemVL |
|---|---|---|
| all | 21.3 | 14.5 |
| `has_bridged_ring` | 9.4 | 7.9 |
| `has_spiro` | 6.2 | 7.6 |
| `has_macrocycle` | 3.1 | 2.8 |
| `has_cage` | 1.4 | 2.7 |
| `is_stereo_dense` | 8.3 | 6.1 |
| `is_large` (≥60 heavy atoms) | 1.5 | 0.7 |
| `is_organometallic` | **28.4** | 2.3 |
| `bridged + stereo-dense` | 2.9 | 1.7 |

Pairwise over all 1,854 images: both correct **7.3 %**, only OCSRGlyph 14.0 %,
only TinyChemVL 7.1 %, **neither 71.6 %**.

What this says:

* **Drawing style is nearly a non-factor.** Exact-match moves only ~4 points
  across the seven templates — the models are style-robust.
* **Small perturbations are nearly a non-factor.** ±5° rotations and tilts cost
  ~1.5 points. The difficulty here is chemical, not photographic.
* **Chemical complexity dominates**, falling to 1–3 % on cages, large molecules
  and macrocycles.
* **Stereochemistry is the largest single error source** — connectivity is
  roughly twice the exact-match rate.

Both models score far below their published numbers (OCSRGlyph reports 93.8 %
exact on USPTO-OCSR-5719), which is the intended signal: that corpus is
dominated by drug-like molecules, this one by the structures 2D OCSR still
cannot read.

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
    └── predict_tinychemvl.py run TinyChemVL over a manifest
```

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

* **Coordination compounds.** Multi-centre bonding is not expressible in plain
  SMILES. Following the PubChem convention the metal is often a separate
  fragment or written with dative bonds, so `n_fragments > 1` on those rows is
  expected. Sandwich complexes (ferrocene, cobaltocene …) are therefore drawn as
  free rings plus the metal ion rather than with haptic bonds — a faithful
  rendering of the deposited structure, but not the classical
  metal-between-two-rings cartoon. They live in a separate subset and should be
  scored on their own.
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
