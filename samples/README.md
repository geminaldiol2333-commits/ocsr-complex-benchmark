# Samples

24 example drawings from the OCSR-complex benchmark, chosen to show what makes
the corpus hard rather than to be a representative sample.

* `gallery.png` — labelled contact sheet, grouped by the point each row makes
* `ground_truth.csv` — one row per image, 34 columns
* `<image_id>.png` — the individual drawings

## What the sections show

| section | point |
|---|---|
| One molecule, five drawing styles | the same structure under `acs1996`, `thieme`, `angewandte`, `nature` and the CPK colour theme |
| Geometric variants (paclitaxel) | the ±5° in-plane rotations and out-of-plane tilts the pipeline applies |
| Bridged, spiro and cage systems | the occlusion cases the benchmark targets |
| Macrocycles and very large molecules | ring-closure bonds up to 236 heavy atoms |
| Coordination complexes | why metal complexes are a separate subset |

## `ground_truth.csv` columns

`image_id`, `file`, `name`, `subset`, `group`, `template`, `variant`,
`rot_z_deg`, `tilt_x_deg`, `tilt_y_deg`, `width_px`, `height_px`,
`formula`, `mw`, `smiles_isomeric`, `inchi`, `inchikey`,
`n_heavy_atoms`, `n_rings_sssr`, `n_bridgehead_atoms`, `n_spiro_atoms`,
`n_macrocycles`, `n_stereo_assigned`, `occlusion_score`,
`has_bridged_ring`, `has_spiro`, `has_macrocycle`, `has_cage`,
`is_stereo_dense`, `is_large`, `is_organometallic`, `focus`,
`pubchem_cid`, `pubchem_url`

`smiles_isomeric` is the primary ground truth. `file` is relative to this
directory.

## Regenerating

These images are copied out of a full benchmark build, so the pipeline itself is
not runnable from this repository alone — the generator lives with the dataset
source. The sample list is fixed: it is the `SECTIONS` table in the dataset's
`tools/make_public_samples.py`.
