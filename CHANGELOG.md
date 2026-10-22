# Changelog

## v2.0.4 — 2026-10-22

Add a Chinese README and a language switch at the top of both files. Numeric changes in the English README are written with a plain arrow, so the page no longer shows the Chinese word for "to" or a trailing currency sign.

## v2.0.3 — 2026-10-22

Write numeric ranges as plain text. A range such as 30–60 had been wrapped in dollar signs, and the closing dollar sign was shown as the currency word.

## v2.0.2 — 2026-10-22

Fix the section 2.3 table. The median heavy-atom header used raw vertical bars inside the math, which Markdown treated as extra columns, so the table did not render.

## v2.0.1 — 2026-10-22

The sample gallery is redrawn with the v2.0 depiction pipeline. It keeps the v1.1 organic legends and no longer includes coordination complexes. The drawings are illustrations, not the held-out evaluation set.

## v2.0.0 — 2026-10-22

The organic headline is the ACS 1996 pose protocol: one journal stylesheet and every geometric variant, scored at compound level. v1.0 and v1.1 found that stylesheet choice moves exact-match by only a few points, so v2.0 spends the evaluation on pose coverage instead of another random template slice. The image grid, manifests, and ground truth remain private.

- `tools/eval_ocsr.py` keeps coordination on the part key, and leaves chiral, E/Z, and tautomer rates blank when a compound does not define that property. `--template acs1996` selects the v2.0 headline. Pass the v1.1 manifest to reproduce the earlier slice.
- The README leaderboard is the 11-model v2.0 result, set next to the v1.1 numbers. Pose variation inside ACS 1996 is reported as a standard deviation, not as a second benchmark.
- `tools/verify_reported_numbers.py` reads `README.md` and the v2.0 column layout.

## v1.1.0

Tagged from `2dbfcd0`, the publication-grade held-out slice analysis already on `main`: compact organics down-weighted, coordination scored with the part key, and the 11-model error taxonomy.
