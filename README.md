# OCSR-complex: A Held-Out Benchmark for Structurally Hard Optical Chemical Structure Recognition

**OCSR-complex** is a held-out evaluation benchmark designed to stress-test Optical Chemical Structure Recognition (OCSR / image-to-structure) systems beyond the drug-like patent regime. Whereas contemporary OCSR models routinely report $>90\%$ exact-match accuracy on standard patent corpora (e.g., USPTO, Clef, UOB), their performance degrades sharply on stereochemically dense, topologically entangled, and multi-centre coordinated depictions that are ubiquitous in natural products, total synthesis, and organometallic literature.

To prevent training-set contamination and benchmark overfitting, **the evaluation image grid, sampling manifests, compound registry, and ground-truth annotations remain strictly private**. This public repository provides the benchmark methodology, the 11-model evaluation leaderboard and diagnostic analysis, an illustrative sample gallery (`samples/`), and the complete inference and scoring harness (`tools/`).

![Sample gallery](samples/gallery.png)

*Figure 1: Illustrative examples of the rendering pipeline under `samples/`. These depictions demonstrate the journal stylesheets and geometric perturbations used in OCSR-complex; they are not part of the held-out evaluation slice.*

---

## 1. Benchmark Design & Evaluation Protocol

### 1.1 Targeted Structural Failure Modes
The benchmark deliberately over-samples five classes of 2D chemical depictions where visual perception and topological reconstruction are genuinely ambiguous:

1. **Bridged Polycyclic & Cage Architectures** — Systems containing bonds drawn *behind* a bridgehead or within high-strain polycyclic cages, inducing severe 2D bond-crossing occlusion (`has_bridged_ring`, `has_cage`).
2. **Quaternary Spiro Systems** — Molecules featuring one or more spiro-fused ring junctions where local bond-angle congestion complicates vertex degree assignment (`has_spiro`).
3. **Macrocycles & Large Biopolymers** — Large-ring macrolides, polyenes, cyclic peptides, and polyether ionophores where long-range ring-closure tracking and conformational degeneracy challenge sequence and graph decoders (`has_macrocycle`, `is_large`).
4. **Stereo-Dense Natural Products** — Scaffolds carrying dense arrays of wedge/hash bonds and configurated alkenes, testing fine-grained local visual discrimination coupled with global Cahn–Ingold–Prelog (CIP) / SMILES parity consistency (`is_stereo_dense`).
5. **Coordination & Organometallic Complexes** — A dedicated subset spanning $\eta^5/\eta^6$ sandwich metallocenes, metallated macrocyclic cavities (porphyrins, chlorins, phthalocyanines, corrins), and $\sigma$-bonded / chelated transition-metal complexes (`is_organometallic`).

### 1.2 Authentic Typographic & Geometric Diversity
All structures are curated from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) (cross-verified against [Wikidata](https://www.wikidata.org/) for constitutional, formula, and InChI consistency) and laid out via explicit 2D coordinate generation without destructive auto-cleaning. Depictions are rendered in **ChemDraw** across multiple authentic publisher stylesheets (matching official journal graphic specifications for bond length, stroke width, bold wedge width, hash spacing, atom label typography, and CPK element coloring), combined with bounded in-plane vector rotations and out-of-plane perspective homography tilts.

### 1.3 Scoring Metrics
Evaluation is conducted on a stratified, seeded held-out slice (`random_seed42`). Because organic molecules and multi-centre coordination complexes obey fundamentally different representation rules in cheminformatics, they are scored separately:

* **Organic Headline (`Organic Exact %`)**: Computed at the compound level (mean accuracy across each compound's evaluated depictions) using a stratified weighting scheme that down-weights compact baseline structures so that complex scaffolds govern the headline metric. A prediction is scored as `Exact` if and only if its RDKit canonical isomeric SMILES is identical to the ground truth.
* **Chirality-Kept Match (`Chirality %`)**: Evaluates canonical SMILES identity after stripping double-bond $E/Z$ (`/` and `\`) directional markers while strictly retaining all tetrahedral (`@`/`@@`) stereocentres. Several OCSR pipelines intentionally omit or strip alkene $E/Z$ stereochemistry during post-processing; comparing `Organic Exact %` against `Chirality %` isolates $E/Z$ convention penalties from true tetrahedral inversion errors.
* **Tautomer Match (`Tautomer %`)**: Evaluates canonical-tautomer InChIKey equivalence via RDKit `TautomerEnumerator` (applied to structures up to 80 heavy atoms), ensuring that valid prototropic tautomers (e.g., keto–enol or amide–imidol forms) are not penalized as topological errors.
* **Constitutional Connectivity (`Connectivity %`)**: Stereo-blind canonical SMILES identity (equivalent to matching the 14-character InChIKey skeleton block on organic structures), measuring pure 2D molecular graph topology recovery.
* **Structural Similarity (`ECFP4 Tanimoto`)**: Mean Morgan fingerprint (radius 2, 2048 bits) Tanimoto similarity over all organic depictions, assigning $0.0$ to unparseable outputs, quantifying partial structural recovery on near-miss predictions. This column is an **image-level, unweighted** mean, unlike the compound-weighted columns beside it; compound-weighting it moves the values down by up to $1.6$ pp (MarkushGlyph $39.1 	o 37.5$).
* **Coordination Ligand-Skeleton Score (`Coordination Parts %`)**: Multi-centre haptic ($\eta^5$-Cp, $\eta^6$-arene) and dative coordination bonding has no unique canonical SMILES representation, and standard InChI normalization frequently corrupts metal–ligand connectivity. Coordination depictions are therefore authored as explicit complexes and evaluated via a canonical **part key** (`coord_depict.py`) that verifies: (i) the identity of each metal centre and its non-haptic $\sigma$/chelate coordination sphere, and (ii) the charge- and radical-normalized canonical skeletons of all coordinated ligands (e.g., aromatic, anionic, and radical cyclopentadienyl encodings match one another, whereas reduced cyclopentane or uncoordinated floating fragments are rejected).

---

## 2. Benchmark Results (11 Models)

### 2.1 Main Leaderboard
We evaluate 11 open-weights OCSR models spanning three primary architectural paradigms:
1. **1D Chemical Line-Notation Autoregressive Decoders** (**DECIMER**, **COMO**, **MolParser-Mobile-V2**, **MolSight**, **MolSight-stereo**, **OCSRGlyph**);
2. **2D Molecular Graph Reconstruction Hybrids** (**MolNexTR**, **MolScribe**, **MolGrapher**);
3. **Vision–Language Models (VLMs)** (**TinyChemVL**, **MarkushGlyph**).

| Model | Paradigm & Backbone | Org. Valid % | **Organic Exact %** (95 % CI) | Chirality ($E/Z$-Blind) % | Tautomer % | Connectivity % | ECFP4 Tanimoto % | **Coord. Parts %** | Published Patent Exact % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DECIMER** (v2.7.1) | EfficientNetV2 + Transformer $\to$ **SELFIES** | 93.0 | **58.6** (52.9–64.3) | **59.2** | **58.2** | **71.0** | **87.2** | **40.8** | ~90–96 (DECIMER) |
| **MolNexTR** | ConvNeXt + ViT + **Graph/SMILES** Decoder | 68.3 | **41.1** (35.5–46.6) | 43.5 | 42.4 | 49.7 | 61.9 | 6.6 | 93.8 (USPTO) |
| **MolScribe** | Swin-B + **Graph** Autoregressive Decoder | 68.2 | **41.1** (35.6–46.8) | 44.0 | 44.1 | 53.8 | 63.8 | 3.1 | 93.3 (USPTO) |
| **COMO** (Tanimoto-RL) | Swin-B + 6L Decoder + Graph Stereo Postproc | 74.0 | **35.5** (29.9–41.2) | 50.3 | 38.5 | 55.6 | 68.5 | 5.3 | 96.1 (USPTO-10K) |
| **MolParser-Mobile-V2** | ViT-Tiny + 6L Decoder (10.0M) $\to$ **E-SMILES** | 79.0 | **27.9** (23.1–33.0) | 28.2 | 29.2 | 41.9 | 65.9 | 3.9 | 90.9 (USPTO) |
| **MolSight** | EfficientViT-L1 + 6L Decoder (67.5M, Char+Edge) | 74.3 | **20.2** (15.9–24.9) | 20.3 | 21.0 | 40.2 | 61.9 | 27.2 | 91.2 (USPTO) |
| **MolSight-stereo** | EfficientViT-L1 + 6L Decoder + Stereo-GRPO LoRA | 76.2 | **19.1** (14.9–23.5) | 19.1 | 20.3 | 38.0 | 61.4 | 37.7 | — |
| **OCSRGlyph** | Swin-B/384 + Transformer Decoder $\to$ **SMILES** | 72.2 | **18.0** (14.0–22.2) | 20.9 | 20.2 | 42.4 | 62.7 | 30.3 | 93.8 (USPTO-5719) |
| **TinyChemVL** | InternViT-6B + Qwen2.5-3B (~9B VLM) $\to$ **SMILES** | **93.5** | **14.6** (11.0–18.4) | 17.2 | 16.9 | 24.1 | 52.1 | 3.5 | — |
| **MolGrapher** | ResNet-18 Keypoints + **GCN** Classifier + OCR | 83.0 | **6.1** (3.7–8.8) | 8.9 | 7.2 | 33.1 | 49.5 | 0.0 | ~92 (USPTO-10K) |
| **MarkushGlyph** | Qwen3.5-2B-Base + LoRA VLM $\to$ **CXSMILES** | 58.2 | **5.2** (2.9–7.8) | 5.6 | 8.0 | 17.5 | 39.1 | 14.5 | — |

> **Confidence intervals and significance.** The 95 % intervals in the exact column come from 10,000 bootstrap resamples of **compounds**, not images: the six drawings of one compound are strongly correlated, so resampling images would understate the uncertainty by roughly a factor of two. Pairwise differences are tested with a paired bootstrap over the same resamples and Holm-corrected across all 55 model pairs (`tools/stats_ci.py`). 45 of the 55 pairs separate at 5 %; the pairs that do **not** are MolNexTR/MolScribe (41.1 vs 41.1, $p_{\text{Holm}} = 1.0$), MolSight/MolSight-stereo/OCSRGlyph ($\le 2.2$ pp apart), MolGrapher/MarkushGlyph (6.1 vs 5.2), and — at the margin — MolSight vs TinyChemVL, MolScribe vs COMO and MolNexTR vs COMO ($p_{\text{Holm}} = 0.07$–$0.09$). Rankings inside those groups are not supported by the data.

> **Note on Compact vs. Complex Scaffolds:** On the compact organic baseline stratum within the benchmark, models achieve substantially higher exact-match rates (**DECIMER** 78.0%, **MolParser-Mobile-V2** 67.4%, **MolScribe** 62.9%, **OCSRGlyph** 58.3%, **COMO** 53.8%, **MolNexTR** 53.0%, **MolSight-stereo** 52.3%, **MolSight** 49.2%, **MolGrapher** 43.2%, **TinyChemVL** 40.9%, **MarkushGlyph** 29.5%). Down-weighting this easy stratum exposes the true performance drop on complex chemical structures, where every evaluated model falls 35–85 percentage points below its published patent-benchmark accuracy.

---

### 2.2 Performance Across Structural Complexity Regimes
Table 2 reports model accuracy across individual structural difficulty tags on the organic evaluation subset (`Exact % / Connectivity %`).

| Model | Bridged Rings | Spiro Centres | Macrocycles | Polycyclic Cages | Stereo-Dense | Large Molecules | Bridged + Stereo-Dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DECIMER** | **44.8 / 58.9** | **48.0 / 64.9** | **41.4 / 60.9** | 14.3 / **34.9** | **49.5 / 63.4** | **35.6 / 48.9** | **36.6 / 51.2** |
| **MolNexTR** | 30.3 / 37.4 | 31.3 / 44.8 | 16.4 / 21.3 | 17.8 / 20.9 | 29.2 / 38.1 | 8.3 / 13.8 | 21.9 / 29.4 |
| **MolScribe** | 28.9 / 39.4 | 26.9 / 45.8 | 22.9 / 31.5 | **22.5** / 25.6 | 30.2 / 43.2 | 12.4 / 15.5 | 17.9 / 31.0 |
| **COMO** | 23.1 / 39.0 | 27.1 / 44.6 | 8.1 / 44.4 | 10.9 / 22.5 | 25.1 / 45.9 | 4.0 / 25.3 | 11.7 / 31.0 |
| **MolParser-Mobile-V2** | 16.8 / 30.6 | 12.4 / 30.3 | 9.5 / 17.4 | 3.9 / 7.4 | 12.5 / 27.6 | 1.7 / 3.7 | 4.9 / 18.2 |
| **MolSight** | 9.6 / 27.6 | 3.0 / 23.5 | 6.2 / 22.0 | 4.3 / 10.1 | 8.0 / 28.0 | 3.2 / 6.0 | 3.4 / 19.2 |
| **MolSight-stereo** | 7.8 / 24.8 | 3.6 / 19.9 | 4.6 / 17.1 | 2.7 / 11.2 | 6.0 / 24.0 | 2.0 / 4.3 | 1.9 / 14.1 |
| **OCSRGlyph** | 10.7 / 30.7 | 6.2 / 33.1 | 3.9 / 19.7 | 1.9 / 9.7 | 8.7 / 31.1 | 1.7 / 4.0 | 3.1 / 21.4 |
| **TinyChemVL** | 8.9 / 16.2 | 7.6 / 11.8 | 3.5 / 8.3 | 3.9 / 8.1 | 6.3 / 12.8 | 0.9 / 2.9 | 1.9 / 9.2 |
| **MolGrapher** | 6.0 / 24.6 | 2.4 / 22.1 | 0.9 / 12.3 | 0.0 / 5.8 | 1.7 / 19.7 | 0.3 / 3.7 | 1.0 / 16.5 |
| **MarkushGlyph** | 1.8 / 8.2 | 0.0 / 2.2 | 0.0 / 2.5 | 2.3 / 2.7 | 0.5 / 6.1 | 0.0 / 0.0 | 0.0 / 3.7 |

---

### 2.3 Error Taxonomy & Stereochemical Fidelity
To disentangle syntactic generation failures, topological graph errors, and stereochemical misassignments, Table 3 partitions $100\%$ of organic image predictions into mutually exclusive outcome categories, alongside conditional stereochemical precision ($\text{Exact} / \text{Connectivity}$) and near-miss similarity on topologically incorrect but valid predictions.

| Model | Exact (image level) % | $E/Z$ Stereo Loss % | Tetrahedral Stereo Loss % | Skeleton / Topo Error % | Unparseable / Invalid % | Conditional Stereo Precision % | Skeleton Error Mean Tanimoto % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DECIMER** | **59.8** | 0.6 | 11.1 | 21.6 | 6.9 | 83.6 | **72.5** |
| **MolNexTR** | 41.8 | 2.4 | 5.7 | 18.8 | 31.4 | **83.8** | 64.1 |
| **MolScribe** | 42.4 | 2.9 | 9.0 | **14.2** | 31.4 | 78.0 | 66.2 |
| **COMO** | 36.6 | 14.3 | **5.0** | 17.9 | 26.1 | 65.4 *(91.0 no-$E/Z$)* | 69.8 |
| **MolParser-Mobile-V2** | 30.3 | 0.4 | 12.7 | 36.2 | 20.4 | 69.8 | 62.1 |
| **MolSight** | 22.0 | 0.1 | 19.3 | 33.4 | 25.2 | 53.1 | 61.4 |
| **MolSight-stereo** | 21.1 | 0.1 | 18.1 | 37.6 | 23.1 | 53.7 | 58.8 |
| **OCSRGlyph** | 20.4 | 2.8 | 20.6 | 29.4 | 26.8 | 46.6 *(53.0 no-$E/Z$)* | 64.2 |
| **TinyChemVL** | 16.2 | 2.6 | 7.1 | 67.7 | **6.5** | 62.5 | 38.7 |
| **MolGrapher** | 8.4 | 2.8 | 23.5 | 49.1 | 16.2 | 24.2 | 30.1 |
| **MarkushGlyph** | 6.6 | 0.4 | 12.5 | 40.5 | 40.0 | 33.9 | 48.4 |

---

### 2.4 Coordination & Organometallic Recognition by Bonding Archetype
Table 4 disaggregates coordination subset performance (`Coordination Parts %` / `Valid %`) across the three major organometallic bonding archetypes in OCSR-complex.

| Model | $\eta^5/\eta^6$ Sandwich Metallocenes | $\sigma$-Bonded / Chelate Complexes | Metallated Macrocyclic Cavities | Overall Coord. Parts % |
|---|---:|---:|---:|---:|
| **DECIMER** | **100.0** / 100.0 | 25.0 / 83.3 | 0.0 / 44.8 | **40.8** |
| **MolSight-stereo** | 91.7 / 7.1* | 25.0 / 91.7 | 0.0 / 9.4 | 37.7 |
| **OCSRGlyph** | 67.9 / 4.8* | **33.3** / **100.0** | 0.0 / 24.0 | 30.3 |
| **MolSight** | 64.3 / 7.1* | 22.2 / 94.4 | 0.0 / 12.5 | 27.2 |
| **MarkushGlyph** | 25.0 / 57.1 | **33.3** / 72.2 | 0.0 / 8.3 | 14.5 |
| **MolNexTR** | 10.7 / 7.1* | 16.7 / 69.4 | 0.0 / 24.0 | 6.6 |
| **COMO** | 7.1 / 7.1 | 16.7 / 66.7 | 0.0 / 7.3 | 5.3 |
| **MolParser-Mobile-V2** | 0.0 / 92.9 | 25.0 / 83.3 | 0.0 / 17.7 | 3.9 |
| **TinyChemVL** | 3.6 / 91.7 | 13.9 / 94.4 | 0.0 / 74.0 | 3.5 |
| **MolScribe** | 4.8 / 7.1* | 8.3 / 69.4 | 0.0 / 26.0 | 3.1 |
| **MolGrapher** | 0.0 / 100.0 | 0.0 / 97.2 | 0.0 / 42.7 | 0.0 |

*\*Note: Unsubstituted aromatic cyclopentadienyl rings (`c1cccc1`) emitted by SMILES decoders fail default RDKit Kekulé sanitization (`Valid %`), but are correctly parsed and verified by the coordination `parts` evaluator.*

---

## 3. Key Scientific Findings

1. **Chemical Complexity Dominates Visual Style and Geometric Perturbation.**  
   Across all seven journal stylesheets and five geometric variants (in-plane rotations and perspective homography tilts), model accuracy varies by only a few percentage points (e.g., **DECIMER** standard deviation across stylesheets is $2.2\%$; **MolNexTR** is $3.0\%$; **MolScribe** is $3.2\%$). By contrast, increasing heavy-atom count beyond 60 or introducing bridged polycyclic cages and macrocycles causes accuracy to collapse by $40\text{--}80$ percentage points. Modern OCSR backbones are largely invariant to typographic conventions and mild camera perspective; **their bottleneck is chemical topology and stereochemical reasoning**.

2. **Grammar-Free 1D Representation (SELFIES) Prevents Catastrophic Syntax Collapse on Large Scaffolds.**  
   **DECIMER** leads the organic benchmark by a wide margin ($58.6\%$ exact, $71.0\%$ connectivity, $87.2\%$ mean ECFP4 Tanimoto). Its advantage is most pronounced on large molecules ($\ge 60$ heavy atoms: $35.6\%$ exact vs. $\le 12.4\%$ for all other models) and macrocycles ($41.4\%$ exact vs. $\le 22.9\%$). Whereas standard SMILES decoders and 2D graph constructors suffer severe syntactic failure on complex cages and large rings (unparseable rates reaching $31.4\%$ for **MolScribe**/**MolNexTR** and $25\text{--}40\%$ for **MolSight**/**OCSRGlyph**/**MarkushGlyph**), **DECIMER**'s SELFIES token stream guarantees syntactic validity on $93.0\%$ of organic depictions, turning potential syntax crashes into high-similarity near-misses ($72.5\%$ mean Tanimoto on skeleton errors).

3. **2D Graph Reconstruction Excels at Local Stereochemistry and Cage Topology—When the Graph Parses.**  
   Hybrid 2D graph models (**MolNexTR** $41.1\%$ and **MolScribe** $41.1\%$) tie for second place on the organic headline and outperform all non-SELFIES 1D sequence models. Notably, **MolScribe** achieves the highest exact accuracy of *all* 11 models on polycyclic cages ($22.5\%$, surpassing **DECIMER** at $14.3\%$ and **MolNexTR** at $17.8\%$) and the lowest topological skeleton error rate overall ($14.2\%$). Furthermore, **MolNexTR** ($83.8\%$) and **DECIMER** ($83.6\%$) achieve the highest conditional stereochemical precision once connectivity is recovered. However, purely bottom-up keypoint-and-GCN pipelines without end-to-end sequence recovery (**MolGrapher**, $6.1\%$ exact, $33.1\%$ connectivity) fail whenever bond crossings or macrocyclic layouts disconnect the predicted graph—emitting a single-carbon sentinel (`C`) on $25.4\%$ of the evaluation slice ($59\%$ of organometallic drawings and $88\%$ of structures exceeding 120 heavy atoms).

4. **Disentangling Alkene $E/Z$ Conventions from Tetrahedral Chirality Reveals Hidden Strengths.**  
   For several 1D sequence models (**OCSRGlyph**, **MolSight**, **MolGrapher**), stereochemical errors account for roughly half of all failures on topologically correct skeletons. However, decomposing stereochemical loss into double-bond $E/Z$ loss versus tetrahedral `@`/`@@` loss (Table 3) exposes a critical evaluation subtlety: **COMO**'s official post-processor reconstructs tetrahedral chirality from its auxiliary graph predictions while intentionally stripping cis/trans markers (`ignore_cistrans=True`). Under $E/Z$-blind tetrahedral evaluation (`Chirality %`), **COMO** surges by $+14.9$ percentage points ($35.5\% \to 50.3\%$), overtaking both **MolScribe** ($44.0\%$) and **MolNexTR** ($43.5\%$) and achieving the lowest tetrahedral stereocentre loss in the benchmark ($5.0\%$, or $91.0\%$ conditional tetrahedral precision). Similarly, **OCSRGlyph** gains $+2.9$ points ($18.0\% \to 20.9\%$) under its native chirality-kept convention.

5. **Post-Training Specialization Does Not Automatically Transfer Across Chemical Domains.**  
   Comparing paired checkpoints highlights the fragility of narrow domain adaptation:
   * **MolSight vs. MolSight-stereo**: Post-training **MolSight** with Group Relative Policy Optimization (GRPO) LoRA on 200k synthetic stereoisomers (`stereo_grpo_10_2.pth`) slightly increases syntactic validity ($74.3\% \to 76.2\%$) and compact-molecule accuracy ($49.2\% \to 52.3\%$), and boosts metallocene recognition ($27.2\% \to 37.7\%$ coordination parts), but *degrades* organic exact accuracy ($20.2\% \to 19.1\%$), stereo-dense accuracy ($8.0\% \to 6.0\%$), and behind-the-bridge occlusion recovery.
   * **OCSRGlyph vs. MarkushGlyph**: Fine-tuning a VLM for patent Markush CXSMILES parsing (**MarkushGlyph**) causes severe catastrophic forgetting on complex natural products ($5.2\%$ organic exact, $0.0\%$ on spiro, macrocyclic, and large structures).
   * **General-Purpose VLMs (TinyChemVL)**: Despite a $\sim 9\text{B}$ parameter count (InternViT-6B + Qwen2.5-3B) and the highest syntactic validity ($93.5\%$), **TinyChemVL** achieves only $14.6\%$ organic exact accuracy and $24.1\%$ connectivity, frequently hallucinating plausible drug-like SMILES with the wrong atom count (mean Tanimoto $38.7\%$ on skeleton errors; median $|\Delta N_{\text{heavy}}| = 5$ atoms vs. $1$ atom for specialized OCSR models).

6. **Metallated Macrocyclic Cavities Remain an Unsolved Frontier in OCSR.**  
   On the redrawn coordination subset, top sequence models successfully parse $\eta^5/\eta^6$ sandwich metallocenes (**DECIMER** $100.0\%$, **MolSight-stereo** $91.7\%$, **OCSRGlyph** $67.9\%$, **MolSight** $64.3\%$), and several models partially recover $\sigma$-bonded/chelate Pt, Rh, Ru, and Co complexes (**OCSRGlyph** and **MarkushGlyph** $33.3\%$; **DECIMER**, **MolParser-Mobile-V2**, and **MolSight-stereo** $25.0\%$). Yet **every single evaluated model scores $0.0\%$ on metallated macrocyclic cavities** (metalloporphyrins, chlorophylls, metallo-phthalocyanines, and cobalamins/corrins)—universal failure caused by dense central $M\text{--}N_4$ coordination bonds intersecting the conjugated macrocycle.

7. **High Cross-Model Complementarity.**  
   Across the entire evaluation slice, the oracle union of all 11 models solves **$67.6\%$** of all depictions ($72.1\%$ of organic images exact, $81.1\%$ of organic images by connectivity, and $44.7\%$ of coordination depictions by ligand-skeleton `parts`), compared to $53.4\%$ for the single best model (**DECIMER**). Seven different models contribute uniquely solved depictions that all 10 competing models miss (**DECIMER**, **MolScribe**, **TinyChemVL**, **MolNexTR**, **OCSRGlyph**, **COMO**, **MarkushGlyph**, **MolParser-Mobile-V2**, and **MolSight**), demonstrating substantial architectural complementarity between SELFIES sequence decoding, 2D graph reconstruction, and VLM visual perception.

---

## 4. Model Provenance & Evaluation Details

All models were evaluated deterministically (greedy decoding, beam size 1) using their official released checkpoints and post-processing pipelines:

* **DECIMER** (`v2.7.1`, [Kohulan/DECIMER-Image_Transformer](https://github.com/Kohulan/DECIMER-Image_Transformer), MIT): Standard TensorFlow SavedModel (`DECIMER.predict_SMILES`), decoding predicted SELFIES tokens into canonical SMILES.
* **MolNexTR** ([CYF200127/MolNexTR](https://huggingface.co/datasets/CYF200127/MolNexTR), Apache 2.0): Checkpoint `molnextr_best.pth`, scored on its official graph-to-SMILES output.
* **MolScribe** ([yujieq/MolScribe](https://huggingface.co/yujieq/MolScribe), MIT): Checkpoint `swin_base_char_aux_1m680k.pth`, scored on its official graph-to-SMILES output.
* **COMO** ([Keylab/COMO](https://huggingface.co/Keylab/COMO), CC BY-NC 4.0): Tanimoto-reward checkpoint `models/tanimoto/best.pth`. Scored on its official `postprocess` SMILES (which transfers tetrahedral stereocentres from the predicted graph and canonicalizes with cis/trans markers ignored, falling back to `decode_smiles` when graph stereo restoration fails).
* **MolParser-Mobile-V2** ([UniParser/MolParser-Mobile-V2](https://huggingface.co/UniParser/MolParser-Mobile-V2), CC BY-NC-SA 4.0): 10.00M-parameter checkpoint run in FP16 with `max_length=384`. Scored on the SMILES trunk preceding any E-SMILES `<sep>` or `|` suffix.
* **MolSight & MolSight-stereo** ([hustvl/MolSight](https://github.com/hustvl/MolSight), [Robert-zwr/MolSight](https://huggingface.co/Robert-zwr/MolSight)): Evaluated on `pubchem_uspto_smiles_edges_10.pth` (`--formats char,edges`) and the Stereo-200k GRPO LoRA checkpoint `stereo_grpo_10_2.pth` (`--lora --formats char`), with upstream functional-group expansion (`_postprocess_smiles`) applied prior to scoring.
* **OCSRGlyph** (`EdisonScientific/glyph`): Swin-B/384 + Transformer decoder evaluated with default post-processing enabled.
* **TinyChemVL** (`InternViT-6B + Qwen2.5-3B`): Evaluated in BF16 with `<image>` token splicing and in-place `num_image_token` mutation patched per batch (`tools/predict_tinychemvl.py`).
* **MolGrapher** ([DS4SD/MolGrapher](https://github.com/DS4SD/MolGrapher), MIT): Evaluated with the `gc_stereo_model` node classifier and `--assign-stereo`, patched via `tools/setup_molgrapher_wsl.py` to run unmodified checkpoints on modern PyTorch/CUDA (`torch 2.11+cu128`) with CPU PaddleOCR caption/label recognition.
* **MarkushGlyph** ([EdisonScientific/MarkushGlyph](https://huggingface.co/EdisonScientific/MarkushGlyph), Apache 2.0): LoRA adapter on `Qwen/Qwen3.5-2B-Base` prompted with `OCSR_NO_EXTENSION_PROMPT`; scored on the CXSMILES core preceding any `|` extension.

---

## 5. Repository Structure & Tooling

```text
.
├── samples/
│   ├── gallery.png                 # Contact sheet of illustrative depictions
│   ├── ground_truth.csv            # Isomeric SMILES, InChI, and structural descriptors for samples/
│   └── <image_id>.png              # Individual sample images (not part of the held-out evaluation set)
└── tools/
    ├── make_random_split.py        # Stratified, seeded evaluation slice generator
    ├── eval_ocsr.py                # Primary scoring harness (Exact, Tautomer, Connectivity, InChIKey, Parts)
    ├── eval_extra.py               # Chirality-kept, E/Z-only, ECFP4 Tanimoto, error decomposition
    ├── stats_ci.py                 # Compound-level bootstrap CIs + paired significance tests
    ├── results_source.py           # Which scored run each table column is read from
    ├── verify_reported_numbers.py  # Machine-checks this README's tables against the tools
    ├── compare_models.py           # Side-by-side multi-model comparison and agreement analyzer
    ├── predict_decimer.py          # Resumable inference driver for DECIMER 2.7.1
    ├── predict_molnextr.py         # Resumable inference driver for MolNexTR
    ├── predict_molscribe.py        # Resumable inference driver for MolScribe
    ├── predict_como.py             # Resumable inference driver for COMO
    ├── predict_molparser.py        # Resumable inference driver for MolParser-Mobile-V2
    ├── predict_molsight.py         # Resumable inference driver for MolSight & MolSight-stereo
    ├── predict_ocsrglyph.py        # Resumable inference driver for OCSRGlyph
    ├── predict_tinychemvl.py       # Resumable inference driver for TinyChemVL
    ├── predict_molgrapher.py       # Resumable/shardable inference driver for MolGrapher
    ├── setup_molgrapher_wsl.py     # Compatibility patcher for MolGrapher on modern PyTorch/CUDA
    └── predict_markushglyph.py     # Resumable inference driver for MarkushGlyph
```

### Reproducing the Tables
`tools/verify_reported_numbers.py` parses Tables 1 and 3 of this README and re-derives every
cell from `tools/eval_ocsr.py`, `tools/eval_extra.py` and `tools/stats_ci.py`; the last run
matched **154 / 154** cells. Scoring is pinned to `rdkit==2026.09.1` (see `requirements.txt`),
and switching the RDKit build is tracked in `data/eval_results/all_models/version_delta.md`:
re-scoring all 15 runs under the pinned build moved 3 of 300 numbers, all of them the
coordination part-key score on runs whose predictions predate the complex redraw.

### Running Inference & Scoring
All `predict_*.py` drivers accept a benchmark manifest (`--manifest`) and write resumable JSONL predictions (`{"image_id": "<id>", "prediction": "<SMILES>"}`):

```bash
# Score a prediction JSONL file against a local manifest
python tools/eval_ocsr.py \
    --manifest path/to/manifest.jsonl \
    --predictions preds.jsonl \
    --out results/

# Generate a side-by-side multi-model comparison report
python tools/compare_models.py \
    --manifest path/to/manifest.jsonl \
    --preds decimer=predictions/decimer.jsonl,molnextr=predictions/molnextr.jsonl,molscribe=predictions/molscribe.jsonl \
    --out results/comparison.md
```

Environment requirements for individual model drivers:
* `predict_decimer.py`: `decimer==2.7.1` (downloads TensorFlow SavedModels into `$PYSTOW_HOME/DECIMER-V2` on first import; sets `TF_ENABLE_ONEDNN_OPTS=0` to prevent graph re-tracing slowdowns).
* `predict_molnextr.py`: `molnextr`, `torch`, `timm`, `opencv-python`, `rdkit`, plus `--checkpoint` or `$MOLNEXTR_CHECKPOINT`.
* `predict_molscribe.py`: `molscribe`, `torch`, `timm`, `opencv-python`, `rdkit`, plus `--checkpoint` or `$MOLSCRIBE_CHECKPOINT`.
* `predict_como.py`: `como-ocsr`, plus `--checkpoint` or `$COMO_CHECKPOINT`.
* `predict_molparser.py`: `torch`, `transformers`, `timm`, `opencv-python`, plus a local checkout via `--model` or `$MOLPARSER_MODEL`.
* `predict_molsight.py`: checkout of `hustvl/MolSight` (`--repo-dir` or `$MOLSIGHT_REPO`) and checkpoint via `--checkpoint` or `$MOLSIGHT_CHECKPOINT` (installs an in-memory stub for the Linux-only training renderer `molsight.indigo`).
* `predict_markushglyph.py`: `glyph` checkout via `$GLYPH_REPO`, adapter directory (`--checkpoint` or `$MARKUSHGLYPH_CHECKPOINT`), and `Qwen3.5-2B-Base` (`--base-model` or `$MARKUSH_BASE_MODEL`).
* `predict_molgrapher.py`: MolGrapher environment (`--python`), checkout (`--repo` or `$MOLGRAPHER_REPO`), and checkpoints (`--checkpoints` or `$MOLGRAPHER_CHECKPOINTS`), or `--mol-dir` to convert an existing MolGrapher output directory.

---

## 6. Methodological Notes & Limitations

* **Separation of Coordination Complexes**: Because haptic and dative organometallic bonding cannot be canonically represented by a single SMILES string and is poorly normalized by standard InChI, coordination complexes are evaluated via the structural `parts` key and reported separately from the organic headline.
* **Exclusion of Homologue Mixtures**: Database records deposited as multi-component mixtures of homologues are excluded from the benchmark in favor of well-defined single-component entities.
* **Algorithmic Depiction Provenance**: All depictions are generated algorithmically from validated chemical graphs and styled/rasterized in ChemDraw rather than cropped from copyrighted journal PDFs. This guarantees ground-truth label fidelity while isolating chemical structure recognition from document-level figure segmentation.
* **Licensing**: All benchmark code, evaluation scripts, and sample files in this repository are released under the [MIT License](LICENSE). ChemDraw is commercial software of Revvity Signals. Ground-truth molecular records are derived from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) (NCBI) and cross-checked against [Wikidata](https://www.wikidata.org/).
