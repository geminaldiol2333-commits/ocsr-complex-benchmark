# -*- coding: utf-8 -*-
"""
Confidence intervals and pairwise significance for the headline scores.

    python tools/stats_ci.py

Reads the scored runs through ``tools/results_source.py`` (so the four models
whose coordination column comes from a re-run are handled the same way as in
the published table) and writes into ``data/eval_results/all_models/``:

    summary_ci.csv   per model and metric: value, 95 % CI, compounds, weights
    pairwise.csv     every model pair: delta, CI of the delta, bootstrap p,
                     Holm p, and a McNemar test on the image level
    summary_ci.md    the same as tables, with the method written out

Method
------
The resampling unit is the **compound**, not the image: the six drawings of one
compound are strongly correlated, so bootstrapping images would produce CIs that
are far too narrow. Each compound contributes its mean per-image score and its
headline weight (0.25 for compact organics, 1 otherwise); the statistic is the
weighted mean over compounds - the number `eval_ocsr.py` reports.

* 10,000 bootstrap resamples of compounds (seed 42), percentile 95 % CI.
* Pairwise comparisons reuse the **same** resample indices for both models
  (paired bootstrap), so the interval of the difference accounts for both models
  being scored on the same images.
* The bootstrap p-value is two-sided and percentile based, floored at
  1/(n_resamples + 1).
* Holm correction across all model pairs.
* McNemar (image level, continuity corrected) is reported next to it as a second
  opinion; it treats images as independent, so it is optimistic.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import results_source as source  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.join(ROOT, "data", "eval_results")
OUT_DIR = os.path.join(EVAL_ROOT, "all_models")

ORGANIC_FIELDS = ["exact", "tautomer", "connectivity"]
N_RESAMPLES = 10000
SEED = 42


def arrays(rows, subset, field, unit_weight=False):
    compounds, rates, weights = source.compound_table(rows, subset, field, unit_weight)
    return compounds, np.array(rates, dtype=float), np.array(weights, dtype=float)


def weighted_mean(rates, weights):
    return float((rates * weights).sum() / weights.sum())


def bootstrap(rates_matrix, weights, n_resamples=N_RESAMPLES, seed=SEED):
    """rates_matrix: (compounds, models) -> (models, n_resamples) weighted means."""
    rng = np.random.default_rng(seed)
    n_compounds, n_models = rates_matrix.shape
    index = rng.integers(0, n_compounds, size=(n_resamples, n_compounds))
    sampled_weights = weights[index]
    weight_sums = sampled_weights.sum(axis=1)
    draws = np.empty((n_models, n_resamples))
    for model_index in range(n_models):
        sampled_rates = rates_matrix[index, model_index]
        draws[model_index] = (sampled_rates * sampled_weights).sum(axis=1) / weight_sums
    return draws


def percentile_ci(draws, alpha=0.05):
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def bootstrap_p(delta_draws):
    n = len(delta_draws)
    below = float(np.mean(delta_draws <= 0))
    above = float(np.mean(delta_draws >= 0))
    return max(min(2 * min(below, above), 1.0), 1 / (n + 1))


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def mcnemar(rows_a, rows_b, field="exact"):
    lookup = {row["image_id"]: bool(row.get(field)) for row in rows_b}
    only_a = only_b = 0
    for row in rows_a:
        other = lookup.get(row["image_id"])
        if other is None:
            continue
        if row.get(field) and not other:
            only_a += 1
        elif other and not row.get(field):
            only_b += 1
    discordant = only_a + only_b
    if discordant == 0:
        return only_a, only_b, 1.0
    from math import erfc, sqrt
    chi2 = (abs(only_a - only_b) - 1) ** 2 / discordant
    return only_a, only_b, erfc(sqrt(chi2 / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    labels = [label for label, _, _ in source.MODELS]
    rows_by_model = {label: source.load_model_rows(label) for label in labels}

    # ---- organic: weighted, over compounds -----------------------------------
    compounds_ref = weights_ref = None
    rates = {}
    for label in labels:
        compounds, weights = None, None
        for field in ORGANIC_FIELDS:
            field_compounds, field_rates, field_weights = arrays(
                rows_by_model[label], "organic", field)
            if compounds_ref is None:
                compounds_ref, weights_ref = field_compounds, field_weights
            elif not np.array_equal(field_compounds, compounds_ref):
                raise SystemExit(f"compound set mismatch for {label}/{field}")
            rates.setdefault(label, {})[field] = field_rates

    matrix = np.column_stack([rates[label]["exact"] for label in labels])
    exact_draws = bootstrap(matrix, weights_ref, args.n_resamples, args.seed)

    summary_rows = []
    for model_index, label in enumerate(labels):
        low, high = percentile_ci(exact_draws[model_index])
        summary_rows.append({
            "model": label,
            "metric": "organic exact (weighted)",
            "value_%": round(100 * weighted_mean(rates[label]["exact"], weights_ref), 2),
            "ci_low_%": round(100 * low, 2),
            "ci_high_%": round(100 * high, 2),
            "ci_width_pp": round(100 * (high - low), 2),
            "compounds": int(len(compounds_ref)),
            "weight_sum": round(float(weights_ref.sum()), 2),
        })
        for field in ORGANIC_FIELDS[1:]:
            draws = bootstrap(rates[label][field].reshape(-1, 1), weights_ref,
                              args.n_resamples, args.seed)[0]
            field_low, field_high = percentile_ci(draws)
            summary_rows.append({
                "model": label,
                "metric": f"organic {field} (weighted)",
                "value_%": round(100 * weighted_mean(rates[label][field], weights_ref), 2),
                "ci_low_%": round(100 * field_low, 2),
                "ci_high_%": round(100 * field_high, 2),
                "ci_width_pp": round(100 * (field_high - field_low), 2),
                "compounds": int(len(compounds_ref)),
                "weight_sum": round(float(weights_ref.sum()), 2),
            })

    # ---- coordination parts: unweighted over compounds ----------------------
    coordination_matrix = []
    coordination_weights = None
    for label in labels:
        compounds, coordination_rates, coordination_w = arrays(
            rows_by_model[label], "coordination", "parts", unit_weight=True)
        coordination_matrix.append(coordination_rates)
        coordination_weights = coordination_w
    coordination_draws = bootstrap(np.column_stack(coordination_matrix),
                                   coordination_weights, args.n_resamples, args.seed)
    coordination_values = {}
    for model_index, label in enumerate(labels):
        low, high = percentile_ci(coordination_draws[model_index])
        value = float(np.mean(coordination_matrix[model_index]))
        coordination_values[label] = value
        summary_rows.append({
            "model": label,
            "metric": "coordination parts",
            "value_%": round(100 * value, 2),
            "ci_low_%": round(100 * low, 2),
            "ci_high_%": round(100 * high, 2),
            "ci_width_pp": round(100 * (high - low), 2),
            "compounds": int(len(coordination_weights)),
            "weight_sum": round(float(coordination_weights.sum()), 2),
        })

    # ---- pairwise -----------------------------------------------------------
    pairs = []
    organic_values = {label: weighted_mean(rates[label]["exact"], weights_ref)
                      for label in labels}
    for i, j in itertools.combinations(range(len(labels)), 2):
        delta_draws = exact_draws[i] - exact_draws[j]
        low, high = percentile_ci(delta_draws)
        only_a, only_b, p_mcnemar = mcnemar(rows_by_model[labels[i]],
                                            rows_by_model[labels[j]])
        pairs.append({
            "model_a": labels[i],
            "model_b": labels[j],
            "delta_pp": round(100 * (organic_values[labels[i]] - organic_values[labels[j]]), 2),
            "ci_low_pp": round(100 * low, 2),
            "ci_high_pp": round(100 * high, 2),
            "p_bootstrap": bootstrap_p(delta_draws),
            "only_a": only_a,
            "only_b": only_b,
            "p_mcnemar": p_mcnemar,
        })
    adjusted = holm([pair["p_bootstrap"] for pair in pairs])
    for pair, value in zip(pairs, adjusted):
        pair["p_holm"] = value
        pair["significant_5pct"] = bool(value < 0.05)
    pairs.sort(key=lambda pair: pair["p_holm"])

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary_ci.csv"), "w", encoding="utf-8") as fh:
        fields = ["model", "metric", "value_%", "ci_low_%", "ci_high_%", "ci_width_pp",
                  "compounds", "weight_sum"]
        fh.write(",".join(fields) + "\n")
        for row in summary_rows:
            fh.write(",".join(str(row[field]) for field in fields) + "\n")

    with open(os.path.join(OUT_DIR, "pairwise.csv"), "w", encoding="utf-8") as fh:
        fields = ["model_a", "model_b", "delta_pp", "ci_low_pp", "ci_high_pp",
                  "p_bootstrap", "p_holm", "significant_5pct", "only_a", "only_b",
                  "p_mcnemar"]
        fh.write(",".join(fields) + "\n")
        for pair in pairs:
            fh.write(",".join(str(pair[field]) for field in fields) + "\n")

    # ---- markdown -----------------------------------------------------------
    def fmt(value):
        return f"{value:.1f}"

    by_model = {}
    for row in summary_rows:
        by_model.setdefault(row["model"], {})[row["metric"]] = row

    lines = ["# Headline scores with 95 % confidence intervals", "",
             f"{args.n_resamples:,} compound-level bootstrap resamples (seed {args.seed}), "
             "weights 0.25 for compact organics and 1 otherwise. Resampling the compound "
             "rather than the image is what makes the intervals honest: the six drawings "
             "of one compound are strongly correlated.", "",
             "## Organic (weighted, 271 compounds)", "",
             "| model | exact % | 95 % CI | tautomer % | CI | connectivity % | CI |",
             "|---|---:|---|---:|---|---:|---|"]
    for label in sorted(labels, key=lambda l: -by_model[l]["organic exact (weighted)"]["value_%"]):
        exact = by_model[label]["organic exact (weighted)"]
        taut = by_model[label]["organic tautomer (weighted)"]
        conn = by_model[label]["organic connectivity (weighted)"]
        lines.append(
            f"| **{label}** | {fmt(exact['value_%'])} | "
            f"[{fmt(exact['ci_low_%'])}, {fmt(exact['ci_high_%'])}] | "
            f"{fmt(taut['value_%'])} | [{fmt(taut['ci_low_%'])}, {fmt(taut['ci_high_%'])}] | "
            f"{fmt(conn['value_%'])} | [{fmt(conn['ci_low_%'])}, {fmt(conn['ci_high_%'])}] |")

    lines += ["", "## Coordination (38 compounds, part-key)", "",
              "| model | parts % | 95 % CI |", "|---|---:|---|"]
    for label in sorted(labels, key=lambda l: -coordination_values[l]):
        row = by_model[label]["coordination parts"]
        lines.append(f"| **{label}** | {fmt(row['value_%'])} | "
                     f"[{fmt(row['ci_low_%'])}, {fmt(row['ci_high_%'])}] |")

    significant = [pair for pair in pairs if pair["significant_5pct"]]
    lines += ["", f"## Pairwise (paired bootstrap, Holm over {len(pairs)} pairs)", "",
              f"{len(significant)} of {len(pairs)} pairs are significant at 5 % after "
              "Holm correction.", "",
              "| pair | delta exact (pp) | 95 % CI of delta | p (bootstrap) | p (Holm) | "
              "significant | McNemar a-only/b-only |",
              "|---|---:|---|---:|---:|---|---|"]
    for pair in pairs:
        lines.append(
            f"| {pair['model_a']} − {pair['model_b']} | {pair['delta_pp']:+.1f} | "
            f"[{pair['ci_low_pp']:+.1f}, {pair['ci_high_pp']:+.1f}] | "
            f"{pair['p_bootstrap']:.4f} | {pair['p_holm']:.4f} | "
            f"{'yes' if pair['significant_5pct'] else 'no'} | "
            f"{pair['only_a']}/{pair['only_b']} |")

    ties = [pair for pair in pairs if abs(pair["delta_pp"]) < 0.05 and pair["significant_5pct"]]
    lines += ["", "## Sanity checks", "",
              "- No pair with a delta below 0.05 pp is flagged significant."
              if not ties else
              f"- {len(ties)} tie pairs are flagged significant, check them by hand.",
              f"- Widest CI: {max(row['ci_width_pp'] for row in summary_rows):.1f} pp "
              f"({sum(1 for row in summary_rows if row['ci_width_pp'] > 12)} intervals "
              "wider than 12 pp).",
              "- McNemar columns are the discordant image counts (a-only / b-only). That "
              "test treats images as independent, so it is optimistic and is only here as "
              "a cross-check."]

    with open(os.path.join(OUT_DIR, "summary_ci.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"wrote {OUT_DIR}/summary_ci.csv, pairwise.csv, summary_ci.md")
    print(f"  models {len(labels)}, organic compounds {len(compounds_ref)}, "
          f"coordination compounds {len(coordination_weights)}")
    print(f"  significant pairs after Holm: {len(significant)}/{len(pairs)}")


if __name__ == "__main__":
    main()
