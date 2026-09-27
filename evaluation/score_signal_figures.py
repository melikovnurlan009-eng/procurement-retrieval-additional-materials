#!/usr/bin/env python3
"""Signal-diagnostic figures supporting the score/signal analysis section of the report.

Four panels, all on the DEV split and all drawn from the frozen candidate cache:
    fig1  each raw signal against its per-query normalised value (saturation behaviour)
    fig2  BM25 raw, essential gold vs non-gold
    fig3  dense cosine raw, essential gold vs non-gold vs wrong-regime gold
    fig4  cross-encoder score, essential gold vs non-gold

The authority nominal-vs-effective comparison is not drawn here: it is report Figure 5 and
is produced by evaluation/make_report_figures.py, so it has exactly one producer.

    python evaluation/score_signal_figures.py

Writes to figures/signal_diagnostics/. Requires only the shipped candidate cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.ce_metrics_audit import load_jsonl, mandatory_requirements_with_targets
from evaluation.paths import CANDIDATE_CACHE, FIGURES_DIR, GOLD_PATH, SCENARIOS_PATH

OUT_DIR = FIGURES_DIR / "signal_diagnostics"
ESSENTIAL, NON_GOLD, WRONG_REGIME = "#d62828", "#adb5bd", "#f77f00"


def load_dev_cache() -> pd.DataFrame:
    cache = pd.read_parquet(CANDIDATE_CACHE)
    scenarios = load_jsonl(SCENARIOS_PATH)
    gold = {r["scenario_id"]: r for r in load_jsonl(GOLD_PATH)}
    dev_ids = {s["scenario_id"] for s in scenarios if str(s.get("split", "")).lower() == "dev"}
    cache = cache[cache.scenario_id.isin(dev_ids)].copy()

    targets = {sid: mandatory_requirements_with_targets(g) for sid, g in gold.items()}
    essential = {
        sid: set().union(*[x["chunk_ids"] for x in t.values()]) if t else set()
        for sid, t in targets.items()
    }

    def tag(row):
        if row.chunk_id in essential.get(row.scenario_id, ()):
            return "essential_gold"
        if row.is_wrong_regime_gold:
            return "wrong_regime"
        return "supporting_gold" if row.is_gold else "non_gold"

    cache["gold_tag"] = [tag(r) for r in cache.itertuples()]
    return cache


def _hist(ax, cache, column, groups, xlabel, title, bins=40):
    for grp, color in groups:
        v = cache[cache.gold_tag == grp][column].dropna()
        ax.hist(v, bins=bins, density=True, alpha=0.55, label=f"{grp} (n={len(v):,})", color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_dev_cache()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sample = cache.sample(min(8000, len(cache)), random_state=1)
    for ax, (raw, norm, name) in zip(
        axes,
        [("bm25_raw", "bm25_norm", "BM25"),
         ("dense_raw", "dense_norm", "Dense"),
         ("authority_raw", "authority_norm", "Authority")],
    ):
        ax.scatter(sample[raw], sample[norm], s=2, alpha=0.15, color="#2a6f97")
        ax.set_xlabel(f"{name} raw")
        ax.set_ylabel(f"{name} normalised")
        ax.set_title(f"{name}: raw vs per-query normalised")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_raw_vs_normalized.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _hist(ax, cache, "bm25_raw", [("essential_gold", ESSENTIAL), ("non_gold", NON_GOLD)],
          "BM25 raw", "BM25: essential gold vs non-gold (DEV)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_bm25_gold_vs_nongold.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _hist(ax, cache, "dense_raw",
          [("essential_gold", ESSENTIAL), ("non_gold", NON_GOLD), ("wrong_regime", WRONG_REGIME)],
          "Dense cosine raw", "Dense: essential gold vs non-gold vs wrong-regime (DEV)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_dense_gold_vs_nongold_wrongregime.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _hist(ax, cache[cache.ce_score.notna()], "ce_score",
          [("essential_gold", ESSENTIAL), ("non_gold", NON_GOLD)],
          "Cross-encoder score (sigmoid)", "Cross-encoder: essential gold vs non-gold (DEV)", bins=30)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_ce_gold_vs_nongold.png", dpi=130)
    plt.close()

    print(f"wrote 4 signal-diagnostic figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
