#!/usr/bin/env python3
"""Regenerate the data-driven figures in the report (Figures 1, 4, 5, 6 and 7).

Every figure is rebuilt from artifacts shipped in this repository, so no corpus database,
Qdrant instance or GPU is required. Figures 2 (two-lane architecture) and 3 (benchmark
construction protocol) are schematic diagrams drawn by hand, not computed from data, and
are therefore not produced here.

    python evaluation/make_report_figures.py

Writes PNGs to figures/. The underlying numbers are not re-emitted here: they already
live in results/, written by the analysis scripts listed in TECHNICAL_APPENDIX.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluation.ce_metrics_audit import (
    LAW_ROLES,
    NONLAW_ROLES,
    load_jsonl,
    mandatory_requirements_with_targets,
    ranked_lane_post,
    ranked_lane_pre,
    req_satisfied,
)
from evaluation.paths import (
    CANDIDATE_CACHE,
    FIGURES_DIR,
    GOLD_PATH,
    RESULTS_DIR,
    SCENARIOS_PATH,
)

# Authority classes as they appear in the report's Figure 1 and Figure 5 labels.
CLASS_LABELS = {
    "OFFICIAL_WORKFLOW": "Official workflow",
    "OFFICIAL_GOVERNMENT_GUIDANCE": "Official government guidance",
    "PROFESSIONAL_INTERPRETATION": "Professional interpretation",
    "NON_AUTHORITATIVE_PROFESSIONAL": "Non-authoritative professional",
    "PRIMARY_LEGISLATION": "Primary legislation",
    "SECONDARY_LEGISLATION": "Secondary legislation",
    "OFFICIAL_REGULATOR_GUIDANCE": "Official regulator guidance",
    "OFFICIAL_TECHNICAL_GUIDANCE": "Official technical guidance",
}
NOMINAL_AUTHORITY = {
    "PRIMARY_LEGISLATION": 1.00,
    "SECONDARY_LEGISLATION": 0.97,
    "OFFICIAL_GOVERNMENT_GUIDANCE": 0.88,
    "OFFICIAL_WORKFLOW": 0.78,
    "PROFESSIONAL_INTERPRETATION": 0.62,
}
BLUE, RED, GREY = "#2a6f97", "#d62828", "#8d99ae"


def _setup():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _dev_context():
    """Cache rows, gold targets and essential-gold flags for the DEV split."""
    cache = pd.read_parquet(CANDIDATE_CACHE)
    scenarios = load_jsonl(SCENARIOS_PATH)
    gold = {r["scenario_id"]: r for r in load_jsonl(GOLD_PATH)}
    scen_by_id = {s["scenario_id"]: s for s in scenarios}
    dev_ids = {s["scenario_id"] for s in scenarios if str(s.get("split", "")).lower() == "dev"}
    cache_dev = cache[cache.scenario_id.isin(dev_ids)].copy()

    targets = {}
    for sid in dev_ids:
        t = mandatory_requirements_with_targets(gold.get(sid, {}))
        if t:
            targets[sid] = t
    essential = {sid: set().union(*[r["chunk_ids"] for r in t.values()]) for sid, t in targets.items()}
    cache_dev["is_essential"] = [
        row.chunk_id in essential.get(row.scenario_id, ()) for row in cache_dev.itertuples()
    ]
    return cache, cache_dev, scenarios, scen_by_id, targets


def figure1_corpus_composition():
    """Figure 1: corpus composition by authority class (N = 1,370 source documents)."""
    stats_path = RESULTS_DIR / "corpus_stats.json"
    if not stats_path.exists():
        print("  Figure 1 skipped: run evaluation/corpus_stats.py --db <corpus> first")
        return
    counts = json.loads(stats_path.read_text())["documents_by_authority_class"]
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [CLASS_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(labels[::-1], values[::-1], color=BLUE)
    for bar, v in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.012, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9)
    ax.set_xlabel("Number of source documents")
    ax.set_title(f"Corpus composition by authority class (N = {sum(values):,} documents)")
    ax.set_xlim(0, max(values) * 1.12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_corpus_composition.png", dpi=150)
    plt.close()
    print("  Figure 1 written")


def figure4_first_stage_auc(cache_dev):
    """Figure 4: first-stage relevance discrimination on DEV (ROC AUC)."""
    y = cache_dev["is_essential"].astype(int)
    aucs = {
        "BM25": roc_auc_score(y, cache_dev.bm25_raw.fillna(cache_dev.bm25_raw.min())),
        "Dense (BGE-M3)": roc_auc_score(y, cache_dev.dense_raw.fillna(cache_dev.dense_raw.min())),
    }
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(list(aucs), list(aucs.values()), color=BLUE, width=0.55)
    for bar, v in zip(bars, aucs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("ROC AUC: essential gold vs. remaining candidates")
    ax.set_title("First-stage relevance discrimination on DEV")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_first_stage_auc.png", dpi=150)
    plt.close()
    print(f"  Figure 4 written (BM25 {aucs['BM25']:.3f}, dense {aucs['Dense (BGE-M3)']:.3f})")


def figure5_authority_normalisation(cache_dev):
    """Figure 5: nominal authority weights vs. mean per-query-normalised values."""
    rows = []
    for cls, nominal in NOMINAL_AUTHORITY.items():
        sub = cache_dev[cache_dev.authority_class == cls]
        if sub.empty:
            continue
        rows.append(
            {
                "authority_class": cls,
                "label": CLASS_LABELS.get(cls, cls),
                "nominal_weight": nominal,
                "mean_normalised_value": round(float(sub.authority_norm.mean()), 4),
                "n_candidate_rows": len(sub),
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = range(len(df))
    h = 0.38
    ax.barh([i + h / 2 for i in y], df.nominal_weight, height=h, color=BLUE, label="Nominal authority weight")
    ax.barh([i - h / 2 for i in y], df.mean_normalised_value, height=h, color=RED,
            label="Mean per-query-normalised value")
    for i, r in df.iterrows():
        ax.text(r.nominal_weight + 0.012, i + h / 2, f"{r.nominal_weight:.2f}", va="center", fontsize=8)
        ax.text(r.mean_normalised_value + 0.012, i - h / 2, f"{r.mean_normalised_value:.3f}",
                va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df.label, fontsize=9)
    ax.set_xlabel("Value")
    ax.set_xlim(0, 1.15)
    ax.set_title("Authority weighting before and after per-query normalisation")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure5_authority_normalisation.png", dpi=150)
    plt.close()

    prim = df.loc[df.authority_class == "PRIMARY_LEGISLATION", "mean_normalised_value"].iloc[0]
    sec = df.loc[df.authority_class == "SECONDARY_LEGISLATION", "mean_normalised_value"].iloc[0]
    print(f"  Figure 5 written (primary {prim:.3f} vs secondary {sec:.3f}; "
          f"amplification {(prim - sec) / 0.03:.2f}x)")


def _dual_evidence(cache_dev, scen_by_id, targets, ce_scores, use_ce: bool) -> tuple[int, int]:
    """Strict dual-evidence coverage at 25/lane: every mandatory requirement on both sides."""
    hit = total = 0
    for sid, treqs in targets.items():
        if not scen_by_id[sid].get("is_mixed_evidence"):
            continue
        law = {r: t for r, t in treqs.items() if t["roles"] & LAW_ROLES}
        nonlaw = {r: t for r, t in treqs.items() if t["roles"] & NONLAW_ROLES}
        if not (law and nonlaw):
            continue
        total += 1
        if use_ce:
            L = ranked_lane_post(cache_dev, sid, "legislation", ce_scores.get((sid, "legislation")))[:25]
            O = ranked_lane_post(cache_dev, sid, "other", ce_scores.get((sid, "other")))[:25]
        else:
            L = ranked_lane_pre(cache_dev, sid, "legislation")[:25]
            O = ranked_lane_pre(cache_dev, sid, "other")[:25]
        law_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in law.values())
        nonlaw_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in nonlaw.values())
        hit += int(law_ok and nonlaw_ok)
    return hit, total


def figure6_dual_evidence(cache_dev, scen_by_id, targets):
    """Figure 6: reranking effect on strict mixed-evidence completeness (DEV)."""
    ce = {}
    for (sid, lane), grp in cache_dev[cache_dev.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    pre_hit, n = _dual_evidence(cache_dev, scen_by_id, targets, ce, use_ce=False)
    post_hit, _ = _dual_evidence(cache_dev, scen_by_id, targets, ce, use_ce=True)
    vals = [pre_hit / n, post_hit / n]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(["Pre-CE first stage", "Baseline cross-encoder"], vals, color=[GREY, BLUE], width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Strict dual-evidence coverage")
    ax.set_title(f"Reranking effect on mixed-evidence completeness (DEV, N={n})")
    ax.text(0.5, 0.06, "All mandatory legislation and non-legislation requirements must be retrieved",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#444")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure6_dual_evidence_dev.png", dpi=150)
    plt.close()
    print(f"  Figure 6 written (pre-CE {vals[0]:.3f} -> post-CE {vals[1]:.3f}, N={n})")


def figure7_final_performance(cache, scenarios, scen_by_id):
    """Figure 7 / Table 5: final top-25-per-lane performance on both splits."""
    gold = {r["scenario_id"]: r for r in load_jsonl(GOLD_PATH)}
    ce = {}
    for (sid, lane), grp in cache[cache.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    rows = []
    for split in ("dev", "test"):
        ids = {s["scenario_id"] for s in scenarios if str(s.get("split", "")).lower() == split}
        sub = cache[cache.scenario_id.isin(ids)]
        targets = {}
        for sid in ids:
            t = mandatory_requirements_with_targets(gold.get(sid, {}))
            if t:
                targets[sid] = t

        n_sat = n_tot = cc = 0
        for sid, treqs in targets.items():
            L = ranked_lane_post(sub, sid, "legislation", ce.get((sid, "legislation")))[:25]
            O = ranked_lane_post(sub, sid, "other", ce.get((sid, "other")))[:25]
            ok_all = True
            for t in treqs.values():
                n_tot += 1
                if req_satisfied(t["chunk_ids"], L, O, 25, 25):
                    n_sat += 1
                else:
                    ok_all = False
            cc += int(ok_all)
        d_hit, d_n = _dual_evidence(sub, scen_by_id, targets, ce, use_ce=True)
        rows.append(
            {
                "split": split.upper(),
                "n_scenarios": len(targets),
                "n_requirements": n_tot,
                "requirement_recall": round(n_sat / n_tot, 4),
                "complete_coverage": round(cc / len(targets), 4),
                "dual_evidence_coverage": round(d_hit / d_n, 4),
                "n_mixed_evidence_scenarios": d_n,
            }
        )

    df = pd.DataFrame(rows)

    metrics = [
        ("requirement_recall", "Requirement recall"),
        ("complete_coverage", "Complete coverage"),
        ("dual_evidence_coverage", "Dual-evidence coverage"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = range(len(metrics))
    w = 0.36
    for off, (split, color) in enumerate([("DEV", BLUE), ("TEST", "#e8a33d")]):
        r = df[df.split == split].iloc[0]
        vals = [r[m] for m, _ in metrics]
        pos = [i + (off - 0.5) * w for i in x]
        ax.bar(pos, vals, width=w, label=split, color=color)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Coverage")
    ax.set_title("Final top-25-per-lane retrieval performance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure7_final_performance.png", dpi=150)
    plt.close()
    print("  Figure 7 written")
    print(df.to_string(index=False))


def main() -> None:
    _setup()
    print("Regenerating report figures from shipped artifacts:")
    cache, cache_dev, scenarios, scen_by_id, targets = _dev_context()
    figure1_corpus_composition()
    figure4_first_stage_auc(cache_dev)
    figure5_authority_normalisation(cache_dev)
    figure6_dual_evidence(cache_dev, scen_by_id, targets)
    figure7_final_performance(cache, scenarios, scen_by_id)
    print(f"\nFigures written to {FIGURES_DIR}")
    print("Figures 2 and 3 are schematic diagrams (architecture; benchmark protocol) and are "
          "not generated from data.")


if __name__ == "__main__":
    main()
