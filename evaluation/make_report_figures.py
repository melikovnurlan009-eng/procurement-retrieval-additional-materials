#!/usr/bin/env python3
"""Regenerate every data-driven figure in the report.

Figure numbers here are the report's own. Figures 2 (two-lane architecture) and 3 (manual
benchmark construction protocol) are schematic diagrams drawn by hand rather than computed
from data, so they are the only two this script does not produce.

    Figure 1   Corpus composition by authority class
    Figure 4   Requirement recall and complete coverage by configuration on TEST
    Figure 5   Requirement recall by TEST query category, pooled vs two-lane
    Figure 6   ROC AUC for essential gold vs remaining DEV candidates
    Figure 7   Authority calibration within the legislation lane
    Figure 8   Reranking effect on mixed-evidence completeness on DEV
    Figure 9   Final top-25-per-lane retrieval performance
    Figure 10  Cumulative requirement recall by per-lane candidate depth k
    Figure 11  Post-CE NDCG@k by evidence lane

Everything is rebuilt from artifacts in this repository, so no corpus database, Qdrant
instance or GPU is required. Several figures plot a results/ file that another script owns
rather than recomputing its numbers, so a figure can never disagree with its table:

    Figure 1      needs results/corpus_stats.json                (corpus_stats.py)
    Figures 4, 5  need  results/rq1_rq3_same_budget/             (rq1_rq3_same_budget.py)
    Figure 9      needs results/top25_per_lane_final_metrics.csv (final_performance.py)
    Figures 10,11 need  results/ir_metrics/                      (ir_metrics.py)

Run those first, or run `python evaluation/reproduce.py --all`, which orders them. A figure
whose input is missing is skipped with a message naming the script that produces it.

    python evaluation/make_report_figures.py
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

BLUE, RED, GREY, AMBER = "#2a6f97", "#d62828", "#8d99ae", "#e8a33d"

# The per-category CSV carries the benchmark's internal suite slugs; the report's Table 4
# prints readable names. Map them so the figure and the table read the same.
CATEGORY_LABELS = {
    "procedural_multi_evidence": "Procedural / multi-evidence",
    "cross_reference_multi_instrument": "Cross-reference / multi-instrument",
    "faq90": "FAQ-derived",
    "expansion_v2_official_guidance": "Official guidance expansion",
    "official_workflow_expansion_v2": "Official workflow expansion",
    "direct legal anchor": "Direct legal anchor",
    "semantic / practitioner phrasing": "Semantic / practitioner phrasing",
    "applicability / transition": "Applicability / transition",
    "vocabulary mismatch": "Vocabulary mismatch",
    "compound / multi-requirement": "Compound / multi-requirement",
    "authority / source-role sensitive": "Authority / source-role sensitive",
}
written: list[str] = []


def _save(name: str, note: str = "") -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150)
    plt.close()
    written.append(name)
    print(f"  {name}{('   ' + note) if note else ''}")


def _skip(fig: str, needs: str) -> None:
    print(f"  Figure {fig} skipped: needs {needs}")


def dev_context():
    """Cache rows, gold targets and essential-gold flags for the DEV split."""
    cache = pd.read_parquet(CANDIDATE_CACHE)
    scenarios = load_jsonl(SCENARIOS_PATH)
    gold = {r["scenario_id"]: r for r in load_jsonl(GOLD_PATH)}
    scen_by_id = {s["scenario_id"]: s for s in scenarios}
    dev_ids = {s["scenario_id"] for s in scenarios if str(s.get("split", "")).lower() == "dev"}

    targets = {}
    for sid in dev_ids:
        t = mandatory_requirements_with_targets(gold.get(sid, {}))
        if t:
            targets[sid] = t

    cache_dev = cache[cache.scenario_id.isin(dev_ids)].copy()
    essential = {sid: set().union(*[r["chunk_ids"] for r in t.values()])
                 for sid, t in targets.items()}
    cache_dev["is_essential"] = [row.chunk_id in essential.get(row.scenario_id, ())
                                 for row in cache_dev.itertuples()]
    return cache_dev, scen_by_id, targets


# ------------------------------------------------------------------------------ figure 1
def figure1_corpus_composition():
    """Corpus composition by authority class (N = 1,370 source documents)."""
    stats = RESULTS_DIR / "corpus_stats.json"
    if not stats.exists():
        return _skip("1", "results/corpus_stats.json (run corpus_stats.py)")
    counts = json.loads(stats.read_text())["documents_by_authority_class"]
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
    _save("figure01_corpus_composition.png")


# ------------------------------------------------------------------------- figures 4 & 5
# The report's Table 3 shows seven representative configurations of the nine computed.
TABLE3_CONFIGS = [
    (1, "BM25 only\n(pooled)"),
    (2, "Dense only\n(pooled)"),
    (3, "BM25 + dense\nhybrid (pooled)"),
    (6, "Pooled hybrid\n+ authority\n+ jurisdiction"),
    (7, "Two-lane,\nno authority"),
    (8, "Two-lane,\npre-CE"),
    (9, "Two-lane,\npost-CE"),
]


def figure4_configuration_comparison():
    """Requirement recall and complete coverage by configuration on TEST (50 chunks)."""
    path = RESULTS_DIR / "rq1_rq3_same_budget" / "rq1_same_budget_50_baselines.csv"
    if not path.exists():
        return _skip("4", "results/rq1_rq3_same_budget/ (run rq1_rq3_same_budget.py)")
    test = pd.read_csv(path).query("split == 'TEST'").set_index("config_id")

    labels = [lab for _, lab in TABLE3_CONFIGS]
    recall = [test.loc[i, "requirement_recall_weighted"] for i, _ in TABLE3_CONFIGS]
    coverage = [test.loc[i, "complete_coverage_mean"] for i, _ in TABLE3_CONFIGS]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = range(len(labels))
    w = 0.38
    for off, (vals, name, color) in enumerate(
        [(recall, "Requirement recall", BLUE), (coverage, "Complete coverage", AMBER)]
    ):
        pos = [i + (off - 0.5) * w for i in x]
        ax.bar(pos, vals, width=w, label=name, color=color)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    # The two-lane configurations start at index 4; mark where the architecture changes.
    ax.axvline(3.5, color=GREY, linestyle=":", linewidth=1.2)
    ax.text(3.58, 0.93, "two-lane", fontsize=8.5, color="#555")
    ax.text(3.42, 0.93, "pooled", fontsize=8.5, color="#555", ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval configuration on TEST at a matched 50-result budget\n"
                 "(94 scenarios, 139 mandatory requirements)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    _save("figure04_configuration_comparison_test.png")


def figure5_category_comparison():
    """Requirement recall by TEST query category, pooled ranking vs two-lane."""
    path = RESULTS_DIR / "rq1_rq3_same_budget" / "rq1_per_category.csv"
    if not path.exists():
        return _skip("5", "results/rq1_rq3_same_budget/ (run rq1_rq3_same_budget.py)")
    df = pd.read_csv(path).query("split == 'TEST'")

    pooled_name = next(c for c in df.config.unique() if c.startswith("pooled"))
    twolane_name = next(c for c in df.config.unique() if c.startswith("two_lane"))
    pooled = df[df.config == pooled_name].set_index("category")
    twolane = df[df.config == twolane_name].set_index("category")

    cats = sorted(pooled.index, key=lambda c: -pooled.loc[c, "n_requirements"])
    n_req = [int(pooled.loc[c, "n_requirements"]) for c in cats]
    p_vals = [pooled.loc[c, "requirement_recall_weighted"] for c in cats]
    t_vals = [twolane.loc[c, "requirement_recall_weighted"] for c in cats]
    labels = [f"{CATEGORY_LABELS.get(c, c)}  (N={n})" for c, n in zip(cats, n_req)]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    y = range(len(cats))
    h = 0.38
    ax.barh([i + h / 2 for i in y], p_vals, height=h, color=GREY, label="Pooled ranking @50")
    ax.barh([i - h / 2 for i in y], t_vals, height=h, color=BLUE, label="Two-lane @25+25")
    for i, (pv, tv) in enumerate(zip(p_vals, t_vals)):
        ax.text(pv + 0.012, i + h / 2, f"{pv:.3f}", va="center", fontsize=7.5)
        ax.text(tv + 0.012, i - h / 2, f"{tv:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("Requirement recall (requirement-weighted)")
    ax.set_title("Requirement recall by TEST query category, both at a 50-result budget\n"
                 "Categories with N < 10 requirements are descriptive only", fontsize=11)
    # Outside the axes: at N=2 the two-lane bar reaches 1.000 and would sit under a legend
    # placed in any corner.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
              frameon=False)
    _save("figure05_category_comparison_test.png")


# ------------------------------------------------------------------------------ figure 6
def figure6_first_stage_auc(cache_dev):
    """ROC AUC for essential gold versus remaining DEV candidates."""
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
    _save("figure06_first_stage_auc.png",
          f"(BM25 {aucs['BM25']:.3f}, dense {aucs['Dense (BGE-M3)']:.3f})")


# ------------------------------------------------------------------------------ figure 7
def figure7_authority_calibration(cache_dev):
    """Authority calibration: nominal prior vs mean per-query-normalised value."""
    rows = []
    for cls, nominal in NOMINAL_AUTHORITY.items():
        sub = cache_dev[cache_dev.authority_class == cls]
        if sub.empty:
            continue
        rows.append({"label": CLASS_LABELS.get(cls, cls), "nominal": nominal,
                     "normalised": float(sub.authority_norm.mean()), "cls": cls})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = range(len(df))
    h = 0.38
    ax.barh([i + h / 2 for i in y], df.nominal, height=h, color=BLUE,
            label="Nominal authority prior")
    ax.barh([i - h / 2 for i in y], df.normalised, height=h, color=RED,
            label="Mean per-query-normalised value")
    for i, r in df.iterrows():
        ax.text(r.nominal + 0.012, i + h / 2, f"{r.nominal:.2f}", va="center", fontsize=8)
        ax.text(r.normalised + 0.012, i - h / 2, f"{r.normalised:.3f}", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df.label, fontsize=9)
    ax.set_xlabel("Value")
    ax.set_xlim(0, 1.15)
    ax.set_title("Authority calibration before and after per-query normalisation")
    ax.legend(loc="lower right", fontsize=8)

    prim = df.loc[df.cls == "PRIMARY_LEGISLATION", "normalised"].iloc[0]
    sec = df.loc[df.cls == "SECONDARY_LEGISLATION", "normalised"].iloc[0]
    _save("figure07_authority_calibration.png",
          f"(primary {prim:.3f} vs secondary {sec:.3f}; {(prim - sec) / 0.03:.2f}x amplification)")


# ------------------------------------------------------------------------------ figure 8
def _dual_evidence(cache_dev, scen_by_id, targets, ce_scores, use_ce: bool):
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
            L = ranked_lane_post(cache_dev, sid, "legislation",
                                 ce_scores.get((sid, "legislation")))[:25]
            O = ranked_lane_post(cache_dev, sid, "other", ce_scores.get((sid, "other")))[:25]
        else:
            L = ranked_lane_pre(cache_dev, sid, "legislation")[:25]
            O = ranked_lane_pre(cache_dev, sid, "other")[:25]
        law_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in law.values())
        nonlaw_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in nonlaw.values())
        hit += int(law_ok and nonlaw_ok)
    return hit, total


def figure8_mixed_evidence(cache_dev, scen_by_id, targets):
    """Reranking effect on strict mixed-evidence completeness (DEV)."""
    ce = {}
    for (sid, lane), grp in cache_dev[cache_dev.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))
    pre_hit, n = _dual_evidence(cache_dev, scen_by_id, targets, ce, use_ce=False)
    post_hit, _ = _dual_evidence(cache_dev, scen_by_id, targets, ce, use_ce=True)
    vals = [pre_hit / n, post_hit / n]

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(["Pre-CE first stage", "Baseline cross-encoder"], vals,
                  color=[GREY, BLUE], width=0.55)
    for bar, v, num in zip(bars, vals, [pre_hit, post_hit]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}\n({num}/{n})", ha="center", fontsize=9.5)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Strict dual-evidence coverage")
    ax.set_title(f"Reranking effect on mixed-evidence completeness (DEV, N={n})")
    ax.text(0.5, 0.04,
            "All mandatory legislation and non-legislation requirements must be retrieved",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#444")
    _save("figure08_mixed_evidence_dev.png", f"({vals[0]:.3f} -> {vals[1]:.3f}, N={n})")


# ------------------------------------------------------------------------------ figure 9
def figure9_final_performance():
    """Final top-25-per-lane retrieval performance. Plots Table 8's own CSV."""
    table = RESULTS_DIR / "top25_per_lane_final_metrics.csv"
    if not table.exists():
        return _skip("9", "results/top25_per_lane_final_metrics.csv (run final_performance.py)")
    df = pd.read_csv(table)
    df["split"] = ["DEV" if str(s).startswith("DEV") else "TEST" for s in df["split"]]

    metrics = [
        ("RequirementRecall@25-per-lane", "Requirement recall"),
        ("CompleteCoverage@25-per-lane", "Complete coverage"),
        ("DualEvidenceCoverage@25-per-lane", "Dual-evidence coverage"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = range(len(metrics))
    w = 0.36
    for off, (split, color) in enumerate([("DEV", BLUE), ("TEST", AMBER)]):
        row = df[df.split == split].iloc[0]
        vals = [row[col] for col, _ in metrics]
        pos = [i + (off - 0.5) * w for i in x]
        n_mixed = int(row.n_mixed_evidence_scenarios)
        ax.bar(pos, vals, width=w, color=color,
               label=f"{split} (N={int(row.n_scenarios)} scenarios)")
        for i, (p, v) in enumerate(zip(pos, vals)):
            extra = f"\n(N={n_mixed})" if i == 2 else ""
            ax.text(p, v + 0.012, f"{v:.3f}{extra}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Coverage")
    ax.set_title("Final top-25-per-lane retrieval performance")
    ax.legend()
    _save("figure09_final_performance.png")


# ---------------------------------------------------------------------- figures 10 & 11
def figure10_cumulative_recall():
    """Cumulative requirement recall by per-lane candidate depth k, pre- and post-CE."""
    path = RESULTS_DIR / "ir_metrics" / "cumulative_requirement_recall_summary.csv"
    if not path.exists():
        return _skip("10", "results/ir_metrics/ (run ir_metrics.py)")
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    styles = {("DEV", "pre_CE"): (BLUE, "--", "DEV, pre-CE"),
              ("DEV", "post_CE"): (BLUE, "-", "DEV, post-CE"),
              ("TEST", "pre_CE"): (AMBER, "--", "TEST, pre-CE"),
              ("TEST", "post_CE"): (AMBER, "-", "TEST, post-CE")}
    for (split, stage), (color, ls, label) in styles.items():
        sub = df[(df.split == split) & (df.stage == stage)].sort_values("k")
        ax.plot(sub.k, sub.requirement_recall_weighted, ls, color=color,
                marker="o", markersize=3.5, label=label, linewidth=1.8)
    ax.axvline(25, color=GREY, linestyle=":", linewidth=1.2)
    ax.text(26, 0.585, "final cutoff\n25 per lane", fontsize=8, color="#555")
    ax.set_xlabel("Per-lane candidate depth k")
    ax.set_ylabel("Cumulative requirement recall")
    ax.set_title("Cumulative requirement recall by candidate depth")
    ax.set_ylim(0.55, 0.95)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25, linewidth=0.6)
    _save("figure10_cumulative_recall.png")


def figure11_ndcg_by_lane():
    """Post-CE NDCG@k by evidence lane, on both splits."""
    path = RESULTS_DIR / "ir_metrics" / "precision_ndcg_summary.csv"
    if not path.exists():
        return _skip("11", "results/ir_metrics/ (run ir_metrics.py)")
    df = pd.read_csv(path).query("stage == 'post_CE'")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    styles = {("DEV", "legislation"): (BLUE, "-", "DEV, legislation lane"),
              ("DEV", "other"): (BLUE, "--", "DEV, other-evidence lane"),
              ("TEST", "legislation"): (AMBER, "-", "TEST, legislation lane"),
              ("TEST", "other"): (AMBER, "--", "TEST, other-evidence lane")}
    for (split, lane), (color, ls, label) in styles.items():
        sub = df[(df.split == split) & (df.lane == lane)].sort_values("k")
        ax.plot(sub.k, sub.ndcg_at_k_mean, ls, color=color,
                marker="o", markersize=3.5, label=label, linewidth=1.8)
    ax.set_xlabel("Per-lane candidate depth k")
    ax.set_ylabel("NDCG@k (binary relevance)")
    ax.set_title("Post-cross-encoder NDCG by evidence lane")
    ax.set_ylim(0.2, 0.7)
    ax.legend(fontsize=8.5, loc="center right")
    ax.grid(alpha=0.25, linewidth=0.6)
    _save("figure11_ndcg_by_lane.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Regenerating the report's data-driven figures:")
    cache_dev, scen_by_id, targets = dev_context()

    figure1_corpus_composition()
    figure4_configuration_comparison()
    figure5_category_comparison()
    figure6_first_stage_auc(cache_dev)
    figure7_authority_calibration(cache_dev)
    figure8_mixed_evidence(cache_dev, scen_by_id, targets)
    figure9_final_performance()
    figure10_cumulative_recall()
    figure11_ndcg_by_lane()

    print(f"\n{len(written)} figures written to {FIGURES_DIR}")
    print("Figures 2 and 3 are schematic diagrams (two-lane architecture; benchmark "
          "construction protocol) and are not generated from data.")


if __name__ == "__main__":
    main()
