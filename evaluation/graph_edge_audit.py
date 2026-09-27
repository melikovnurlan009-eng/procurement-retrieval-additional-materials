#!/usr/bin/env python3
"""Stratified audit of the citation graph's edge quality (report Section 6.2.3).

122 edges were sampled across 14 strata - one stratum per (relation, derivation method)
combination - and each was judged against the underlying legal text on two criteria:

    MEANINGFUL            the edge is legally correct AND carries retrieval value
    CORRECT_BUT_USELESS   the edge is legally correct but adds nothing to retrieval
    WRONG                 the edge does not hold
    UNSURE                could not be decided from the text

Two rates follow: the strict one (MEANINGFUL alone) and the permissive one
(MEANINGFUL + CORRECT_BUT_USELESS, i.e. not wrong). Both are reported, per stratum and
overall, along with the per-relation breakdown.

    python evaluation/graph_edge_audit.py

Reads benchmark/graph_edge_review/verdicts.csv. Writes results/graph_edge_audit.json and
results/graph_edge_audit_by_stratum.csv.

The verdicts themselves are shipped as data in benchmark/graph_edge_review/verdicts.csv, one
row per edge; this script aggregates them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evaluation.paths import BENCHMARK_DIR, RESULTS_DIR

VERDICTS = ["MEANINGFUL", "CORRECT_BUT_USELESS", "WRONG", "UNSURE"]


def main() -> None:
    src = BENCHMARK_DIR / "graph_edge_review" / "verdicts.csv"
    if not src.exists():
        raise SystemExit(f"Edge verdicts not found at {src}")
    df = pd.read_csv(src)

    unfilled = df[~df.verdict.isin(VERDICTS)]
    if len(unfilled):
        raise SystemExit(f"{len(unfilled)} rows carry no usable verdict; audit is incomplete")

    counts = {v: int((df.verdict == v).sum()) for v in VERDICTS}
    n = len(df)
    strict = counts["MEANINGFUL"] / n
    not_wrong = (counts["MEANINGFUL"] + counts["CORRECT_BUT_USELESS"]) / n

    by_stratum = (
        df.assign(**{v: (df.verdict == v).astype(int) for v in VERDICTS})
          .groupby("stratum", sort=False)
          .agg(n=("item", "count"), **{v: (v, "sum") for v in VERDICTS})
          .reset_index()
    )
    by_stratum["meaningful_rate_pct"] = (100 * by_stratum.MEANINGFUL / by_stratum.n).round(1)
    by_stratum["not_wrong_rate_pct"] = (
        100 * (by_stratum.MEANINGFUL + by_stratum.CORRECT_BUT_USELESS) / by_stratum.n
    ).round(1)
    by_stratum = by_stratum.sort_values("meaningful_rate_pct", ascending=False)

    by_relation = (
        df.assign(**{v: (df.verdict == v).astype(int) for v in VERDICTS})
          .groupby("relation", sort=False)
          .agg(n=("item", "count"), meaningful=("MEANINGFUL", "sum"))
          .reset_index()
    )
    by_relation["meaningful_rate_pct"] = (
        100 * by_relation.meaningful / by_relation.n).round(1)

    summary = {
        "n_edges_reviewed": n,
        "n_strata": int(df.stratum.nunique()),
        "verdict_counts": counts,
        "strict_meaningful_rate": round(strict, 4),
        "not_wrong_rate": round(not_wrong, 4),
        "best_strata_meaningful_rate_pct": by_stratum.head(4)[
            ["stratum", "n", "meaningful_rate_pct"]].to_dict("records"),
        "worst_strata_meaningful_rate_pct": by_stratum.tail(4)[
            ["stratum", "n", "meaningful_rate_pct"]].to_dict("records"),
        "by_relation": by_relation.to_dict("records"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "graph_edge_audit.json").write_text(json.dumps(summary, indent=2))
    by_stratum.to_csv(RESULTS_DIR / "graph_edge_audit_by_stratum.csv", index=False)

    print(f"Stratified graph edge audit: {n} edges across {summary['n_strata']} strata\n")
    for v in VERDICTS:
        print(f"  {v:22} {counts[v]:4}   {100 * counts[v] / n:5.1f}%")
    print(f"\n  strict (MEANINGFUL only)          {counts['MEANINGFUL']}/{n} = {strict:.3f}")
    print(f"  permissive (not WRONG)            "
          f"{counts['MEANINGFUL'] + counts['CORRECT_BUT_USELESS']}/{n} = {not_wrong:.3f}")
    print("\nBy stratum (strict meaningful rate):")
    for row in by_stratum.itertuples():
        print(f"  {row.meaningful_rate_pct:5.1f}%  n={row.n:3}  {row.stratum}")
    print(f"\nwrote {RESULTS_DIR / 'graph_edge_audit.json'} and "
          f"{RESULTS_DIR / 'graph_edge_audit_by_stratum.csv'}")


if __name__ == "__main__":
    main()
