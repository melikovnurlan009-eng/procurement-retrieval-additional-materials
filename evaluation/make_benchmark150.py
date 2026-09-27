#!/usr/bin/env python3
"""Reconstruct the earlier 150-scenario benchmark view from the shipped 208-scenario one.

Two sections of the report - the regime-compatibility experiment (Section 6.2.2) and the
graph-expansion comparison (Section 6.2.3) - were computed on an earlier, smaller version of
the benchmark: the 150 scenarios collected in the first two rounds, under their own
stratified DEV/TEST split. The later 58-scenario expansion round, and the re-split that came
with it, post-date those two experiments.

Nothing is lost, because the 208-scenario file carries both. Each scenario records its
collection round in `source_set`, and each of the original 150 also records its split under
that earlier protocol in `source_split_150rebalance`. This script materialises that view:
the same records, filtered to the original 150, with `split` set to the earlier split, so
the ablation scripts can run against it unchanged.

That is why the earlier benchmark is not shipped as a second file. It is a view of the one
that is shipped, derived by a script you can read, rather than a near-duplicate 400 KB of
JSON whose relationship to the real benchmark you would have to take on trust.

    python evaluation/make_benchmark150.py

Writes benchmark/derived/scenarios_all_150.jsonl and benchmark/derived/gold_evidence_150.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.paths import BENCHMARK_DIR, GOLD_PATH, SCENARIOS_PATH

ORIGINAL_ROUNDS = {"current60", "faq90"}
SPLIT_FIELD = "source_split_150rebalance"


def main() -> None:
    scenarios = [json.loads(l) for l in open(SCENARIOS_PATH, encoding="utf-8") if l.strip()]
    gold = [json.loads(l) for l in open(GOLD_PATH, encoding="utf-8") if l.strip()]

    kept = []
    for s in scenarios:
        if s.get("source_set") not in ORIGINAL_ROUNDS:
            continue
        split = s.get(SPLIT_FIELD)
        if not split:
            raise SystemExit(
                f"{s['scenario_id']} is from the original rounds but carries no "
                f"{SPLIT_FIELD}; the earlier split cannot be reconstructed."
            )
        row = dict(s)
        row["split"] = split
        row["split_208"] = s.get("split")     # keep the later split, so nothing is discarded
        kept.append(row)

    kept_ids = {s["scenario_id"] for s in kept}
    gold_kept = [g for g in gold if g["scenario_id"] in kept_ids]

    out_dir = BENCHMARK_DIR / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    scen_out = out_dir / "scenarios_all_150.jsonl"
    gold_out = out_dir / "gold_evidence_150.jsonl"
    scen_out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in kept))
    gold_out.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gold_kept))

    n_dev = sum(1 for s in kept if s["split"] == "dev")
    n_test = sum(1 for s in kept if s["split"] == "test")
    print(f"Reconstructed the earlier benchmark view: {len(kept)} scenarios "
          f"({n_dev} dev / {n_test} test), {len(gold_kept)} gold records")
    missing = kept_ids - {g["scenario_id"] for g in gold_kept}
    if missing:
        print(f"  scenarios with no gold: {sorted(missing)}")
    print(f"wrote {scen_out}")
    print(f"wrote {gold_out}")


if __name__ == "__main__":
    main()
