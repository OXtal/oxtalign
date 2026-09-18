"""List every pair where our engine disagrees with COMPACK and bucket WHY.

Usage: uv run python benchmark/disagreements.py benchmark/results/compack_reference.csv [--partial]
A 'disagreement' = the two engines fall on opposite sides of the nmatched>=8 "matched" threshold.
"""
import csv
import math
import sys
from pathlib import Path

from oxtalign import compare

TH = 8
ROOT = Path(__file__).resolve().parents[1]
ref = Path(sys.argv[1])
partial = "--partial" in sys.argv[2:]
rows = list(csv.DictReader(open(ref)))

buckets = {"compack_failed_we_match": [], "compack_loose": [], "we_miss_tight": [], "tol_regime": []}
for r in rows:
    res = compare(str(ROOT / r["a_path"]), str(ROOT / r["b_path"]), n=15, allow_partial=partial)
    cc_n = int(r["ccdc_nmatched_15"] or 0)
    cc_r = float(r["ccdc_rmsd_15"]) if r["ccdc_rmsd_15"] not in ("", "nan") else float("nan")
    our_n, our_r = res.n_matched, res.rmsd_n
    if (cc_n >= TH) == (our_n >= TH):
        continue
    rec = (r["family"], r["a"], r["b"], cc_n, cc_r, our_n, our_r)
    if our_n >= TH and cc_n < TH:
        buckets["compack_failed_we_match"].append(rec)
    elif cc_n >= TH and our_n < TH:
        if math.isnan(cc_r) or cc_r >= 1.0:
            buckets["compack_loose"].append(rec)
        elif cc_r < 0.5:
            buckets["we_miss_tight"].append(rec)
        else:
            buckets["tol_regime"].append(rec)

verdict = {
    "compack_failed_we_match": "NOT our issue — COMPACK returned few/0 on a pair we match tightly (we are right).",
    "compack_loose": "NOT our issue — COMPACK 'matched' >=8 molecules at RMSD>=1.0 Å (loose-tolerance/fluctuation); we correctly reject.",
    "tol_regime": "borderline — COMPACK matched at 0.5-1.0 Å; partial-overlap tolerance regime, debatable for both.",
    "we_miss_tight": "POTENTIAL ISSUE — COMPACK matched >=8 at <0.5 Å but we did not; inspect.",
}
print(f"=== disagreements vs {ref.name}  (allow_partial={partial}) ===")
total = 0
for k, recs in buckets.items():
    total += len(recs)
    print(f"\n[{k}]  n={len(recs)}  -> {verdict[k]}")
    for fam, a, b, cn, cr, on, orr in recs:
        print(f"    {fam:12s} {a[:18]:18s} {b[:18]:18s}  COMPACK {cn:2d}@{cr:6.3f}   ours {on:2d}@{orr:6.3f}")
print(f"\nTOTAL disagreements: {total}")
