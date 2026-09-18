"""Compare oxtalign against a COMPACK reference CSV produced by fetch_and_compack.py.

Run in the project env:  uv run python benchmark/compare_to_compack.py [benchmark/results/compack_reference.csv]
SOURCE=curated / SOURCE=random restricts the report to one half of the reference set.
CIF paths in the CSVs are repo-relative; they are resolved against the repository root.

Reports, on the same experimental CIF pairs: a side-by-side table, the RMSD_15 correlation
(on pairs COMPACK matched), and same/different-form classification agreement.
"""
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from oxtalign import compare
from oxtalign.model import ComparisonResult

ROOT = Path(__file__).resolve().parents[1]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmark" / "results" / "compack_reference.csv"
SOURCE = os.environ.get("SOURCE", "")          # "curated" / "random" to score one half only
MATCH_THRESHOLD = 8          # COMPACK MATCH_FRACTION 0.5 * 15 -> "matched" if nmatched >= 8
ALLOW_PARTIAL = os.environ.get("ALLOW_PARTIAL", "0") == "1"   # set ALLOW_PARTIAL=1 to test the flag


def main():
    if not REF.exists():
        raise SystemExit(f"missing {REF}; run fetch_and_compack.py in the ccdc env first")
    rows = [r for r in csv.DictReader(open(REF)) if not SOURCE or r.get("source") == SOURCE]
    recs = []
    print(f"{'family':12s} {'a':20s} {'b':20s}  COMPACK(n,rmsd)   ours(n,rmsd)   status")
    for r in rows:
        try:
            res = compare(str(ROOT / r["a_path"]), str(ROOT / r["b_path"]), n=15,
                          allow_partial=ALLOW_PARTIAL)
        except Exception as exc:   # no whole molecules (coordination polymer): counts as no match
            res = ComparisonResult(float("nan"), 15, 0, "failed", message=f"{type(exc).__name__}")
        cc_n = int(r["ccdc_nmatched_15"] or 0)
        cc_r = float(r["ccdc_rmsd_15"]) if r["ccdc_rmsd_15"] not in ("", "nan") else float("nan")
        rec = dict(family=r["family"], a=r["a"], b=r["b"], cc_n=cc_n, cc_r=cc_r,
                   our_n=res.n_matched, our_r=res.rmsd_n, status=res.status)
        recs.append(rec)
        flag = "" if (cc_n >= MATCH_THRESHOLD) == (res.n_matched >= MATCH_THRESHOLD) else "  <-- DISAGREE"
        print(f"{r['family']:12s} {r['a'][:20]:20s} {r['b'][:20]:20s}  "
              f"{cc_n:2d} {cc_r:7.4f}     {res.n_matched:2d} {res.rmsd_n:7.4f}   {res.status:7s}{flag}")

    # RMSD correlation on pairs COMPACK matched (both produce a meaningful overlay).
    both = [(x["cc_r"], x["our_r"]) for x in recs
            if x["cc_n"] >= MATCH_THRESHOLD and x["our_n"] >= MATCH_THRESHOLD
            and not math.isnan(x["cc_r"]) and not math.isnan(x["our_r"])]
    print("\n=== summary ===")
    if len(both) >= 3:
        cc, ours = np.array([b[0] for b in both]), np.array([b[1] for b in both])
        print(f"RMSD_15 correlation on {len(both)} mutually-matched pairs: "
              f"Pearson={pearsonr(cc, ours)[0]:.3f}  Spearman={spearmanr(cc, ours)[0]:.3f}")
        print(f"mean |ours - COMPACK| RMSD = {np.mean(np.abs(cc - ours)):.4f} Å")
    else:
        print(f"only {len(both)} mutually-matched pairs (need >=3 for correlation)")

    agree = sum((x["cc_n"] >= MATCH_THRESHOLD) == (x["our_n"] >= MATCH_THRESHOLD) for x in recs)
    print(f"same/different-form classification agreement: {agree}/{len(recs)} "
          f"({100 * agree / len(recs):.0f}%)  [threshold nmatched >= {MATCH_THRESHOLD}]")
    # Highlight the golden same-form anchors.
    for x in recs:
        if x["family"] == "HOLSUX":
            print(f"  anchor {x['family']}: COMPACK rmsd_15={x['cc_r']:.4f} (n={x['cc_n']}) | "
                  f"ours={x['our_r']:.4f} (n={x['our_n']})")


if __name__ == "__main__":
    main()
