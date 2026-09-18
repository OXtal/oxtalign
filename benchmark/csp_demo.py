"""CSP use case: 30 predicted CAPRYL clusters (mmCIF, Cartesian, no cell) vs each other and vs the
experimental structure. Writes the all-pairs RMSD_15 matrix and the per-sample match to experiment.

    uv run python benchmark/csp_demo.py        # -> benchmark/results/csp_demo_matrix.csv, csp_demo_vs_experiment.csv
"""
import csv
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from oxtalign import compare, load  # noqa: E402
from oxtalign.batch import distance_matrix  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmark" / "results"


def main():
    preds = sorted((REPO / "tests/data/predicted").glob("CAPRYL_seed*_sample_0.cif"),
                   key=lambda p: int(p.stem.split("seed")[1].split("_")[0]))
    exp = load(str(REPO / "tests/data/experimental/CAPRYL.cif"))
    M, results, structs = distance_matrix([str(p) for p in preds], n=15, n_jobs=8)
    RESULTS.mkdir(exist_ok=True)
    names = [p.stem.replace("_sample_0", "") for p in preds]
    with open(RESULTS / "csp_demo_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", *names])
        for name, row in zip(names, M, strict=True):
            w.writerow([name, *[("inf" if np.isinf(v) else f"{v:.4f}") for v in row]])
    with open(RESULTS / "csp_demo_vs_experiment.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample", "n_matched", "rmsd_15", "status"])
        w.writeheader()
        for name, s in zip(names, structs, strict=True):
            r = compare(exp, s, n=15)
            w.writerow({"sample": name, "n_matched": r.n_matched, "rmsd_15": r.rmsd_n, "status": r.status})
    print("wrote", RESULTS / "csp_demo_matrix.csv", "and csp_demo_vs_experiment.csv")


if __name__ == "__main__":
    main()
