"""oxtalign timing / parallel-scaling / shell-size runs, mirroring compack_bench.py. Writes CSVs under
benchmark/results/.

    uv run python benchmark/oxtalign_bench.py timing  --shell 15
    uv run python benchmark/oxtalign_bench.py scaling --shell 15 --workers 1,2,4,8,16,32,64
    uv run python benchmark/oxtalign_bench.py sweep   --shells 1,2,3,5,8,10,15,20,25,30,40,50,75,100,150
    uv run python benchmark/oxtalign_bench.py select-sweep-pairs        # writes results/sweep_pairs.csv

Per-pair times cover ``compare_structures`` only (structures preloaded), matching the COMPACK runs.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from oxtalign import load  # noqa: E402
from oxtalign.matching import compare_structures  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmark" / "results"
_STRUCTS: dict = {}


def all_pairs():
    seen = {}
    for r in csv.DictReader(open(RESULTS / "compack_reference.csv")):
        seen.setdefault((r["a_path"], r["b_path"]), r)
    return sorted(seen.items())


def _struct(rel):
    if rel not in _STRUCTS:
        _STRUCTS[rel] = load(str(REPO / rel))
    return _STRUCTS[rel]


_UNLOADABLE: set = set()


def _loadable(rel):
    """Whether a structure assembles at all, memoized. A coordination polymer / metal salt whose
    covalent-radius bonding percolates has no finite whole molecule, so RMSD_n is undefined for it."""
    if rel in _UNLOADABLE:
        return False
    try:
        _struct(rel)
        return True
    except Exception:
        _UNLOADABLE.add(rel)
        return False


def run_one(a, b, size):
    A, B = _struct(a), _struct(b)
    t0 = time.perf_counter()
    r = compare_structures(A, B, n=size)
    return {"a": a, "b": b, "shell": size, "t": time.perf_counter() - t0, "nmatched": r.n_matched,
            "rmsd": r.rmsd_n, "status": r.status, "inverted": r.inverted}


def _task(args):
    return run_one(*args)


def _init(structs, cores=None):
    _STRUCTS.update(structs)
    if cores:                                   # pin this worker to one physical core (see --affinity)
        import multiprocessing
        me = multiprocessing.current_process()._identity[0] - 1
        os.sched_setaffinity(0, {cores[me % len(cores)]})
    _warm_up()


def _warm_up():
    """One throw-away comparison so lazy scipy/numpy initialisation is not billed to the first task."""
    if _STRUCTS:
        first = next(iter(_STRUCTS.values()))
        compare_structures(first, first, n=5)


def _write(rows, path):
    RESULTS.mkdir(exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def _physical_cores():
    """One logical CPU per physical core (first sibling of each core), from /sys."""
    seen, cores = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        sib = open(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id").read().strip()
        pkg = open(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id").read().strip()
        if (pkg, sib) not in seen:
            seen.add((pkg, sib))
            cores.append(cpu)
    return cores


def _cost_proxy(a, b):
    """Rough per-pair cost estimate from structure size: (molecules) x (atoms), both sides."""
    A, B = _struct(a), _struct(b)
    return (len(A.molecules) + len(B.molecules)) * (sum(m.n_atoms for m in A.molecules) + sum(m.n_atoms for m in B.molecules))


def timing(tasks, workers, affinity=False):
    """Same dispatch as oxtalign.batch.distance_matrix: largest-expected pairs first, chunks of len/(16 W)."""
    keep = [t for t in tasks if _loadable(t[0]) and _loadable(t[1])]
    if len(keep) < len(tasks):
        bad = {p for t in tasks for p in t[:2] if not _loadable(p)}
        print(f"  skipping {len(tasks) - len(keep)} pair(s) over {len(bad)} structures that do not "
              f"assemble into whole molecules", flush=True)
    tasks = keep
    cores = _physical_cores() if affinity else None
    tasks = sorted(tasks, key=lambda t: -_cost_proxy(t[0], t[1]))
    chunk = max(1, len(tasks) // (workers * 16))
    _warm_up()
    t0 = time.perf_counter()
    if workers == 1:
        rows = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(_STRUCTS, cores)) as ex:
            rows = list(ex.map(_task, tasks, chunksize=chunk))
    return rows, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["timing", "scaling", "sweep", "select-sweep-pairs"])
    ap.add_argument("--shell", type=int, default=15)
    ap.add_argument("--shells", default="1,2,3,5,8,10,15,20,25,30,40,50,75,100,150")
    ap.add_argument("--workers", default="1")
    ap.add_argument("--pairs", default=str(RESULTS / "sweep_pairs.csv"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--count", type=int, default=48, help="select-sweep-pairs: pairs to draw (half matched)")
    ap.add_argument("--affinity", action="store_true", help="pin each worker to its own physical core")
    args = ap.parse_args()

    if args.mode == "select-sweep-pairs":
        # half the pairs COMPACK matched fully at n=15, half it clearly did not (no COMPACK error).
        # A small subsample is noisy at large n, where a few dense cells dominate the median.
        rng = random.Random(7)
        rows = [r for _, r in all_pairs() if not r["error"]]
        same = [r for r in rows if int(r["ccdc_nmatched_15"]) == 15]
        diff = [r for r in rows if int(r["ccdc_nmatched_15"]) < 8]
        half = args.count // 2
        pick = rng.sample(same, min(half, len(same))) + rng.sample(diff, min(half, len(diff)))
        _write([{"family": r["family"], "a_path": r["a_path"], "b_path": r["b_path"],
                 "ccdc_nmatched_15": r["ccdc_nmatched_15"], "ccdc_rmsd_15": r["ccdc_rmsd_15"]} for r in pick],
               Path(args.pairs))
        return
    if args.mode == "timing":
        tasks = [(a, b, args.shell) for (a, b), _ in all_pairs()] * args.repeat
        rows, wall = timing(tasks, int(args.workers))
        ts = sorted(r["t"] for r in rows)
        print(f"{len(rows)} comparisons, wall {wall:.1f}s, median {1000*ts[len(ts)//2]:.1f} ms, mean {1000*sum(ts)/len(ts):.1f} ms")
        _write(rows, RESULTS / f"oxtalign_timing_n{args.shell}.csv")
        return
    if args.mode == "scaling":
        tasks = [(a, b, args.shell) for (a, b), _ in all_pairs()] * args.repeat
        out = []
        for w in [int(x) for x in args.workers.split(",")]:
            rows, wall = timing(tasks, w, args.affinity)
            cpu = sum(r["t"] for r in rows)
            out.append({"engine": "oxtalign", "shell": args.shell, "workers": w, "comparisons": len(rows),
                        "wall_s": wall, "pairs_per_s": len(rows) / wall, "cpu_s": cpu})
            print(f"workers={w:3d}: {len(rows)} comparisons in {wall:.2f}s = {len(rows)/wall:.1f} pairs/s"
                  f"  (in-worker compute {cpu:.1f}s -> {100*cpu/(w*wall):.0f}% of {w} cores busy; "
                  f"per-task mean {1000*cpu/len(rows):.1f} ms)", flush=True)
        _write(out, RESULTS / f"oxtalign_scaling_n{args.shell}.csv")
        return
    pairs = [(r["a_path"], r["b_path"]) for r in csv.DictReader(open(args.pairs))]
    shells = [int(x) for x in args.shells.split(",")]
    tasks = [(a, b, s) for s in shells for a, b in pairs]
    rows, _ = timing(tasks, int(args.workers))
    for r in rows:
        if isinstance(r["rmsd"], float) and math.isnan(r["rmsd"]):
            r["rmsd"] = float("nan")
    _write(rows, RESULTS / "oxtalign_shell_sweep.csv")
    for s in shells:
        ts = sorted(r["t"] for r in rows if r["shell"] == s)
        print(f"shell {s:3d}: median {1000*ts[len(ts)//2]:.1f} ms  max {1000*ts[-1]:.0f} ms")


if __name__ == "__main__":
    main()
