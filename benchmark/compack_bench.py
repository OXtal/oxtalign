"""COMPACK timing / parallel-scaling / shell-size runs (CCDC env). Writes CSVs under benchmark/results/.

    micromamba run -n ccdc python benchmark/compack_bench.py timing  --shell 15 --workers 1
    micromamba run -n ccdc python benchmark/compack_bench.py scaling --shell 15 --workers 1,2,4,8,16,32
    micromamba run -n ccdc python benchmark/compack_bench.py sweep   --shells 1,2,3,5,8,10,15,20,25,30 --pairs benchmark/results/sweep_pairs.csv

Settings mirror the reference CSVs (0.5 A / 75 deg tolerances, hydrogens + bond counts/types ignored,
artificial inversion allowed). Per-pair times cover ``PackingSimilarity.compare`` only; crystals are
read once per worker process. ``sweep`` runs every (pair, shell) task in its own subprocess with a hard
wall-clock kill, because COMPACK's internal timeout is not reliably honoured for large shells.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmark" / "results"
_CRYSTALS: dict = {}


def all_pairs():
    seen = {}
    for r in csv.DictReader(open(RESULTS / "compack_reference.csv")):
        seen.setdefault((r["a_path"], r["b_path"]), r["family"])
    return [(a, b, fam) for (a, b), fam in sorted(seen.items())]


def _configure(engine, size, timeout_ms):
    s = engine.settings
    s.packing_shell_size = size
    s.distance_tolerance, s.angle_tolerance, s.match_entire_packing_shell = 0.5, 75, False
    s.ignore_hydrogen_counts = s.ignore_hydrogen_positions = True
    s.ignore_bond_counts = s.ignore_bond_types = True
    s.allow_artificial_inversion = True
    s.timeout_ms = timeout_ms


def _crystal(rel):
    if rel not in _CRYSTALS:
        from ccdc.io import CrystalReader
        with CrystalReader(str(REPO / rel)) as reader:
            _CRYSTALS[rel] = reader[0]
    return _CRYSTALS[rel]


def run_one(a, b, size, timeout_ms=20000):
    from ccdc.crystal import PackingSimilarity
    ca, cb = _crystal(a), _crystal(b)
    engine = PackingSimilarity()
    _configure(engine, size, timeout_ms)
    t0 = time.perf_counter()
    try:
        c = engine.compare(ca, cb)
        n, rmsd, err = (int(c.nmatched_molecules), float(c.rmsd), "") if c is not None else (0, float("nan"), "")
    except Exception as e:  # noqa: BLE001
        n, rmsd, err = 0, float("nan"), f"{type(e).__name__}:{e}"
    return {"a": a, "b": b, "shell": size, "t": time.perf_counter() - t0, "nmatched": n, "rmsd": rmsd, "error": err}


def _task(args):
    return run_one(*args)


def _write(rows, path):
    RESULTS.mkdir(exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def timing(shell, workers, repeat=1):
    pairs = all_pairs() * repeat
    tasks = [(a, b, shell) for a, b, _ in pairs]
    t0 = time.perf_counter()
    if workers == 1:
        rows = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_task, tasks, chunksize=1))
    wall = time.perf_counter() - t0
    return rows, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["timing", "scaling", "sweep", "one"])
    ap.add_argument("--shell", type=int, default=15)
    ap.add_argument("--shells", default="1,2,3,5,8,10,15,20,25,30")
    ap.add_argument("--workers", default="1")
    ap.add_argument("--pairs", default=str(RESULTS / "sweep_pairs.csv"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="", help="sweep: write here instead of results/compack_shell_sweep.csv")
    ap.add_argument("--kill-after", type=float, default=600.0, help="sweep: hard per-task wall clock (s)")
    ap.add_argument("--compack-timeout", type=float, default=120.0,
                    help="sweep: COMPACK's own timeout_ms (s); timing/scaling use the reference CSVs' 20 s")
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()

    if args.mode == "one":                       # subprocess entry used by `sweep`
        a, b, size = args.rest[0], args.rest[1], int(args.rest[2])
        print(json.dumps(run_one(a, b, size, timeout_ms=int(args.compack_timeout * 1000))))
        return
    if args.mode == "timing":
        rows, wall = timing(args.shell, int(args.workers), args.repeat)
        print(f"{len(rows)} comparisons, wall {wall:.1f}s, {len(rows)/wall:.2f} pairs/s")
        _write(rows, RESULTS / f"compack_timing_n{args.shell}.csv")
        return
    if args.mode == "scaling":
        out = []
        for w in [int(x) for x in args.workers.split(",")]:
            rows, wall = timing(args.shell, w, args.repeat)
            out.append({"engine": "COMPACK", "shell": args.shell, "workers": w, "comparisons": len(rows),
                        "wall_s": wall, "pairs_per_s": len(rows) / wall})
            print(f"workers={w:3d}: {len(rows)} comparisons in {wall:.1f}s = {len(rows)/wall:.2f} pairs/s", flush=True)
        _write(out, RESULTS / f"compack_scaling_n{args.shell}.csv")
        return
    # sweep: subprocess per (pair, shell) with a hard kill
    pairs = [(r["a_path"], r["b_path"]) for r in csv.DictReader(open(args.pairs))]
    shells = [int(x) for x in args.shells.split(",")]
    tasks = [(a, b, s) for s in shells for a, b in pairs]

    def run(task):
        a, b, s = task
        # option first: argparse only collects `rest` positionals that directly follow the mode
        cmd = [sys.executable, __file__, "--compack-timeout", str(args.compack_timeout), "one", a, b, str(s)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.kill_after, cwd=REPO)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            if line:
                return json.loads(line[-1])
            return {"a": a, "b": b, "shell": s, "t": float("nan"), "nmatched": 0, "rmsd": float("nan"),
                    "error": "crash:" + p.stderr.strip()[-120:]}
        except subprocess.TimeoutExpired:
            return {"a": a, "b": b, "shell": s, "t": args.kill_after, "nmatched": 0, "rmsd": float("nan"),
                    "error": f"killed>{args.kill_after:.0f}s"}

    with ThreadPoolExecutor(int(args.workers)) as ex:
        rows = list(ex.map(run, tasks))
    _write(rows, Path(args.out) if args.out else RESULTS / "compack_shell_sweep.csv")
    for s in shells:
        rs = [r for r in rows if r["shell"] == s]
        ts = sorted(r["t"] for r in rs)
        print(f"shell {s:3d}: median {ts[len(ts)//2]:.3f}s  errors/timeouts {sum(bool(r['error']) for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
