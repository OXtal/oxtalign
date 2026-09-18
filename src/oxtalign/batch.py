"""Many-vs-many helpers: distance matrices, polymorph-set scoring, and a legacy-CSV shim.

These cover the two production use cases: comparing polymorphs, and screening many predicted
structures against all polymorphs of a ground truth (a drop-in for the CCDC-based pipeline, but
reading truth from local CIFs and without a CCDC licence).
"""
from __future__ import annotations

import math
import multiprocessing
import os
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .compare import compare, load
from .elements import vdw_radius
from .matching import compare_structures
from .model import ComparisonResult, Structure

# Pairwise comparisons are independent and CPU-bound (numpy/scipy), so the work parallelizes
# cleanly across processes. Workers hold the loaded structures in a module global (set once per
# worker via the pool initializer) so only indices cross the process boundary. Both paths call the
# SAME `_compare_one` on a process main thread (the SIGALRM guard is therefore active in both), so
# results are identical regardless of n_jobs -- except a pair that actually hits the wall-clock
# timeout, whose outcome depends on machine load rather than n_jobs.
_WORKER_STRUCTS: list[Structure] = []
_WORKER_TRUTHS: dict[str, Structure | None] = {}


class _Timeout(Exception):
    pass


def _compare_one(A: Structure, B: Structure, n: int, kw: dict, timeout: float | None
                 ) -> ComparisonResult:
    """compare_structures with an optional per-pair wall-clock guard (defensive; pairs are bounded).

    The SIGALRM timer only works on the main thread; off the main thread it is silently skipped.
    """
    use_timer = timeout and threading.current_thread() is threading.main_thread()
    if use_timer:
        def _raise(signum, frame):
            raise _Timeout()
        old = signal.signal(signal.SIGALRM, _raise)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return compare_structures(A, B, n=n, **kw)
    except _Timeout:
        return ComparisonResult(float("nan"), n, 0, "failed", message=f"timeout>{timeout}s")
    finally:
        if use_timer:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def _load_job(job) -> Structure | None:
    path, load_kw = job
    return _try_load(path, **load_kw)


def _try_load(path: str, **load_kw) -> Structure | None:
    """load(), or None when the file cannot be assembled into finite molecules (e.g. a garbage prediction
    whose molecules overlap into an infinite bonded network). Batch callers record such inputs as failed
    instead of aborting the whole run."""
    try:
        return load(path, **load_kw)
    except ValueError:
        return None


def physical_cores() -> list[int]:
    """One logical CPU per physical core among those this process may use ([] where unknown).

    With SMT/hyper-threading the scheduler happily places two busy workers on sibling threads of one
    core, and each then runs at ~60% speed; pinning one worker per physical core avoids that. Linux only.
    """
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return []
    seen, cores = set(), []
    for cpu in cpus:
        try:
            base = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
            key = (open(base + "physical_package_id").read().strip(), open(base + "core_id").read().strip())
        except OSError:
            return []
        if key not in seen:
            seen.add(key)
            cores.append(cpu)
    return cores


def _init_worker(structs, truths=None, cores=None, counter=None):
    global _WORKER_STRUCTS, _WORKER_TRUTHS
    _WORKER_STRUCTS = structs
    _WORKER_TRUTHS = truths or {}
    if cores and counter is not None:
        with counter.get_lock():
            index, counter.value = counter.value, counter.value + 1
        try:
            os.sched_setaffinity(0, {cores[index % len(cores)]})
        except (AttributeError, OSError):
            pass


def _pool(n_jobs: int, structs, truths=None, pin_cores: bool = False) -> ProcessPoolExecutor:
    cores = physical_cores() if pin_cores else []
    counter = multiprocessing.Value("i", 0) if cores else None
    return ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                               initargs=(structs, truths, cores, counter))


def _size(struct: Structure | None) -> tuple[int, int]:
    return (0, 0) if struct is None else (len(struct.molecules), sum(m.n_atoms for m in struct.molecules))


def _pair_job(arg):
    i, j, n, kw, timeout = arg
    return i, j, _compare_one(_WORKER_STRUCTS[i], _WORKER_STRUCTS[j], n, kw, timeout)


def intermolecular_clash(struct: Structure, tolerance: float = 0.0) -> float:
    """Largest van-der-Waals overlap (Å) between atoms of different molecules; <=0 means no clash.

    overlap = vdw_i + vdw_j - distance. Useful for flagging unphysical predicted geometries.
    Scoped to the molecules as given -- i.e. a predicted cluster's whole neighbourhood. For a
    periodic cell this sees only the single-cell molecule list, not contacts across the lattice
    boundary, so call it on cluster inputs (which is how score_predictions uses it).
    """
    coords, mol_id, radii = [], [], []
    for mi, m in enumerate(struct.molecules):
        coords.append(m.coords)
        mol_id.extend([mi] * m.n_atoms)
        radii.extend(vdw_radius(e) for e in m.elements)
    coords = np.vstack(coords)
    mol_id = np.array(mol_id)
    radii = np.array(radii)
    if len(coords) < 2:
        return float("-inf")
    tree = cKDTree(coords)
    pairs = tree.query_pairs(2.0 * radii.max(), output_type="ndarray")
    if len(pairs) == 0:
        return float("-inf")
    i, j = pairs[:, 0], pairs[:, 1]
    inter = mol_id[i] != mol_id[j]
    if not inter.any():
        return float("-inf")
    d = np.linalg.norm(coords[i[inter]] - coords[j[inter]], axis=1)
    overlap = radii[i[inter]] + radii[j[inter]] - d
    return float(overlap.max())


def distance_matrix(paths: list[str], *, n: int = 15, n_jobs: int = 1,
                    timeout: float | None = 120.0, pin_cores: bool = False, **kw):
    """Symmetric RMSD_n matrix over a set of structures (failed/non-matching pairs -> inf).

    n_jobs > 1 fans the pairwise comparisons out across processes; results are identical to the
    serial path (same `_compare_one`). Pairs are dispatched largest-first in small chunks so a few
    slow pairs cannot leave the other workers idle at the end. `pin_cores=True` additionally binds
    one worker to each physical core (Linux; useful on a dedicated many-core node, counterproductive
    when other jobs share the machine). `timeout` is a defensive per-pair wall-clock cap (s).
    Returns (matrix (N,N), results dict[(i,j)] -> ComparisonResult, loaded list[Structure | None]).
    A structure that cannot be assembled (infinite bonded network) is None; its pairs are inf/failed.
    """
    # Split load-time options out of kw; whatever remains is forwarded to compare_structures.
    load_kw = {k: kw.pop(k) for k in ("include_hydrogens", "drop_minor_occupancy") if k in kw}
    if n_jobs == 1 or len(paths) < 8:            # a pool costs ~0.1 s to start; a CIF ~17 ms to load
        structs = [_try_load(p, **load_kw) for p in paths]
    else:                                        # parsing + assembly is per-file work too: do it in the pool
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(paths))) as ex:
            jobs = [(p, load_kw) for p in paths]
            structs = list(ex.map(_load_job, jobs, chunksize=max(1, len(jobs) // (n_jobs * 4))))
    N = len(structs)
    M = np.zeros((N, N))
    results = {}
    unloadable = ComparisonResult(float("nan"), n, 0, "failed", message="structure could not be assembled")
    for i in range(N):
        for j in range(i + 1, N):
            if structs[i] is None or structs[j] is None:
                results[(i, j)] = unloadable
                M[i, j] = M[j, i] = float("inf")
    pairs = [(i, j, n, kw, timeout) for i in range(N) for j in range(i + 1, N) if (i, j) not in results]
    # Longest-expected-first: the cost of a pair grows with molecules x atoms on both sides, and
    # handing the big ones out first (in small chunks) keeps the tail of the run balanced.
    sizes = [_size(s) for s in structs]
    pairs.sort(key=lambda p: -((sizes[p[0]][0] + sizes[p[1]][0]) * (sizes[p[0]][1] + sizes[p[1]][1])))

    def _record(i, j, r):
        results[(i, j)] = r
        val = r.rmsd_n if (r.status != "failed" and not math.isnan(r.rmsd_n)) else float("inf")
        M[i, j] = M[j, i] = val

    if n_jobs == 1:
        for i, j, _, _, _ in pairs:
            _record(i, j, _compare_one(structs[i], structs[j], n, kw, timeout))
    else:
        with _pool(n_jobs, structs, pin_cores=pin_cores) as ex:
            for i, j, r in ex.map(_pair_job, pairs, chunksize=max(1, len(pairs) // (n_jobs * 16))):
                _record(i, j, r)
    return M, results, structs


def compare_to_polymorph_set(query: str | Structure, truth_cifs: list[str], *, n: int = 15,
                             **kw) -> tuple[ComparisonResult, str]:
    """Compare a query to every truth polymorph; return (best result, best truth path).

    "Best" = passing match with lowest RMSD_n; falls back to the most molecules matched.
    """
    q = query if isinstance(query, Structure) else load(query)
    best, best_path = None, None
    for tpath in truth_cifs:
        r = compare(q, tpath, n=n, **kw)
        if _rank(r) > _rank(best):
            best, best_path = r, tpath
    return best, best_path


def _rank(r: ComparisonResult | None):
    if r is None:
        return (-1, -1, 0.0)
    full = 1 if r.status in ("full", "partial") else 0
    rms = -r.rmsd_n if (r.rmsd_n == r.rmsd_n) else -1e9   # NaN-safe; lower rmsd ranks higher
    return (full, r.n_matched, rms)


def refcode_from_filename(name: str) -> str:
    """CSD refcode = filename prefix before the first underscore (e.g. CAPRYL_seed0... -> CAPRYL)."""
    return Path(name).stem.split("_")[0]


def score_predictions(pred_dir: str, truth_map: dict[str, list[str]], *, n: int = 15,
                      mol_rmsd_tol: float = 1.0, match_fraction: float = 0.5,
                      clash_tol: float = 0.4, lattice_tol: float = 0.5, n_jobs: int = 1,
                      pin_cores: bool = False) -> list[dict]:
    """Score every predicted CIF against its ground-truth polymorph set (legacy CSV schema).

    Each row: file, csd_refcode, best_true_refcode, passed, clash, nmatched_1/rmsd_1,
    nmatched_{n}/rmsd_{n}, status, inverted. Truth CIFs are loaded once; ``n_jobs`` > 1 scores the
    predictions in parallel processes (rows are identical to the serial run and keep file order).
    """
    need = math.ceil(n * match_fraction)
    files = sorted(Path(pred_dir).glob("*.cif"))
    truths = {t: load(t) for t in sorted({t for ts in truth_map.values() for t in ts})}
    jobs = [(str(f), refcode_from_filename(f.name), tuple(truth_map.get(refcode_from_filename(f.name), [])),
             n, need, mol_rmsd_tol, match_fraction, clash_tol, lattice_tol) for f in files]
    if n_jobs == 1:
        _init_worker([], truths)
        return [_score_job(job) for job in jobs]
    with _pool(n_jobs, [], truths, pin_cores=pin_cores) as ex:
        return list(ex.map(_score_job, jobs, chunksize=max(1, len(jobs) // (n_jobs * 16))))


def _score_job(job) -> dict:
    fpath, ref, truth_paths, n, need, mol_rmsd_tol, match_fraction, clash_tol, lattice_tol = job
    fname = Path(fpath).name
    pred = _try_load(fpath)
    if pred is None:
        row = _row(fname, ref, None, None, n, need, False, mol_rmsd_tol, lattice_tol)
        return {**row, "status": "load_error"}
    clash = intermolecular_clash(pred) > clash_tol
    best, best_path = None, None
    for tpath in truth_paths:
        r = compare_structures(pred, _WORKER_TRUTHS[tpath], n=n, mol_rmsd_tol=mol_rmsd_tol,
                               match_fraction=match_fraction)
        if _rank(r) > _rank(best):
            best, best_path = r, tpath
    return _row(fname, ref, best, best_path, n, need, clash, mol_rmsd_tol, lattice_tol)


def _row(fname, ref, r, best_path, n, need, clash, mol_rmsd_tol, lattice_tol) -> dict:
    if r is None:
        return {"file": fname, "csd_refcode": ref, "best_true_refcode": "", "passed": False,
                "clash": clash, "nmatched_1": 0, "rmsd_1": float("nan"),
                f"nmatched_{n}": 0, f"rmsd_{n}": float("nan"), "status": "no_truth", "inverted": False}
    r1 = r.rmsd_1 if r.rmsd_1 is not None else float("nan")
    nm1 = int(r1 == r1 and r1 < lattice_tol)
    rmsd_n = r.rmsd_n if r.n_matched > 0 else float("nan")
    return {
        "file": fname, "csd_refcode": ref,
        "best_true_refcode": refcode_from_filename(Path(best_path).name) if best_path else "",
        "passed": bool(r.n_matched >= need and not clash),
        "clash": clash, "nmatched_1": nm1, "rmsd_1": r1,
        f"nmatched_{n}": r.n_matched, f"rmsd_{n}": rmsd_n,
        "status": r.status, "inverted": r.inverted,
    }


def write_csv(rows: list[dict], path: str) -> None:
    import csv
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
