"""Command-line interface.

    oxtalign compare A.cif B.cif [--n 15] [--profile] [--json]   pairwise RMSD_n (or: oxtalign A.cif B.cif)
    oxtalign matrix *.cif [--jobs 8] [--csv matrix.csv]           all-pairs RMSD_n
    oxtalign dedup *.cif [--threshold 0.25] [--jobs 8]            group a landscape into distinct packings
    oxtalign score preds/ --truth truths/ [--csv scores.csv]      score predictions against experimental forms
    oxtalign overlay A.cif B.cif [-o match.html] [--pdb m.pdb]    interactive 3D overlay / PDB for viewers
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .batch import distance_matrix, refcode_from_filename, score_predictions
from .compare import compare, match_profile
from .overlay import build_overlay, pdb_text, render_html

VERBS = ("compare", "matrix", "dedup", "score", "overlay")


def _match_options(sp, *, inversion=True):
    sp.add_argument("--n", type=int, default=15, help="coordination-shell size (default 15)")
    sp.add_argument("--tol", type=float, default=1.0,
                    help="per-molecule RMSD inlier tolerance in Å (default 1.0)")
    sp.add_argument("--include-hydrogens", action="store_true",
                    help="keep H atoms (default: heavy atoms only)")
    sp.add_argument("--allow-partial", action="store_true",
                    help="tolerate small atom-count / counter-salt differences (partial overlay)")
    if inversion:
        sp.add_argument("--no-inversion", action="store_true", help="disallow the mirror-image overlay")


def _match_kwargs(args) -> dict:
    return dict(n=args.n, allow_inversion=not args.no_inversion, mol_rmsd_tol=args.tol,
                include_hydrogens=args.include_hydrogens, allow_partial=args.allow_partial)


def _nan_safe(o):
    return None if isinstance(o, float) and np.isnan(o) else o


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oxtalign", description="OXtalign: fast, open RMSD_n packing similarity for molecular crystals.")
    p.add_argument("--version", action="version", version=f"oxtalign {__version__}")
    sub = p.add_subparsers(dest="verb", required=True, metavar="{" + ",".join(VERBS) + "}")

    sp = sub.add_parser("compare", help="pairwise RMSD_n between two structures")
    sp.add_argument("a")
    sp.add_argument("b")
    _match_options(sp)
    sp.add_argument("--profile", action="store_true",
                    help="coverage curve: n_matched + RMSD_n at tol = 1, 2, 3 Å")
    sp.add_argument("--json", action="store_true", help="emit JSON")

    sp = sub.add_parser("matrix", help="all-pairs RMSD_n distance matrix")
    sp.add_argument("cifs", nargs="+")
    _match_options(sp)
    sp.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    sp.add_argument("--pin-cores", action="store_true", help="one worker per physical core (dedicated node)")
    sp.add_argument("--csv", help="write the matrix to this CSV file")
    sp.add_argument("--json", action="store_true", help="emit JSON")

    sp = sub.add_parser("dedup", help="group structures whose packings match (RMSD_n below a threshold)")
    sp.add_argument("cifs", nargs="+")
    _match_options(sp)
    sp.add_argument("--threshold", type=float, default=0.25,
                    help="same packing = all n molecules matched with RMSD_n below this many Å")
    sp.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    sp.add_argument("--pin-cores", action="store_true", help="one worker per physical core (dedicated node)")
    sp.add_argument("--csv", help="write file,cluster,representative rows to this CSV file")

    sp = sub.add_parser("score", help="score predicted structures against experimental truths")
    sp.add_argument("pred_dir", help="directory of predicted CIFs named <REFCODE>_*.cif")
    sp.add_argument("--truth", nargs="+", required=True,
                    help="truth CIFs or directories, matched to predictions by refcode "
                         "(filename prefix before the first underscore; trailing polymorph digits ignored)")
    sp.add_argument("--n", type=int, default=15)
    sp.add_argument("--tol", type=float, default=1.0)
    sp.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    sp.add_argument("--pin-cores", action="store_true", help="one worker per physical core (dedicated node)")
    sp.add_argument("--csv", help="write the per-prediction rows to this CSV file")

    sp = sub.add_parser("overlay", help="overlay B onto A's coordination shell and write a viewer file")
    sp.add_argument("a")
    sp.add_argument("b")
    _match_options(sp)
    sp.add_argument("-o", "--html", help="self-contained interactive HTML viewer (default overlay.html)")
    sp.add_argument("--pdb", help="two-chain PDB: A = reference shell, B = overlaid, B-factor = RMSD")
    sp.add_argument("--match-tol", type=float, default=None,
                    help="per-molecule RMSD below which a molecule is drawn as matched (default: --tol)")
    return p


def _legacy(argv: list[str]) -> list[str]:
    """`oxtalign A.cif B.cif ...` means compare; the old `--matrix` flag means matrix."""
    if not argv or argv[0] in VERBS or argv[0] in ("-h", "--help", "--version"):
        return argv
    if "--matrix" in argv:
        return ["matrix", *[a for a in argv if a != "--matrix"]]
    return ["compare", *argv]


def _compare(args) -> int:
    if args.profile:
        rows = match_profile(args.a, args.b, n=args.n, allow_inversion=not args.no_inversion,
                             allow_partial=args.allow_partial, include_hydrogens=args.include_hydrogens)
        if args.json:
            print(json.dumps({"a": args.a, "b": args.b, "profile": rows}, indent=2, default=_nan_safe))
        else:
            print(f"coverage profile  {args.a}  vs  {args.b}")
            for row in rows:
                rms = "  –  " if row["rmsd_n"] != row["rmsd_n"] else f"{row['rmsd_n']:.4f}"
                print(f"  tol={row['tol']:.1f} Å:  matched {row['n_matched']:2d}/{args.n}"
                      f"   RMSD_{args.n}={rms}   {row['status']}")
        return 0
    r = compare(args.a, args.b, **_match_kwargs(args))
    if args.json:
        out = {"a": args.a, "b": args.b, "n": r.n, "rmsd_n": r.rmsd_n, "n_matched": r.n_matched,
               "status": r.status, "rmsd_1": r.rmsd_1, "inverted": r.inverted,
               "radius_of_gyration": list(r.radius_of_gyration), "message": r.message}
        print(json.dumps(out, indent=2, default=_nan_safe))
    else:
        print(f"RMSD_{r.n} = {r.rmsd_n:.4f} Å   matched {r.n_matched}/{r.n}   status: {r.status}")
        if r.rmsd_1 is not None:
            print(f"RMSD_1 (central conformer) = {r.rmsd_1:.4f} Å"
                  + ("   [enantiomer / mirror overlay]" if r.inverted else ""))
        if r.message:
            print(r.message)
    return 0


def _matrix(args) -> int:
    M, _, _ = distance_matrix(args.cifs, n_jobs=args.jobs, pin_cores=args.pin_cores, **_match_kwargs(args))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", *args.cifs])
            for f, row in zip(args.cifs, M, strict=True):
                w.writerow([f, *("inf" if np.isinf(v) else f"{v:.6f}" for v in row)])
        print(f"wrote {args.csv}")
    if args.json:
        print(json.dumps({"files": args.cifs, "rmsd_matrix": M.tolist()}, indent=2))
    elif not args.csv:
        print("       " + "".join(f"{i:>8d}" for i in range(len(args.cifs))))
        for i, row in enumerate(M):
            print(f"{i:>5d}  " + "".join(f"{v:8.3f}" for v in row))
        for i, f in enumerate(args.cifs):
            print(f"  [{i}] {f}")
    return 0


def _dedup(args) -> int:
    M, results, structs = distance_matrix(args.cifs, n_jobs=args.jobs, pin_cores=args.pin_cores,
                                          **_match_kwargs(args))
    n = len(args.cifs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, j), r in results.items():
        if r.status == "full" and r.rmsd_n < args.threshold:
            parent[max(find(i), find(j))] = min(find(i), find(j))
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    ordered = sorted(clusters.values(), key=lambda m: (-len(m), m[0]))
    unloadable = [args.cifs[i] for i, s in enumerate(structs) if s is None]
    print(f"{len(ordered)} distinct packings among {n} structures "
          f"(same packing = all {args.n} molecules matched with RMSD_{args.n} < {args.threshold} Å)")
    for k, members in enumerate(ordered, 1):
        rep = args.cifs[members[0]]
        others = ", ".join(args.cifs[i] for i in members[1:])
        print(f"  cluster {k} ({len(members)}): {rep}" + (f"  = {others}" if others else ""))
    if unloadable:
        print(f"  could not be assembled (left as singletons): {', '.join(unloadable)}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "cluster", "representative", "n_members"])
            w.writeheader()
            for k, members in enumerate(ordered, 1):
                for i in members:
                    w.writerow({"file": args.cifs[i], "cluster": k, "representative": args.cifs[members[0]],
                                "n_members": len(members)})
        print(f"wrote {args.csv}")
    return 0


def _score(args) -> int:
    truth_map: dict[str, list[str]] = {}
    for t in args.truth:
        paths = sorted(Path(t).glob("*.cif")) if Path(t).is_dir() else [Path(t)]
        for pth in paths:
            code = refcode_from_filename(pth.name)
            for key in {code, code.rstrip("0123456789")}:        # QAXMEH24.cif serves QAXMEH_* predictions
                truth_map.setdefault(key, []).append(str(pth))
    rows = score_predictions(args.pred_dir, truth_map, n=args.n, mol_rmsd_tol=args.tol, n_jobs=args.jobs,
                             pin_cores=args.pin_cores)
    passed = sum(bool(r["passed"]) for r in rows)
    need = int(np.ceil(args.n / 2))
    print(f"{passed}/{len(rows)} predictions match a truth polymorph "
          f"(>= {need} of {args.n} molecules, no clash)")
    for r in rows:
        rms = r[f"rmsd_{args.n}"]
        verdict = "PASS" if r["passed"] else "fail"
        print(f"  {r['file']:40s} {verdict:4s}  matched {r[f'nmatched_{args.n}']:2d}/{args.n}"
              f"  RMSD={rms if rms != rms else round(rms, 4)}  truth={r['best_true_refcode'] or '-'}"
              f"  {r['status']}")
    if args.csv:
        from .batch import write_csv
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    return 0


def _overlay(args) -> int:
    match_tol = args.tol if args.match_tol is None else args.match_tol
    payload = build_overlay(args.a, args.b, n=args.n, match_tol=match_tol,
                            allow_inversion=not args.no_inversion, include_hydrogens=args.include_hydrogens,
                            allow_partial=args.allow_partial)
    html = args.html or (None if args.pdb else "overlay.html")
    if html:
        Path(html).write_text(render_html(payload))
    if args.pdb:
        Path(args.pdb).write_text(pdb_text(payload))
    rmsd = payload["rmsd_n"]
    print(f"RMSD_{payload['n']} = {rmsd if rmsd is not None else float('nan'):.4f} Å   "
          f"matched {payload['n_matched']}/{payload['n']}   status: {payload['status']}"
          + ("   [mirror overlay]" if payload["inverted"] else ""))
    for path in (html, args.pdb):
        if path:
            print(f"wrote {Path(path).resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = _legacy(list(sys.argv[1:] if argv is None else argv))
    args = _parser().parse_args(argv)
    verbs = {"compare": _compare, "matrix": _matrix, "dedup": _dedup, "score": _score, "overlay": _overlay}
    return verbs[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())
