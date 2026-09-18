"""Sample real polymorph families from the CSD, save CIFs, and score all within-family pairs with
COMPACK. Run in the ccdc env:  micromamba run -n ccdc python fetch_and_compack.py [--families N] [--jobs N]

A "family" = entries sharing a 6-letter refcode base (CSD's own grouping for the same compound /
its redeterminations & polymorphs). We sample random entries, expand each to its family, keep
organic-ish families with >=3 members of reasonable size, write CIFs, and run COMPACK pairwise.
Writes results/compack_reference.csv, which compare_to_compack.py and plot_figures.py read.
"""
import argparse
import csv
import itertools
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ccdc.crystal import PackingSimilarity
from ccdc.io import CrystalReader, EntryReader

REPO = Path(__file__).resolve().parents[1]
CIF_DIR = REPO / "tests" / "data" / "csd_families"
OUT = Path(__file__).resolve().parent / "results" / "compack_reference.csv"
CURATED = REPO / "tests" / "data" / "experimental"
TARGET_PAIRS = 2000
MAX_MEMBERS = 5
MAX_HEAVY = 70
TIMEOUT_MS = 15000
SCAN = 40000            # CSD entries sampled while looking for families; fixed so that raising
                        # --families extends the same sequence rather than reshuffling it


def usable(entry):
    try:
        if not entry.has_3d_structure or entry.is_polymeric:
            return False
        # entry.is_polymeric is False even for catena structures; the formula is the reliable
        # marker, CSD writes a polymeric component as "(...)n". RMSD_n needs whole molecules.
        if ")n" in (entry.formula or ""):
            return False
        mol = entry.molecule
        heavy = sum(1 for a in mol.atoms if a.atomic_symbol != "H")
        return 0 < heavy <= MAX_HEAVY
    except Exception:
        return False


def collect_families(reader, n_families):
    families, bases, n = {}, set(), len(reader)
    for idx in random.sample(range(n), min(n, SCAN)):
        if len(families) >= n_families:
            break
        base = reader[idx].identifier[:6].rstrip("0123456789")
        if base in bases or len(base) < 4:
            continue
        bases.add(base)
        members = []
        for suf in [""] + [f"{i:02d}" for i in range(1, 30)]:
            try:
                e = reader.entry(base + suf)
            except Exception:
                continue
            if usable(e):
                members.append((e.identifier, e.crystal))
        if len(members) >= 3:
            families[base] = members[:MAX_MEMBERS]
    return families


def curated_pairs():
    """Within-family pairs of the curated polymorph families shipped under tests/data/experimental."""
    pairs = []
    for d in sorted(CURATED.iterdir()):
        cifs = sorted(d.glob("*.cif")) if d.is_dir() else []
        if d.is_dir() and len(cifs) >= 2:
            pairs += [(d.name, a, b) for a, b in itertools.combinations(cifs, 2)]
    return pairs


def _assembles(paths):
    """Whether every member of a family resolves into whole molecules.

    A family is dropped whole rather than in part, so a polymeric framework, a structure with
    atoms overlapping at an impossible distance, or anything else outside the molecular-crystal
    domain contributes no pairs at all instead of a half-populated family.
    """
    from oxtalign.compare import load
    for p in paths:
        try:
            load(str(p))
        except Exception:
            return False
    return True


def write_cifs(families):
    pairs, dropped = [], []
    for base, members in families.items():
        d = CIF_DIR / base
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for refcode, crystal in members:
            p = d / f"{refcode}.cif"
            p.write_text(crystal.to_string("cif"))
            paths.append(p)
        if not _assembles(paths):
            dropped.append(base)
            continue
        pairs += [(base, a, b) for a, b in itertools.combinations(paths, 2)]
    if dropped:
        print(f"dropped {len(dropped)} families that do not assemble: {', '.join(dropped)}",
              flush=True)
    return pairs


def configure(engine, size):
    s = engine.settings
    s.packing_shell_size = size
    if size == 1:
        s.distance_tolerance, s.angle_tolerance, s.match_entire_packing_shell = 0.2, 20, True
    else:
        s.distance_tolerance, s.angle_tolerance, s.match_entire_packing_shell = 0.5, 75, False
    s.ignore_hydrogen_counts = s.ignore_hydrogen_positions = True
    s.ignore_bond_counts = s.ignore_bond_types = True
    s.allow_artificial_inversion = True
    s.timeout_ms = TIMEOUT_MS


def run_pair(a, b, size):
    engine = PackingSimilarity()
    configure(engine, size)
    with CrystalReader(str(a)) as ra, CrystalReader(str(b)) as rb:
        comp = engine.compare(ra[0], rb[0])
    return (0, float("nan")) if comp is None else (int(comp.nmatched_molecules), float(comp.rmsd))


def score(task):
    source, fam, a, b = task
    a, b = Path(a), Path(b)
    row = {"family": fam, "source": source, "a": a.name, "b": b.name,
           "a_path": str(a.relative_to(REPO)), "b_path": str(b.relative_to(REPO))}
    try:
        row["ccdc_nmatched_15"], row["ccdc_rmsd_15"] = run_pair(a, b, 15)
        row["ccdc_nmatched_1"], row["ccdc_rmsd_1"] = run_pair(a, b, 1)
        row["error"] = ""
    except Exception as e:
        row.update(ccdc_nmatched_15=0, ccdc_rmsd_15=float("nan"),
                   ccdc_nmatched_1=0, ccdc_rmsd_1=float("nan"), error=f"{type(e).__name__}:{e}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=1, help="processes for the COMPACK pass")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    random.seed(args.seed)

    curated = curated_pairs()
    reader = EntryReader("CSD")
    families = collect_families(reader, args.families)
    print(f"{len(families)} CSD families: " + ", ".join(f"{k}({len(v)})" for k, v in families.items()))
    sampled = write_cifs(families)
    # exactly TARGET_PAIRS rows: every curated pair, then random pairs in family order until full
    room = max(0, TARGET_PAIRS - len(curated))
    if len(sampled) < room:
        raise SystemExit(f"only {len(curated) + len(sampled)} pairs available; raise --families")
    sampled = sampled[:room]
    print(f"{len(curated)} curated + {len(sampled)} random = {len(curated) + len(sampled)} pairs",
          flush=True)
    tasks = ([("curated", fam, str(a), str(b)) for fam, a, b in curated]
             + [("random", fam, str(a), str(b)) for fam, a, b in sampled])
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(score, tasks, chunksize=1))
    else:
        rows = [score(t) for t in tasks]
    rows.sort(key=lambda r: (r["source"], r["family"], r["a"], r["b"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
