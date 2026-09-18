"""Top-level pairwise comparison API."""
from __future__ import annotations

from .assemble import assemble
from .io_cif import read
from .matching import compare_structures
from .model import ComparisonResult, Structure


def load(path: str, *, include_hydrogens: bool = False,
         drop_minor_occupancy: bool = True) -> Structure:
    """Read a CIF and assemble it into whole molecules (one unit cell, or the cluster as-is)."""
    return assemble(read(path), include_hydrogens=include_hydrogens,
                    drop_minor_occupancy=drop_minor_occupancy)


def compare(a: str | Structure, b: str | Structure, *, n: int = 15, allow_inversion: bool = True,
            mol_rmsd_tol: float = 1.0, match_fraction: float = 0.5,
            include_hydrogens: bool = False, drop_minor_occupancy: bool = True,
            allow_partial: bool = False,
            refine_schedule: tuple[float, ...] = (1.5,)) -> ComparisonResult:
    """Compute RMSD_n between two structures (CIF paths or pre-loaded Structures).

    Returns a ComparisonResult with rmsd_n, n_matched, status, rmsd_1 (central conformer check),
    the optimal transform, radius of gyration, and the chirality relationship (``inverted``).

    ``allow_partial`` (default off) tolerates molecules whose atom counts differ slightly between
    determinations (disorder / borderline-bond drift) or that carry a different counter-salt: the
    species gate accepts a close element-composition match derived from the molecules themselves,
    and the per-molecule overlay uses a partial atom correspondence over the shared scaffold. Off,
    behaviour is unchanged.

    ``refine_schedule`` is the inlier tolerance the trimmed ICP uses in its first iterations before
    settling on ``mol_rmsd_tol``; it widens the search basin without changing what counts as a match.
    """
    def _prep(x):
        return x if isinstance(x, Structure) else load(
            x, include_hydrogens=include_hydrogens, drop_minor_occupancy=drop_minor_occupancy)

    return compare_structures(_prep(a), _prep(b), n=n, allow_inversion=allow_inversion,
                              mol_rmsd_tol=mol_rmsd_tol, match_fraction=match_fraction,
                              allow_partial=allow_partial, refine_schedule=refine_schedule)


def match_profile(a: str | Structure, b: str | Structure, *, n: int = 15,
                  tols: tuple[float, ...] = (1.0, 2.0, 3.0), allow_inversion: bool = True,
                  allow_partial: bool = False, include_hydrogens: bool = False,
                  drop_minor_occupancy: bool = True) -> list[dict]:
    """Packing-overlap coverage curve: (n_matched, RMSD_n) at increasing per-molecule tolerances.

    Each tolerance is fit independently (its alignment maximizes molecules within it), so the curve
    shows how much packing is shared as the criterion loosens: a tight same-form core at small tol,
    the partial motif at larger tol. Opt-in (re-runs the match per tolerance); structures load once.
    Returns a list of {tol, n_matched, rmsd_n, status}.
    """
    A = a if isinstance(a, Structure) else load(a, include_hydrogens=include_hydrogens,
                                                drop_minor_occupancy=drop_minor_occupancy)
    B = b if isinstance(b, Structure) else load(b, include_hydrogens=include_hydrogens,
                                                drop_minor_occupancy=drop_minor_occupancy)
    rows = []
    for tol in tols:
        r = compare_structures(A, B, n=n, allow_inversion=allow_inversion,
                               mol_rmsd_tol=float(tol), allow_partial=allow_partial)
        rows.append({"tol": float(tol), "n_matched": r.n_matched, "rmsd_n": r.rmsd_n,
                     "status": r.status})
    return rows
