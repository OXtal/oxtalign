"""Bond perception by covalent-radius cutoff.

Two atoms are bonded iff their distance <= r_cov(i) + r_cov(j) + tol. This is the *only* tolerance
in the comparison pipeline, and it is chemistry (element radii), not packing geometry — which is
why the resulting RMSD_n does not suffer COMPACK's distance/angle-tolerance fragility.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .elements import covalent_radius

BOND_TOL = 0.40


def perceive_bonds(coords: np.ndarray, elements: list[str], tol: float = BOND_TOL, *,
                   fast: bool = False, radii: np.ndarray | None = None) -> np.ndarray:
    """Return bonded index pairs as an (K, 2) int array (i < j) among the given atoms.

    ``fast=True`` builds a cheaper (unbalanced) KD-tree for large clouds: ``query_pairs`` is an exact
    predicate, so the pair *set* is unchanged and only its order can differ -- callers whose result
    depends on bond order must keep the default tree. ``radii`` passes in per-atom covalent radii a
    caller already holds, in which case ``elements`` may be None.
    """
    n = len(coords)
    if n < 2:
        return np.empty((0, 2), dtype=int)
    if radii is None:
        radius_of = {e: covalent_radius(e) for e in set(elements)}  # one lookup per unique element
        radii = np.fromiter(map(radius_of.__getitem__, elements), dtype=float, count=n)
    tree = (cKDTree(coords, balanced_tree=False, compact_nodes=False) if fast
            else cKDTree(coords))
    pairs = tree.query_pairs(2.0 * radii.max() + tol, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=int)
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(coords[i] - coords[j], axis=1)
    return pairs[d <= radii[i] + radii[j] + tol]
