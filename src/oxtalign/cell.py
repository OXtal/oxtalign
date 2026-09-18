"""Unit-cell geometry: symmetry expansion (with position dedup), supercell replication.

The dedup after symmetry expansion is the key correctness guarantee: it makes the pipeline
behave identically whether a CIF stores only the asymmetric unit (symops generate new atoms) or
the already-expanded full cell (regenerated positions collapse onto existing ones). See HOLSUX.
"""
from __future__ import annotations

import gemmi
import numpy as np

from .model import UnitCell

_DEDUP_CART_TOL_A = 0.02
_PREEXPANDED_DEDUP_CART_TOL_A = 0.005


def symop_matrices(symops: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse xyz triplets into (rotation (3,3), translation (3,)) acting on fractional coords."""
    out = []
    for s in symops:
        seitz = np.array(gemmi.Op(s).float_seitz())
        out.append((seitz[:3, :3], seitz[:3, 3]))
    return out


def _wrap01(frac: np.ndarray) -> np.ndarray:
    f = np.mod(frac, 1.0)
    # guard against fp producing exactly 1.0 (cKDTree boxsize requires [0, 1))
    return np.clip(f, 0.0, np.nextafter(1.0, 0.0))


def _min_image_distances(delta_frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Cartesian lengths of fractional displacements taken to their nearest lattice image.

    Wrapping every fractional component into [-0.5, 0.5) *is* the minimum image whenever the
    displacement is tiny compared with the cell -- the coincident-site regime this dedup works in
    (tolerances of 0.02 A against cells of several A). For large displacements it is only an upper
    bound on the true minimum-image distance, which can never merge two sites that are far apart.
    """
    wrapped = delta_frac - np.round(delta_frac)
    return np.linalg.norm(wrapped @ lattice, axis=1)


def expand_unit_cell(elements: list[str], labels: list[str], frac: np.ndarray, cell: UnitCell
                     ) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Apply all symmetry operators to the input sites and deduplicate coincident positions.

    Returns (elements, labels, fractional (M,3) in [0,1), cartesian (M,3)) for one full unit cell.
    """
    ops = symop_matrices(cell.symops)
    all_frac = []
    all_elem = []
    all_label = []
    all_source = []
    for R, t in ops:
        all_frac.append(frac @ R.T + t)
        all_elem.extend(elements)
        all_label.extend(labels)
        all_source.extend(range(len(frac)))
    cand = _wrap01(np.vstack(all_frac))

    # A special-position molecule can map one labeled site onto a differently labeled site,
    # so exact cross-site coincidences must collapse. The tighter cross-site tolerance and
    # element guard preserve distinct sites near a symmetry plane. Greedy representatives
    # avoid transitive A~B~C over-merging.
    lattice = cell.lattice_vectors
    groups: dict[str, list[int]] = {}
    keep = []
    for index, (element, source) in enumerate(zip(all_elem, all_source, strict=True)):
        key = str(element)
        representatives = groups.setdefault(key, [])
        if representatives:
            distances = _min_image_distances(cand[index] - cand[representatives], lattice)
            tolerances = np.asarray([
                _DEDUP_CART_TOL_A if all_source[other] == source
                else _PREEXPANDED_DEDUP_CART_TOL_A
                for other in representatives
            ])
            if np.any(distances < tolerances):
                continue
        representatives.append(index)
        keep.append(index)

    keep_frac = cand[keep]
    keep_elem = [all_elem[i] for i in keep]
    keep_label = [all_label[i] for i in keep]
    keep_cart = keep_frac @ cell.orth.T
    return keep_elem, keep_label, keep_frac, keep_cart


def replicate(cart_cell: np.ndarray, cell: UnitCell, k: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile one cell's cartesian atoms into a (2k+1)^3 supercell.

    Returns (coords (N,3), base_index (N,), shift_magnitude (N,)) where base_index maps each
    supercell atom to its index within the input cell and shift_magnitude = max(|shift| per axis).
    """
    lattice = cell.lattice_vectors  # rows a,b,c
    rng = range(-k, k + 1)
    shifts = np.array([(i, j, l) for i in rng for j in rng for l in rng])
    offsets = shifts @ lattice                                    # (S,3)
    coords = (cart_cell[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    m = len(cart_cell)
    base_index = np.tile(np.arange(m), len(shifts))
    shift_mag = np.repeat(np.abs(shifts).max(axis=1), m)
    return coords, base_index, shift_mag


def lattice_shifts(cell: UnitCell, k: int) -> np.ndarray:
    """Cartesian translation vectors for all integer cell shifts in [-k, k]^3."""
    rng = np.arange(-k, k + 1)
    # Same C-order enumeration as the nested loops it replaces, without building (2k+1)^3 tuples.
    shifts = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3)
    return shifts @ cell.lattice_vectors
