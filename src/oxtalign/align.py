"""Rigid superposition (Kabsch/SVD), RMSD, and radius of gyration.

Convention: coordinates are (n, 3) arrays of row vectors. A rigid transform maps a point set
P onto Q as ``P @ R.T + t``. ``kabsch`` returns the optimal proper rotation R (det = +1, no
reflection) and translation t. Enantiomer/mirror relationships are handled upstream by inverting
coordinates (see matching), not by allowing an improper rotation here.
"""
from __future__ import annotations

import math

import numpy as np

_I3 = np.eye(3)

# kabsch calls svd ~3x10^4 times per comparison on a single 3x3, where numpy's wrapper costs ~3 us
# of preamble. Bind the LAPACK gufunc it dispatches to, checked bit-identical on import.
try:                                                   # pragma: no branch
    from numpy.linalg import _umath_linalg as _ula

    _PROBE = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    if not all(np.array_equal(a, b) for a, b in
               zip(_ula.svd_f(_PROBE, signature="d->ddd"), np.linalg.svd(_PROBE),
                   strict=True)):
        raise ImportError("svd gufunc disagrees with numpy.linalg.svd")

    def _svd(H):
        return _ula.svd_f(H, signature="d->ddd")
except Exception:                                      # pragma: no cover - numpy layout change
    def _svd(H):
        return np.linalg.svd(H)


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal proper rotation R and translation t minimizing ||P @ R.T + t - Q||.

    Returns (R (3,3), t (3,)). Handles single-atom and degenerate (collinear/planar) inputs.

    This is called ~10^5 times per comparison on 3-70 atom molecules, so it avoids numpy's
    high-level wrappers (``mean``, ``diag``, ``linalg.det``) in favour of the equivalent primitives;
    every result is bit-identical to the textbook formulation.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    n = len(P)
    cP = np.add.reduce(P, axis=0) / n          # == P.mean(axis=0)
    cQ = np.add.reduce(Q, axis=0) / n
    if n < 2:
        return _I3.copy(), cQ - cP
    Pc = P - cP
    Qc = Q - cQ
    H = Pc.T @ Qc
    U, _, Vt = _svd(H)
    R = Vt.T @ U.T
    d = _sign_det3(R)                          # reflection guard: force a proper rotation
    if d != 1.0:                               # V @ I3 == V bitwise, so the common (already proper)
        D = _I3.copy()                         # case is exactly V U^T and skips two 3x3 products
        D[2, 2] = d
        R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def _sign_det3(M: np.ndarray) -> float:
    """sign(det M) for a 3x3 (near-)orthogonal M, without the LAPACK round trip of ``linalg.det``."""
    m = M.ravel().tolist()
    det = (m[0] * (m[4] * m[8] - m[5] * m[7])
           - m[1] * (m[3] * m[8] - m[5] * m[6])
           + m[2] * (m[3] * m[7] - m[4] * m[6]))
    return 1.0 if det > 0 else (-1.0 if det < 0 else det)   # 0 / nan pass through like np.sign


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(coords) @ R.T + t


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Plain RMSD between two equally-sized, already-corresponded point sets (no fitting)."""
    diff = np.asarray(P, dtype=float) - np.asarray(Q, dtype=float)
    # == sqrt(mean(sum(diff**2, axis=1))) via the ufunc primitives the wrappers reduce to; math.sqrt
    # is the same correctly-rounded IEEE op on a float64 scalar, without the 0-d array round trip.
    return math.sqrt(np.add.reduce(np.add.reduce(diff * diff, axis=1)) / len(diff))


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal proper superposition of P onto Q."""
    R, t = kabsch(P, Q)
    return rmsd(apply_transform(P, R, t), Q)


def radius_of_gyration(coords: np.ndarray) -> float:
    """sqrt of the trace of the gyration tensor = sqrt(mean squared distance to centroid)."""
    c = np.asarray(coords, dtype=float)
    n = len(c)
    d = c - np.add.reduce(c, axis=0) / n       # == c.mean(axis=0), without the wrapper
    return math.sqrt(np.add.reduce(np.add.reduce(d * d, axis=1)) / n)
