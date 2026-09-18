"""Core data model: UnitCell, RawStructure, Molecule, Structure, ComparisonResult."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class UnitCell:
    """A crystallographic unit cell with frac<->cart transforms and symmetry operators."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    symops: list[str]          # xyz triplet strings, e.g. "1/2+x,-y,1/2+z"
    orth: np.ndarray           # (3,3) fractional -> cartesian
    frac: np.ndarray           # (3,3) cartesian -> fractional

    def frac_to_cart(self, f: np.ndarray) -> np.ndarray:
        return np.asarray(f) @ self.orth.T

    def cart_to_frac(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ self.frac.T

    @property
    def lattice_vectors(self) -> np.ndarray:
        """Rows are the a, b, c lattice vectors in cartesian coordinates."""
        return self.orth.T

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.orth)))


@dataclass
class RawStructure:
    """Atoms as parsed from a CIF, before symmetry expansion / molecule assembly.

    Coordinates are CARTESIAN (already converted from fractional when a cell is present).
    """

    elements: list[str]
    coords: np.ndarray                       # (n,3) cartesian
    labels: list[str]
    occupancies: np.ndarray                  # (n,)
    cell: UnitCell | None
    explicit_bonds: list[tuple[int, int]] | None  # global atom indices, or None
    source: str

    def __len__(self) -> int:
        return len(self.elements)


@dataclass(slots=True)
class Molecule:
    """A whole molecule (one connected component), in cartesian coordinates."""

    elements: tuple[str, ...]
    coords: np.ndarray                       # (n,3) cartesian
    labels: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]       # local atom indices
    species_key: str = ""
    # `coords` is never mutated in place, so the centroid is memoizable; out of eq/repr.
    _centroid_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def centroid(self) -> np.ndarray:
        c = self._centroid_cache
        if c is None:
            c = self._centroid_cache = self.coords.mean(axis=0)
        return c

    def formula(self) -> str:
        return "".join(f"{el}{n}" for el, n in sorted(Counter(self.elements).items()))

    def translated(self, shift: np.ndarray) -> Molecule:
        return Molecule(self.elements, self.coords + shift, self.labels, self.bonds, self.species_key)


@dataclass
class Structure:
    """Assembled structure: whole molecules (one unit cell's worth if periodic) + optional cell."""

    molecules: list[Molecule]
    cell: UnitCell | None
    source: str
    # Derived tables (lattice-image tables, per-molecule radii) memoized by cluster.py. Every entry
    # is a pure function of the fields above, so it is excluded from equality and dropped on pickling
    # -- batch workers rebuild it lazily rather than receiving it over the process boundary.
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __getstate__(self) -> dict:
        return {**self.__dict__, "cache": {}}


@dataclass
class ComparisonResult:
    """Result of an RMSD_n packing comparison."""

    rmsd_n: float
    n: int                                   # requested shell size
    n_matched: int                           # molecules actually matched
    status: str                              # "full" | "partial" | "failed"
    per_molecule_rmsd: list[float] = field(default_factory=list)
    transform: np.ndarray | None = None   # (4,4) mapping B onto A
    radius_of_gyration: tuple[float, float] = (float("nan"), float("nan"))
    rmsd_1: float | None = None           # central-molecule conformer/lattice check
    inverted: bool = False                   # True if the optimal fit used an improper rotation
    central_a: int = -1                      # index of A's central molecule the match anchored on
    message: str = ""

    def __repr__(self) -> str:  # concise, readable
        chi = " inverted" if self.inverted else ""
        return (f"ComparisonResult(rmsd_{self.n}={self.rmsd_n:.4f} "
                f"n_matched={self.n_matched}/{self.n} status={self.status}{chi})")
