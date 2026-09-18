"""Element data: covalent radii (bond perception) and van der Waals radii (clash detection).

Covalent radii are from Cordero et al., Dalton Trans. 2008, 2832 (single-bond, Å).
van der Waals radii are Bondi 1964 with Alvarez 2013 extensions (Å).
A generous fallback is used for any element not tabulated so perception never crashes.
"""
from __future__ import annotations

# Cordero covalent radii (Å). Subset covering organics, halogens, common metals/organometallics.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92,
    "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75, "Ta": 1.70,
    "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36,
    "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "U": 1.96,
}

# van der Waals radii (Å).
VDW_RADII: dict[str, float] = {
    "H": 1.20, "He": 1.40, "Li": 1.82, "B": 1.92, "C": 1.70, "N": 1.55,
    "O": 1.52, "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73, "Al": 1.84,
    "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88, "K": 2.75,
    "Ca": 2.31, "Ni": 1.97, "Cu": 1.40, "Zn": 1.39, "Ga": 1.87, "Ge": 2.11,
    "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02, "Pd": 1.63, "Ag": 1.72,
    "Cd": 1.58, "Sn": 2.17, "Te": 2.06, "I": 1.98, "Xe": 2.16, "Pt": 1.75,
    "Au": 1.66, "Hg": 1.55, "Pb": 2.02, "U": 1.86,
}

# Group 1/2 metals, whose Cordero radii call a 2.8 A cation-anion contact a bond. Used only by the
# percolation fallback in assemble.py, never as the first perception.
IONIC_ELEMENTS: frozenset[str] = frozenset(
    {"Li", "Na", "K", "Rb", "Cs", "Fr", "Be", "Mg", "Ca", "Sr", "Ba", "Ra"})

_FALLBACK_COVALENT = 1.50
_FALLBACK_VDW = 1.80


def covalent_radius(element: str) -> float:
    return COVALENT_RADII.get(element, _FALLBACK_COVALENT)


def bonding_radius(element: str) -> float:
    """Covalent radius for bond perception; zero for the ionic metals (see IONIC_ELEMENTS)."""
    return 0.0 if element in IONIC_ELEMENTS else covalent_radius(element)


def vdw_radius(element: str) -> float:
    return VDW_RADII.get(element, _FALLBACK_VDW)


def is_hydrogen(element: str) -> bool:
    return element == "H" or element == "D"
