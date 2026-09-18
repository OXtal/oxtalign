"""In-memory input: build a Structure from plain arrays, ASE / pymatgen objects, or an XYZ / PDB file.

Nothing here imports ASE or pymatgen — `from_ase` and `from_pymatgen` only read the attributes those
objects expose — so the package keeps its three runtime dependencies. Every entry point funnels into
`from_arrays`, so the assembly, symmetry and bond perception are identical to the CIF path.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import gemmi
import numpy as np

from .assemble import assemble
from .io_cif import _normalize_element
from .model import RawStructure, Structure, UnitCell


def _unit_cell(cell, symops) -> UnitCell:
    cell = np.asarray(cell, dtype=float)
    if cell.shape == (6,):
        a, b, c, alpha, beta, gamma = cell
    elif cell.shape == (3, 3):                      # rows are the a, b, c lattice vectors
        a, b, c = np.linalg.norm(cell, axis=1)

        def angle(u, v):
            cosine = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
            return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))

        alpha, beta, gamma = angle(cell[1], cell[2]), angle(cell[0], cell[2]), angle(cell[0], cell[1])
    else:
        raise ValueError("cell must be (a, b, c, alpha, beta, gamma) or a 3x3 lattice matrix (rows a, b, c)")
    g = gemmi.UnitCell(float(a), float(b), float(c), float(alpha), float(beta), float(gamma))
    orth = np.array(g.orth.mat.tolist())
    frac = np.array(g.frac.mat.tolist())
    ops = list(symops) if symops else ["x,y,z"]
    return UnitCell(g.a, g.b, g.c, g.alpha, g.beta, g.gamma, ops, orth, frac)


def from_arrays(elements: Sequence[str], coords, *, cell=None, fractional: bool | None = None,
                symops: Sequence[str] | None = None, bonds=None, labels: Sequence[str] | None = None,
                occupancies=None, include_hydrogens: bool = False, drop_minor_occupancy: bool = True,
                source: str = "arrays") -> Structure:
    """Assemble a Structure from element symbols and coordinates (numpy only).

    ``cell``: ``None`` for an isolated cluster (Cartesian coordinates), else ``(a, b, c, alpha, beta, gamma)``
    or a 3x3 lattice matrix whose rows are the a, b, c vectors. With a cell, ``coords`` are fractional
    unless ``fractional=False``; pass one full unit cell, or the asymmetric unit together with ``symops``
    (xyz triplets such as ``"-x,1/2+y,1/2-z"``). ``bonds`` (index pairs) are used for clusters; periodic
    input is bond-perceived from covalent radii exactly like a CIF.
    """
    elements = [_normalize_element(str(e)) for e in elements]
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    n = len(elements)
    if len(coords) != n:
        raise ValueError(f"{n} elements but {len(coords)} coordinates")
    labels = [f"{e}{i + 1}" for i, e in enumerate(elements)] if labels is None else [str(x) for x in labels]
    occ = np.ones(n) if occupancies is None else np.asarray(occupancies, dtype=float).reshape(n)
    if cell is None:
        if fractional:
            raise ValueError("fractional coordinates need a cell")
        explicit = None if bonds is None else [(int(i), int(j)) for i, j in np.asarray(bonds).reshape(-1, 2)]
        raw = RawStructure(elements, coords, labels, occ, None, explicit, source)
    else:
        uc = _unit_cell(cell, symops)
        if fractional is None or fractional:
            cart = uc.frac_to_cart(coords)
        else:                                       # Cartesian in the caller's frame -> the cell's own frame
            lattice = np.asarray(cell, dtype=float)
            if lattice.shape == (3, 3):
                frac_coords = coords @ np.linalg.inv(lattice)
            else:
                frac_coords = uc.cart_to_frac(coords)
            cart = uc.frac_to_cart(frac_coords)
        raw = RawStructure(elements, cart, labels, occ, uc, None, source)
    return assemble(raw, include_hydrogens=include_hydrogens, drop_minor_occupancy=drop_minor_occupancy)


def from_ase(atoms, **kwargs) -> Structure:
    """Build from an ``ase.Atoms`` (duck-typed); periodic when ``pbc`` is set and the cell has volume."""
    elements = list(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell, dtype=float)
    if bool(np.any(atoms.pbc)) and cell.shape == (3, 3) and abs(np.linalg.det(cell)) > 1e-9:
        return from_arrays(elements, atoms.get_scaled_positions(wrap=False), cell=cell, fractional=True,
                           source=kwargs.pop("source", "ase.Atoms"), **kwargs)
    return from_arrays(elements, atoms.get_positions(), cell=None,
                       source=kwargs.pop("source", "ase.Atoms"), **kwargs)


def from_pymatgen(structure, **kwargs) -> Structure:
    """Build from a ``pymatgen`` ``Structure`` (periodic) or ``Molecule`` (cluster), duck-typed."""
    elements = [getattr(s, "symbol", None) or getattr(getattr(s, "element", s), "symbol", str(s))
                for s in structure.species]
    lattice = getattr(structure, "lattice", None)
    if lattice is not None:
        return from_arrays(elements, structure.frac_coords, cell=np.asarray(lattice.matrix, dtype=float),
                           fractional=True, source=kwargs.pop("source", "pymatgen.Structure"), **kwargs)
    return from_arrays(elements, structure.cart_coords, cell=None,
                       source=kwargs.pop("source", "pymatgen.Molecule"), **kwargs)


def _extxyz_lattice(comment: str) -> np.ndarray | None:
    """The extended-XYZ ``Lattice="ax ay az bx by bz cx cy cz"`` cell, if the comment line carries one."""
    match = re.search(r"""Lattice\s*=\s*["']([^"']*)["']""", comment, re.IGNORECASE)
    if match is None:
        return None
    values = np.fromstring(match.group(1), sep=" ")
    if values.size != 9:
        raise ValueError(f"extended-XYZ Lattice needs 9 numbers (rows a, b, c), got {values.size}")
    return values.reshape(3, 3)


def from_xyz(path, *, cell=None, **kwargs) -> Structure:
    """Read an XYZ or extended-XYZ file (the first frame; any columns after x, y, z are ignored).

    Coordinates are Cartesian. The cell is taken from an extended-XYZ ``Lattice="..."`` on the comment
    line (rows a, b, c) unless ``cell`` is given explicitly; with neither, the atoms are read as an
    isolated cluster and bonds are perceived from covalent radii.
    """
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: not an XYZ file")
    count = int(lines[0].split()[0])
    rows = [line.split() for line in lines[2:2 + count] if line.strip()]
    if len(rows) != count:
        raise ValueError(f"{path}: header declares {count} atoms, found {len(rows)}")
    coords = np.array([row[1:4] for row in rows], dtype=float)
    kwargs.setdefault("fractional", False)
    return from_arrays([row[0] for row in rows], coords,
                       cell=_extxyz_lattice(lines[1]) if cell is None else cell,
                       source=kwargs.pop("source", str(path)), **kwargs)


# What modelling codes write when a PDB has no real periodicity; gemmi reports the same for a file
# with no CRYST1 record at all.
_PDB_NO_CELL = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)


def from_pdb(path, **kwargs) -> Structure:
    """Read a PDB file (first model; the first of any alternative conformations).

    ``CRYST1`` supplies the cell and, through its space group, the symmetry operators, so a PDB holding
    only the asymmetric unit expands exactly like a small-molecule CIF. Without ``CRYST1`` — or with the
    ``1 1 1 90 90 90`` placeholder modelling codes emit — the file is read as an isolated cluster.
    """
    st = gemmi.read_structure(str(path))
    atoms = [atom for chain in st[0] for residue in chain for atom in residue]
    if not atoms:
        raise ValueError(f"{path}: no ATOM/HETATM records")
    # Keep one alternative conformation, chosen deterministically. gemmi's own
    # remove_alternative_conformations() deduplicates by atom name within a residue, which silently
    # drops real atoms from files that repeat names across the molecules of a single residue.
    alternates = sorted({atom.altloc for atom in atoms} - {"", "\x00"})
    if alternates:
        atoms = [atom for atom in atoms if atom.altloc in ("", "\x00", alternates[0])]
    elements = [atom.element.name for atom in atoms]
    coords = np.array([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in atoms], dtype=float)
    labels = [atom.name for atom in atoms]
    occupancies = [float(atom.occ) for atom in atoms]
    g = st.cell
    params = (g.a, g.b, g.c, g.alpha, g.beta, g.gamma)
    periodic = params != _PDB_NO_CELL and g.volume > 0
    symops = None
    if periodic and st.spacegroup_hm:
        try:
            symops = [op.triplet() for op in gemmi.SpaceGroup(st.spacegroup_hm).operations()]
        except (ValueError, RuntimeError):
            symops = None       # unreadable symbol: take the file as one full cell, as for a CIF
    return from_arrays(elements, coords, cell=params if periodic else None,
                       fractional=False, symops=symops, labels=labels, occupancies=occupancies,
                       source=kwargs.pop("source", str(path)), **kwargs)
