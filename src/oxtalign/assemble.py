"""Assemble a parsed RawStructure into whole molecules (a Structure).

Periodic inputs: expand the unit cell, tile a small supercell, perceive bonds, and read off
connected components seeded from the central cell — which yields whole molecules with atoms in
adjacent images correctly placed (no separate "unwrap" pass needed). The supercell is grown until
no central-cell molecule touches its outer boundary, guaranteeing completeness.

Cluster inputs (no cell): use explicit bonds (or perceive them) and split into components directly.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .cell import expand_unit_cell, replicate
from .connectivity import perceive_bonds
from .elements import bonding_radius, covalent_radius, is_hydrogen
from .model import Molecule, RawStructure, Structure

_START_K = 2
_MAX_K = 5


def assemble(raw: RawStructure, *, include_hydrogens: bool = False,
             drop_minor_occupancy: bool = True) -> Structure:
    keep = np.ones(len(raw), dtype=bool)
    if not include_hydrogens:
        keep &= np.array([not is_hydrogen(e) for e in raw.elements])
    if drop_minor_occupancy:
        keep &= raw.occupancies >= 0.5
        keep &= np.array(["?" not in lab for lab in raw.labels])
    idx = np.where(keep)[0]
    elements = [raw.elements[i] for i in idx]
    labels = [raw.labels[i] for i in idx]
    coords = raw.coords[idx]
    if not len(idx):
        raise ValueError(
            "no atoms remain after hydrogen/minor-occupancy filtering; "
            "choose an explicit disorder configuration or disable the relevant filter"
        )

    if raw.cell is None:
        mols = _assemble_cluster(elements, labels, coords, raw, idx)
        struct = Structure(mols, None, raw.source)
    else:
        frac = raw.cell.cart_to_frac(coords)
        try:
            mols = _assemble_periodic(elements, labels, frac, raw.cell)
        except ValueError:
            # Bond graph percolated: usually a group 1/2 cation. Retrying with ionic.
            mols = _assemble_periodic(elements, labels, frac, raw.cell, ionic=True)
        struct = Structure(mols, raw.cell, raw.source)
    for m in struct.molecules:
        m.species_key = m.formula()       # formula is robust to flexible-molecule bond-perception noise
    return struct


def _components(n: int, bonds) -> np.ndarray:
    bonds = np.asarray(bonds, dtype=int)
    if len(bonds) == 0:
        return np.arange(n)
    adj = coo_matrix((np.ones(len(bonds)), (bonds[:, 0], bonds[:, 1])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


def _build_molecule(node_idx: list[int], coords, elements, labels, bonds_among) -> Molecule:
    """Build a Molecule from a set of atom indices and the bonds among them."""
    node_idx = sorted(node_idx)
    local = {g: i for i, g in enumerate(node_idx)}
    els = tuple(elements[g] for g in node_idx)
    labs = tuple(labels[g] for g in node_idx)
    crd = np.array([coords[g] for g in node_idx])
    mbonds = tuple(sorted({(min(local[a], local[b]), max(local[a], local[b]))
                           for a, b in bonds_among}))
    return Molecule(els, crd, labs, mbonds)


def _assemble_cluster(elements, labels, coords, raw, idx) -> list[Molecule]:
    if raw.explicit_bonds is not None:
        old_to_new = {int(g): i for i, g in enumerate(idx)}
        bonds = [(old_to_new[a], old_to_new[b]) for a, b in raw.explicit_bonds
                 if a in old_to_new and b in old_to_new]
    else:
        bonds = perceive_bonds(coords, elements)
    comp = _components(len(elements), bonds)
    by_comp = defaultdict(list)
    for i, c in enumerate(comp):
        by_comp[c].append(i)
    bonds_by_comp = defaultdict(list)
    for a, b in bonds:
        bonds_by_comp[comp[a]].append((a, b))
    return [_build_molecule(nodes, coords, elements, labels, bonds_by_comp[c])
            for c, nodes in sorted(by_comp.items())]


def _assemble_periodic(elements, labels, frac, cell, ionic=False) -> list[Molecule]:
    """Return the whole molecules whose centroid lies in the reference cell (exactly Z of them).

    A molecule broken across a cell boundary has atoms in the reference cell that belong to
    *different* whole-molecule images; assigning each complete molecule to the cell containing its
    centroid counts every molecule exactly once.
    """
    el_cell, lab_cell, _, cart_cell = expand_unit_cell(elements, labels, frac, cell)
    # One cell's radii, tiled by base index: the k = 5 supercell can carry >10^5 atoms, and
    # re-deriving their radii from element strings cost more than the bond search itself.
    radius = bonding_radius if ionic else covalent_radius
    radius_of = {e: radius(e) for e in set(el_cell)}
    radii_cell = np.fromiter(map(radius_of.__getitem__, el_cell), dtype=float, count=len(el_cell))

    for k in range(_START_K, _MAX_K + 1):
        coords_sc, base_index, shift_mag = replicate(cart_cell, cell, k)
        # fast=True is safe: this feeds only connected_components, which canonicalizes edge order.
        comp = _components(len(base_index),
                           perceive_bonds(coords_sc, None, fast=True,
                                          radii=radii_cell[base_index]))
        ncomp = comp.max() + 1

        # Per-component centroid and max shift, vectorized; keep only components centred in the cell.
        sums = np.zeros((ncomp, 3))
        np.add.at(sums, comp, coords_sc)
        centroids = sums / np.bincount(comp, minlength=ncomp)[:, None]
        max_shift = np.zeros(ncomp)
        np.maximum.at(max_shift, comp, shift_mag)
        central = np.flatnonzero(np.all(np.floor(cell.cart_to_frac(centroids) + 1e-6) == 0, axis=1))
        if np.any(max_shift[central] >= k):               # a central molecule touches ring -> grow
            continue

        nodes_by_comp = _group_indices(comp, ncomp)
        molecules = [_build_molecule_periodic(nodes_by_comp[c], coords_sc, el_cell, lab_cell,
                                              base_index) for c in central]
        return molecules
    raise ValueError(f"could not assemble complete molecules within {_MAX_K} cells "
                     f"(molecule larger than supercell?)")


def _group_indices(labels: np.ndarray, ncomp: int) -> list[np.ndarray]:
    """For each component label, the array of atom indices belonging to it (single argsort pass)."""
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(ncomp + 1))
    return [order[bounds[c]:bounds[c + 1]] for c in range(ncomp)]


def _build_molecule_periodic(nodes, coords_sc, el_cell, lab_cell, base_index) -> Molecule:
    """Build a whole molecule from its supercell nodes, indexed by base (in-cell) atom.

    Bonds are re-perceived on the assembled (unwrapped) coordinates — cheap, and gives the correct
    intramolecular graph without threading the supercell bond list through.
    """
    node_of = {int(base_index[nd]): nd for nd in nodes}      # one node per base atom (whole molecule)
    bases = sorted(node_of)
    els = tuple(el_cell[b] for b in bases)
    labs = tuple(lab_cell[b] for b in bases)
    crd = np.array([coords_sc[node_of[b]] for b in bases])
    bonds = tuple((int(a), int(b)) for a, b in perceive_bonds(crd, list(els)))
    return Molecule(els, crd, labs, bonds)
