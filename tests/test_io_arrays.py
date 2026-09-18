"""In-memory input builds the same structures as the CIF reader, without importing ASE or pymatgen."""
from types import SimpleNamespace

import numpy as np
import pytest

from oxtalign import compare, from_arrays, from_ase, from_pdb, from_pymatgen, from_xyz, load
from oxtalign.io_cif import read


def _same_structure(s1, s2, n=15):
    assert len(s1.molecules) == len(s2.molecules)
    assert sorted(m.species_key for m in s1.molecules) == sorted(m.species_key for m in s2.molecules)
    r = compare(s1, s2, n=n)
    assert r.status == "full" and r.n_matched == n and r.rmsd_n < 1e-6


def test_from_arrays_asymmetric_unit_with_symops_matches_cif(exp):
    path = str(exp / "CAPRYL.cif")
    raw = read(path)
    s = from_arrays(raw.elements, raw.cell.cart_to_frac(raw.coords),
                    cell=(raw.cell.a, raw.cell.b, raw.cell.c, raw.cell.alpha, raw.cell.beta, raw.cell.gamma),
                    symops=raw.cell.symops, labels=raw.labels, occupancies=raw.occupancies)
    _same_structure(s, load(path))


def test_from_arrays_lattice_matrix_and_cartesian_coordinates(exp):
    path = str(exp / "HOLSUX" / "HOLSUX.cif")                # stores the full P1-expanded cell
    ref = load(path)
    raw = read(path)
    lattice = raw.cell.lattice_vectors                        # rows a, b, c in the CIF's Cartesian frame
    s = from_arrays(raw.elements, raw.coords, cell=lattice, fractional=False, symops=raw.cell.symops)
    _same_structure(s, ref)


def test_from_arrays_cluster_with_explicit_bonds(pred):
    path = str(pred / "CAPRYL_seed0_sample_0.cif")
    raw = read(path)
    s = from_arrays(raw.elements, raw.coords, bonds=raw.explicit_bonds, labels=raw.labels)
    _same_structure(s, load(path))
    assert s.cell is None


def test_from_ase_duck_typed(exp):
    path = str(exp / "CAPRYL.cif")
    raw = read(path)
    frac = raw.cell.cart_to_frac(raw.coords)
    # expand to one full cell the way ase would hold it (P1), via our own reader as the oracle
    from oxtalign.cell import expand_unit_cell
    el, _, frac_cell, _ = expand_unit_cell(raw.elements, raw.labels, frac, raw.cell)
    atoms = SimpleNamespace(cell=raw.cell.lattice_vectors, pbc=np.array([True, True, True]),
                            get_chemical_symbols=lambda: list(el),
                            get_scaled_positions=lambda wrap=False: np.asarray(frac_cell),
                            get_positions=lambda: np.asarray(frac_cell) @ raw.cell.lattice_vectors)
    _same_structure(from_ase(atoms), load(path))


def test_from_pymatgen_duck_typed(exp):
    path = str(exp / "HOLSUX" / "HOLSUX.cif")
    raw = read(path)
    structure = SimpleNamespace(lattice=SimpleNamespace(matrix=raw.cell.lattice_vectors),
                                frac_coords=raw.cell.cart_to_frac(raw.coords),
                                species=[SimpleNamespace(symbol=e) for e in raw.elements])
    _same_structure(from_pymatgen(structure), load(path))
    molecule = SimpleNamespace(cart_coords=load(path).molecules[0].coords,
                               species=[SimpleNamespace(symbol=e) for e in load(path).molecules[0].elements])
    s = from_pymatgen(molecule)
    assert s.cell is None and len(s.molecules) == 1


def test_from_arrays_rejects_inconsistent_input():
    import pytest
    with pytest.raises(ValueError):
        from_arrays(["C", "C"], np.zeros((3, 3)))
    with pytest.raises(ValueError):
        from_arrays(["C"], np.zeros((1, 3)), fractional=True)
    with pytest.raises(ValueError):
        from_arrays(["C"], np.zeros((1, 3)), cell=(1, 2, 3))


def _write_xyz(path, elements, coords, lattice=None):
    if lattice is None:
        header = "cluster"
    else:
        vectors = " ".join(f"{v:.10f}" for v in np.ravel(lattice))
        header = f'Lattice="{vectors}" Properties=species:S:1:pos:R:3'
    body = "".join(f"{e} {x:.10f} {y:.10f} {z:.10f}\n"
                   for e, (x, y, z) in zip(elements, coords, strict=True))
    path.write_text(f"{len(elements)}\n{header}\n{body}")
    return str(path)


def _write_pdb(path, elements, coords, cell=None, spacegroup="P 1"):
    lines = []
    if cell is not None:
        a, b, c, alpha, beta, gamma = cell
        lines.append(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} "
                     f"{spacegroup:<11s}   1")
    for i, (e, (x, y, z)) in enumerate(zip(elements, coords, strict=True), 1):
        # Names deliberately repeat across molecules, as bulk-written PDBs do.
        lines.append(f"HETATM{i:5d} {e[:3] + str(i % 100):<4s} LIG A   1    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2s}")
    path.write_text("\n".join(lines + ["END", ""]))
    return str(path)


def test_from_xyz_extended_lattice_matches_cif(exp, tmp_path):
    """extended-XYZ Lattice="..." (rows a, b, c) reproduces a P1-expanded cell exactly."""
    path = str(exp / "HOLSUX" / "HOLSUX.cif")
    raw = read(path)
    xyz = _write_xyz(tmp_path / "cell.xyz", raw.elements, raw.coords, raw.cell.lattice_vectors)
    _same_structure(from_xyz(xyz), load(path))


def test_from_xyz_without_lattice_is_a_cluster(pred, tmp_path):
    path = str(pred / "CAPRYL_seed0_sample_0.cif")
    raw = read(path)
    s = from_xyz(_write_xyz(tmp_path / "cluster.xyz", raw.elements, raw.coords))
    assert s.cell is None
    _same_structure(s, load(path))


def test_from_xyz_cell_argument_overrides_a_missing_lattice(exp, tmp_path):
    path = str(exp / "HOLSUX" / "HOLSUX.cif")
    raw = read(path)
    xyz = _write_xyz(tmp_path / "bare.xyz", raw.elements, raw.coords)     # no Lattice on the comment line
    _same_structure(from_xyz(xyz, cell=raw.cell.lattice_vectors), load(path))


def test_from_pdb_cryst1_spacegroup_expands_the_asymmetric_unit(exp, tmp_path):
    """CAPRYL's CIF stores only the asymmetric unit; CRYST1's space group must regenerate the rest."""
    path = str(exp / "CAPRYL.cif")
    raw = read(path)
    cell = (raw.cell.a, raw.cell.b, raw.cell.c, raw.cell.alpha, raw.cell.beta, raw.cell.gamma)
    pdb = _write_pdb(tmp_path / "asym.pdb", raw.elements, raw.coords, cell=cell, spacegroup="C 1 c 1")
    reference = load(path)
    s = from_pdb(pdb)
    assert len(s.molecules) == len(reference.molecules)
    # PDB stores coordinates to 0.001 A, so the overlay is exact only to that rounding.
    r = compare(s, reference, n=15)
    assert r.status == "full" and r.n_matched == 15 and r.rmsd_n < 1e-2


def test_from_pdb_without_cryst1_is_a_cluster_and_keeps_every_atom(pred, tmp_path):
    """Repeated atom names across molecules must not be deduplicated away (gemmi's altloc removal does)."""
    path = str(pred / "CAPRYL_seed0_sample_0.cif")
    raw = read(path)
    s = from_pdb(_write_pdb(tmp_path / "cluster.pdb", raw.elements, raw.coords))
    assert s.cell is None
    assert sum(m.n_atoms for m in s.molecules) == sum(m.n_atoms for m in load(path).molecules)
    assert len(s.molecules) == len(load(path).molecules)


def test_from_pdb_keeps_the_first_alternative_conformation(tmp_path):
    pdb = tmp_path / "altloc.pdb"
    pdb.write_text(
        "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  O1ALIG A   1       1.400   0.000   0.000  0.60  0.00           O\n"
        "HETATM    3  O1BLIG A   1       1.500   0.100   0.000  0.40  0.00           O\n"
        "END\n")
    s = from_pdb(str(pdb))
    assert [m.n_atoms for m in s.molecules] == [2]            # C plus the A conformer only
    assert s.molecules[0].elements == ("C", "O")


def test_from_xyz_rejects_a_truncated_file(tmp_path):
    bad = tmp_path / "short.xyz"
    bad.write_text("3\ncomment\nC 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="declares 3 atoms"):
        from_xyz(str(bad))
