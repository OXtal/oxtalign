"""Parsing and molecule-assembly correctness, including the pre-expanded-cell dedup guard."""
from collections import Counter

import numpy as np
import pytest

from oxtalign.assemble import assemble
from oxtalign.cell import expand_unit_cell
from oxtalign.compare import load
from oxtalign.io_cif import read
from oxtalign.model import RawStructure, UnitCell


def test_parse_all_experimental_without_error(exp):
    """Every experimental CIF parses, has a cell, and yields physically sane parameters."""
    for p in sorted(exp.rglob("*.cif")):
        raw = read(str(p))
        assert raw.cell is not None, p
        assert all(x > 0 for x in (raw.cell.a, raw.cell.b, raw.cell.c)), p
        assert len(raw) > 0, p


def test_element_normalization_in_predicted(pred):
    raw = read(str(pred / "RUSXEM_seed0_sample_0.cif"))
    assert raw.cell is None and raw.explicit_bonds
    assert {"Br", "Co", "C", "N", "O"} <= set(raw.elements)   # 'BR'/'CO' normalized to Br/Co


def test_holsux_preexpanded_dedup(exp):
    """HOLSUX stores the already-expanded cell (labels x4); expansion+dedup must NOT inflate it."""
    raw = read(str(exp / "HOLSUX" / "HOLSUX.cif"))
    frac = raw.cell.cart_to_frac(raw.coords)
    el, _, _, _ = expand_unit_cell(raw.elements, raw.labels, frac, raw.cell)
    assert len(el) == len(raw), "dedup should collapse the regenerated positions back to the input"


def test_caprylunit_expands(exp):
    """CAPRYL stores only the asymmetric unit; the 4 symops genuinely quadruple it."""
    raw = read(str(exp / "CAPRYL.cif"))
    frac = raw.cell.cart_to_frac(raw.coords)
    el, _, _, _ = expand_unit_cell(raw.elements, raw.labels, frac, raw.cell)
    assert len(el) == 4 * len(raw)


def _p1(side=100.0):
    orth = np.eye(3) * side
    return UnitCell(side, side, side, 90, 90, 90, ["x,y,z"], orth, np.linalg.inv(orth))


def test_symmetry_dedup_never_merges_distinct_nearby_sites():
    frac = np.asarray([[0, 0, 0], [0.00015, 0, 0]])
    elements, labels, *_ = expand_unit_cell(["C", "C"], ["C1", "C2"], frac, _p1())
    assert elements == ["C", "C"]
    assert labels == ["C1", "C2"]

    elements, *_ = expand_unit_cell(["C", "O"], ["A", "A"], frac, _p1())
    assert elements == ["C", "O"]


def test_repeated_label_dedup_does_not_merge_a_nearby_chain_transitively():
    frac = np.asarray([[0, 0, 0], [0.00015, 0, 0], [0.00030, 0, 0]])
    elements, *_ = expand_unit_cell(["C"] * 3, ["C1"] * 3, frac, _p1())
    assert len(elements) == 3


def test_special_position_can_map_between_differently_labelled_sites():
    cell = _p1()
    cell.symops = ["x,y,z", "-x,-y,-z"]
    frac = np.asarray([[0.1, 0, 0], [0.9, 0, 0]])
    elements, labels, *_ = expand_unit_cell(["C", "C"], ["C1", "C2"], frac, cell)
    assert elements == ["C", "C"]
    assert labels == ["C1", "C2"]


@pytest.mark.parametrize("rel,expected", [
    ("CAPRYL.cif", {"C8N1O1": 4}),
    ("HOLSUX/HOLSUX.cif", {"C4N1": 4, "O3P1": 4}),
    ("HOLSUX/HOLSUX01.cif", {"C4N1": 4, "O3P1": 4}),
    ("RUSXEM.cif", {"Br4C23Co1N2O4": 4, "C6N1": 4}),
    ("XAFQON.cif", {"C4N2O2S1": 4, "Cl1": 4, "O1": 4}),
])
def test_molecule_counts_and_species(exp, rel, expected):
    s = load(str(exp / rel))
    got = Counter(m.species_key.split("|")[0] for m in s.molecules)
    assert dict(got) == expected


def test_same_compound_same_composition(exp):
    """The two HOLSUX polymorphs must assemble to identical molecular composition."""
    a = Counter(m.species_key for m in load(str(exp / "HOLSUX" / "HOLSUX.cif")).molecules)
    b = Counter(m.species_key for m in load(str(exp / "HOLSUX" / "HOLSUX01.cif")).molecules)
    assert a == b


def test_predicted_clusters_and_hydrogen_dropped(pred):
    s = load(str(pred / "CAPRYL_seed0_sample_0.cif"))
    assert len(s.molecules) == 30 and s.cell is None
    assert all("H" not in m.elements for m in s.molecules)   # H dropped by default


def test_assembly_rejects_when_occupancy_filter_removes_every_atom():
    raw = RawStructure(
        ["C"],
        np.zeros((1, 3)),
        ["C1"],
        np.asarray([0.287]),
        _p1(),
        None,
        "all-minor-occupancy",
    )
    with pytest.raises(ValueError, match="no atoms remain after .* filtering"):
        assemble(raw)


def test_species_key_matches_across_with_and_without_hydrogen(exp, pred):
    e = load(str(exp / "CAPRYL.cif")).molecules[0].species_key
    p = load(str(pred / "CAPRYL_seed0_sample_0.cif")).molecules[0].species_key
    assert e == p


def test_symmetry_dedup_is_periodic_across_the_cell_boundary():
    """Sites 0.002 A apart *through* the cell boundary (9.998 A apart in the raw cell) coincide."""
    orth = np.eye(3) * 10.0
    cell = UnitCell(10.0, 10.0, 10.0, 90, 90, 90, ["x,y,z"], orth, np.linalg.inv(orth))
    frac = np.asarray([[0.0001, 0.5, 0.5], [0.9999, 0.5, 0.5]])
    elements, labels, *_ = expand_unit_cell(["C", "C"], ["C1", "C2"], frac, cell)
    assert elements == ["C"] and labels == ["C1"]
    # ... but not when the through-boundary gap exceeds the cross-site tolerance (0.005 A).
    frac = np.asarray([[0.001, 0.5, 0.5], [0.999, 0.5, 0.5]])
    elements, *_ = expand_unit_cell(["C", "C"], ["C1", "C2"], frac, cell)
    assert elements == ["C", "C"]
