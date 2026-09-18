"""CIF reader: disorder-group parsing, dominant-configuration selection, occupancy fidelity."""
import gemmi
import numpy as np
import pytest

from oxtalign import io_cif

_CIF = """\
data_disorder
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_cell_formula_units_Z 1
_chemical_formula_sum 'C N O'
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_charge
_atom_site_disorder_assembly
_atom_site_disorder_group
O1 O 0.5 0.5 0.5 1.0 0 . .
C1A C 0.1 0.1 0.1 0.6 0 A 1
C1B C 0.2 0.1 0.1 0.4 0 A 2
N1A N 0.3 0.3 0.3 0.5 0 B 1
N1B N 0.4 0.3 0.3 0.5 0 B 2
"""


def _block(text=_CIF):
    return gemmi.cif.read_string(text).sole_block()


def test_disorder_selects_one_dominant_group_per_assembly(tmp_path):
    path = tmp_path / "disorder.cif"
    path.write_text(_CIF)

    assert io_cif.dominant_disorder_labels(_block()) == {"O1", "C1A", "N1A"}
    raw = io_cif.read(str(path), disorder="dominant")
    assert raw.labels == ["O1", "C1A", "N1A"]
    assert raw.occupancies.tolist() == [1.0, 0.6, 0.5]

    deposited = io_cif.read(str(path))                      # default: every deposited site
    assert deposited.labels == ["O1", "C1A", "C1B", "N1A", "N1B"]


def test_disorder_model_enumerates_joint_configurations():
    model = io_cif.disorder_model(_block())
    assert model.ordered_labels == frozenset({"O1"})
    assert [a.assembly_id for a in model.assemblies] == ["A", "B"]
    assert [[g.group_id for g in a.groups] for a in model.assemblies] == [["1", "2"], ["1", "2"]]
    np.testing.assert_allclose([g.weight for g in model.assemblies[0].groups], [0.6, 0.4])
    assert model.labels_for((0, 1)) == frozenset({"O1", "C1A", "N1B"})
    assert model.labels_for((1, 0)) == frozenset({"O1", "C1B", "N1A"})
    with pytest.raises(ValueError):
        model.labels_for((0,))


def test_selected_labels_overrides_the_disorder_policy(tmp_path):
    path = tmp_path / "disorder.cif"
    path.write_text(_CIF)
    raw = io_cif.read(str(path), selected_labels={"O1", "C1B"})
    assert raw.labels == ["O1", "C1B"]
    with pytest.raises(ValueError):
        io_cif.read(str(path), selected_labels={"O1"}, disorder="dominant")


def test_reader_preserves_explicit_zero_occupancy(tmp_path):
    path = tmp_path / "zero-occupancy.cif"
    path.write_text(_CIF.replace("O1 O 0.5 0.5 0.5 1.0", "O1 O 0.5 0.5 0.5 0.0"))
    raw = io_cif.read(str(path), disorder="dominant")
    assert raw.occupancies.tolist() == [0.0, 0.6, 0.5]


def test_ungrouped_partial_occupancy_is_flagged_not_resolved():
    text = _CIF
    for token in (" A 1", " B 1", " A 2", " B 2"):
        text = text.replace(token, " . .")
    model = io_cif.disorder_model(_block(text))
    assert not model.assemblies
    assert model.has_ungrouped_partial is True
    assert io_cif.dominant_disorder_labels(_block(text)) is None


def test_malformed_group_occupancy_falls_back_to_uniform_weights():
    model = io_cif.disorder_model(_block(_CIF.replace("C1A C 0.1 0.1 0.1 0.6", "C1A C 0.1 0.1 0.1 ?")))
    assert model.occupancy_fallback is True
    assert [group.weight for group in model.assemblies[0].groups] == [0.5, 0.5]


def test_singleton_partial_group_is_not_promoted_to_a_configuration():
    text = _CIF.replace("C1B C 0.2 0.1 0.1 0.4 0 A 2\n", "").replace(
        "N1A N 0.3 0.3 0.3 0.5 0 B 1\nN1B N 0.4 0.3 0.3 0.5 0 B 2\n", "")
    model = io_cif.disorder_model(_block(text))
    assert not model.assemblies
    assert model.has_ungrouped_partial is True


def test_unknown_disorder_policy_is_rejected(tmp_path):
    path = tmp_path / "disorder.cif"
    path.write_text(_CIF)
    with pytest.raises(ValueError, match="unknown disorder policy"):
        io_cif.read(str(path), disorder="majority")
