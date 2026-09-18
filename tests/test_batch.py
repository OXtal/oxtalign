"""Batch-mode correctness: parallel must equal serial; matrix symmetric; clash detection sane."""
import numpy as np

from oxtalign.batch import distance_matrix, intermolecular_clash
from oxtalign.compare import load


def _pred_set(pred, k=8):
    return sorted(str(p) for p in pred.glob("CAPRYL_seed*_sample_0.cif"))[:k]


def test_parallel_matches_serial(pred):
    files = _pred_set(pred, 8)
    M1, r1, _ = distance_matrix(files, n=15, n_jobs=1)
    M4, r4, _ = distance_matrix(files, n=15, n_jobs=4)
    assert np.array_equal(M1, M4), "parallel distance matrix must be identical to serial"
    for key in r1:
        assert r1[key].n_matched == r4[key].n_matched
        a, b = r1[key].rmsd_n, r4[key].rmsd_n
        assert a == b or (np.isnan(a) and np.isnan(b))


def test_matrix_symmetric_zero_diagonal(pred):
    M, _, _ = distance_matrix(_pred_set(pred, 6), n=15)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)


def test_clash_detection(pred):
    # A predicted cluster may have mild clashes; a single isolated molecule cannot.
    one = load(str(next(pred.glob("CAPRYL_seed0_sample_0.cif"))))
    one.molecules[:] = one.molecules[:1]
    assert intermolecular_clash(one) == float("-inf")


def test_unassemblable_structure_fails_its_pairs_without_aborting(exp, tmp_path):
    """A garbage input whose atoms bond into an infinite network cannot be assembled; batch callers must
    report it as failed rather than raise (a CSP scoring run should not die on one bad prediction)."""
    import pytest

    from oxtalign import load
    from oxtalign.batch import score_predictions

    bad = tmp_path / "CAPRYL_seed99_sample_0.cif"      # a 2.8 A P1 cell of carbon -> infinite C-C chain
    bad.write_text("data_bad\n_cell_length_a 2.8\n_cell_length_b 10\n_cell_length_c 10\n"
                   "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                   "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_symmetry_equiv_pos_as_xyz\nx,y,z\n"
                   "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
                   "_atom_site_fract_z\nC1 C 0.0 0.5 0.5\nC2 C 0.5 0.5 0.5\n")
    with pytest.raises(ValueError):
        load(str(bad))
    good = [str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif")]   # a matching pair
    M, results, structs = distance_matrix(good + [str(bad)], n=15)
    assert structs[2] is None and np.isinf(M[0, 2]) and np.isinf(M[1, 2]) and np.isfinite(M[0, 1])
    assert results[(0, 2)].status == "failed" and "assembled" in results[(0, 2)].message
    rows = score_predictions(str(tmp_path), {"CAPRYL": good[:1]}, n=15)
    assert rows[0]["status"] == "load_error" and rows[0]["passed"] is False


def test_pinned_parallel_matches_serial(pred):
    files = _pred_set(pred, 8)
    M1, r1, _ = distance_matrix(files, n=15, n_jobs=1)
    M4, r4, _ = distance_matrix(files, n=15, n_jobs=4, pin_cores=True)
    assert np.array_equal(M1, M4)
    assert all(r1[k].n_matched == r4[k].n_matched for k in r1)


def test_score_predictions_parallel_matches_serial(exp, pred, tmp_path):
    import shutil

    from oxtalign.batch import score_predictions

    truth = {"CAPRYL": [str(exp / "CAPRYL.cif")], "HOLSUX": [str(exp / "HOLSUX" / "HOLSUX.cif")]}
    for p in sorted(pred.glob("CAPRYL_seed*_sample_0.cif"))[:4]:
        shutil.copy(p, tmp_path / p.name)
    shutil.copy(exp / "HOLSUX" / "HOLSUX01.cif", tmp_path / "HOLSUX_pred.cif")
    serial = score_predictions(str(tmp_path), truth, n=15)
    parallel = score_predictions(str(tmp_path), truth, n=15, n_jobs=3)
    assert serial == parallel
    by_file = {r["file"]: r for r in serial}
    assert by_file["HOLSUX_pred.cif"]["passed"] is True and by_file["HOLSUX_pred.cif"]["nmatched_15"] == 15
