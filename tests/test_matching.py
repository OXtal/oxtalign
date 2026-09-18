"""Matching correctness: invariances, determinism, golden same-form anchors, polymorph separation."""
import numpy as np
import pytest

from oxtalign import compare
from oxtalign.cluster import image_radius, nearest_shell
from oxtalign.compare import load
from oxtalign.model import Molecule, Structure, UnitCell


def _transform_cluster(s: Structure, R, t, mirror=False) -> Structure:
    sign = -1.0 if mirror else 1.0
    mols = [Molecule(m.elements, (sign * m.coords) @ R.T + t, m.labels, m.bonds, m.species_key)
            for m in s.molecules]
    return Structure(mols, None, s.source + "_T")


def _rand_rot(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_identity_is_zero(exp):
    for rel in ["CAPRYL.cif", "RUSXEM.cif", "XAFQON.cif", "ROY/QAXMEH24.cif", "NOFJEX/NOFJEX.cif"]:
        s = load(str(exp / rel))
        r = compare(s, s, n=15)
        assert r.status == "full" and r.n_matched == 15 and r.rmsd_n < 1e-6, rel


def test_rotation_translation_invariance(pred):
    s = load(str(pred / "CAPRYL_seed0_sample_0.cif"))
    moved = _transform_cluster(s, _rand_rot(7), np.array([12.3, -4.5, 6.7]))
    r = compare(s, moved, n=15)
    assert r.n_matched == 15 and r.rmsd_n < 1e-6


def test_enantiomer_detected_via_inversion(pred):
    s = load(str(pred / "CAPRYL_seed0_sample_0.cif"))
    mirrored = _transform_cluster(s, _rand_rot(3), np.zeros(3), mirror=True)
    r_inv = compare(s, mirrored, n=15, allow_inversion=True)
    assert r_inv.n_matched == 15 and r_inv.rmsd_n < 1e-6 and r_inv.inverted
    r_noinv = compare(s, mirrored, n=15, allow_inversion=False)
    assert r_noinv.rmsd_n > 0.1 or r_noinv.n_matched < 15


def test_determinism(exp):
    a, b = str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif")
    r1, r2 = compare(a, b, n=15), compare(a, b, n=15)
    assert r1.rmsd_n == r2.rmsd_n and r1.n_matched == r2.n_matched


def test_different_compounds_fail(exp):
    r = compare(str(exp / "CAPRYL.cif"), str(exp / "MUZKOL.cif"), n=15)
    assert r.status == "failed" and "not in other structure" in r.message


def test_golden_same_form_anchor(exp):
    """COMPACK calls this pair the same form with RMSD_15 < 0.1 — we must reproduce that."""
    r = compare(str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif"), n=15)
    assert r.status == "full" and r.n_matched == 15 and r.rmsd_n < 0.1


def test_elongated_molecule_uses_compack_contact_shell(csd_families):
    """Centroid-nearest shells miss the long-axis contacts in this elongated molecule.

    The contact-nearest shell reproduces CCDC PackingSimilarity's full-shell RMSD_15 and keeps the
    two distinct packings above the 0.25 A duplicate threshold.
    """
    result = compare(
        str(csd_families / "YUFZOQ" / "YUFZOQ01.cif"),
        str(csd_families / "YUFZOQ" / "YUFZOQ04.cif"),
        n=15,
    )

    assert result.status == "full"
    assert result.n_matched == 15
    assert result.rmsd_n == pytest.approx(0.4407111218991615, abs=1e-10)


def test_symmetric_central_molecule_matches_full_shell(csd_families):
    """Ferrocene's inertia tensor is degenerate, so no principal-axis sign flip reaches the right
    orientation and the packing match is missed outright. Near-isometry seeding must recover it."""
    result = compare(
        str(csd_families / "FEROCE" / "FEROCE28.cif"),
        str(csd_families / "FEROCE" / "FEROCE31.cif"),
        n=15,
    )

    assert result.status == "full"
    assert result.n_matched == 15
    assert result.rmsd_n == pytest.approx(0.219, abs=1e-3)   # COMPACK reports 0.219 A


def test_refine_schedule_leaves_a_full_match_unchanged(exp):
    """The wider early tolerance only widens the basin searched: the assignment is always recomputed
    at ``mol_rmsd_tol``, so a full match keeps its RMSD -- to round-off, not bit for bit."""
    a, b = str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif")
    scheduled, plain = compare(a, b, n=15), compare(a, b, n=15, refine_schedule=())

    assert scheduled.n_matched == plain.n_matched == 15
    assert scheduled.rmsd_n == pytest.approx(plain.rmsd_n, abs=1e-12)


def test_contact_shell_radius_covers_molecule_spanning_narrow_cells():
    """Molecular-radius padding must expose contacts beyond the centroid-only tiling."""
    orth = np.diag([10.0, 10.0, 10.0])
    cell = UnitCell(
        10.0,
        10.0,
        10.0,
        90.0,
        90.0,
        90.0,
        ["x,y,z"],
        orth,
        np.linalg.inv(orth),
    )
    molecule = Molecule(
        ("C", "C"),
        np.asarray([[0.0, 0.0, 0.0], [29.9, 0.0, 0.0]]),
        ("C1", "C2"),
        ((0, 1),),
        "ELONGATED",
    )
    structure = Structure([molecule], cell, "elongated")

    assert image_radius(structure, 2) == 2
    assert image_radius(structure, 2, contact=True) == 5
    shell = nearest_shell(structure, 0, 2)

    assert shell[0].centroid == pytest.approx(molecule.centroid)
    assert np.linalg.norm(shell[1].centroid - molecule.centroid) == pytest.approx(30.0)


def test_distinct_polymorphs_separate(exp):
    """Genuinely different ROY polymorphs must not collapse to a full 15/15 match."""
    r = compare(str(exp / "ROY" / "QAXMEH24.cif"), str(exp / "ROY" / "ROY_R05.cif"), n=15)
    assert r.n_matched < 15


def test_allow_partial_recovers_disorder_drift(csd_families):
    """CEGHED's main molecule is perceived as C21 in some determinations and C24 in others (a
    disordered tert-butyl modelled with vs. without its alternate site). The species gate rejects
    this by default; allow_partial overlays the shared scaffold and recovers COMPACK's tight match."""
    a, b = str(csd_families / "CEGHED" / "CEGHED.cif"), str(csd_families / "CEGHED" / "CEGHED01.cif")
    assert compare(a, b, n=15).status == "failed"                 # formula drift -> gate aborts
    r = compare(a, b, n=15, allow_partial=True)
    assert r.status == "full" and r.n_matched == 15 and r.rmsd_n < 0.1


def test_allow_partial_off_is_unchanged(csd_families):
    """allow_partial must not alter results for a pair whose species already match exactly."""
    a, b = str(csd_families / "CEGHED" / "CEGHED01.cif"), str(csd_families / "CEGHED" / "CEGHED04.cif")
    off, on = compare(a, b, n=15), compare(a, b, n=15, allow_partial=True)
    assert off.n_matched == on.n_matched and off.rmsd_n == on.rmsd_n


def test_allow_partial_still_rejects_different_compounds(exp):
    """The close-formula gate must still refuse two genuinely different compounds with the flag on."""
    r = compare(str(exp / "CAPRYL.cif"), str(exp / "MUZKOL.cif"), n=15, allow_partial=True)
    assert r.status == "failed" and r.n_matched == 0


def _opaque_single_molecule(elements, species_key):
    count = len(elements)
    coords = np.column_stack(
        (
            np.arange(count, dtype=float),
            np.arange(count, dtype=float) ** 2 / max(count, 1),
            np.arange(count, dtype=float) ** 3 / max(count * count, 1),
        )
    )
    molecule = Molecule(
        tuple(elements),
        coords,
        tuple(f"A{index}" for index in range(count)),
        tuple((index, index + 1) for index in range(count - 1)),
        species_key,
    )
    return Structure([molecule], None, species_key)


def test_allow_partial_uses_elements_for_distinct_opaque_species_keys():
    left = _opaque_single_molecule(["C"] * 10, "GLOBAL:CONFORMER:LEFT")
    right = _opaque_single_molecule(["C"] * 10, "GLOBAL:CONFORMER:RIGHT")

    assert compare(left, right, n=1).status == "failed"
    result = compare(left, right, n=1, allow_partial=True)

    assert result.status == "full"
    assert result.n_matched == 1
    assert result.rmsd_n < 1e-8


def test_allow_partial_rejects_incompatible_elements_for_opaque_species_keys():
    left = _opaque_single_molecule(["C"] * 10, "GLOBAL:CONFORMER:LEFT")
    right = _opaque_single_molecule(["C"] * 8 + ["N"] * 2, "GLOBAL:CONFORMER:RIGHT")

    result = compare(left, right, n=1, allow_partial=True)

    assert result.status == "failed"
    assert result.n_matched == 0
    assert "not in other structure" in result.message


def test_match_profile_monotone(exp):
    """The coverage curve is non-decreasing in tolerance and reveals the partial motif at larger tol."""
    from oxtalign import match_profile
    rows = match_profile(str(exp / "galunisertib" / "Galunisertib_FORM_IX.cif"),
                         str(exp / "galunisertib" / "Galunisertib_FORM_VIII.cif"), tols=(1.0, 2.0, 3.0))
    ns = [r["n_matched"] for r in rows]
    assert ns == sorted(ns) and ns[-1] > ns[0]          # tighter core grows into the partial overlay


def test_match_profile_same_form_full(exp):
    """A genuine same-form pair is fully matched at every tolerance."""
    from oxtalign import match_profile
    rows = match_profile(str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif"))
    assert all(r["n_matched"] == 15 for r in rows)
