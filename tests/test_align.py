import numpy as np

from oxtalign.align import kabsch, kabsch_rmsd, radius_of_gyration


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_kabsch_recovers_rotation_translation():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(12, 3))
    R = _random_rotation(rng)
    Q = P @ R.T + np.array([3.0, -2.0, 1.0])
    R_fit, _ = kabsch(P, Q)
    assert np.allclose(R_fit, R, atol=1e-9)
    assert kabsch_rmsd(P, Q) < 1e-9


def test_kabsch_rejects_reflection_but_inversion_recovers():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(10, 3))
    mirrored = P * np.array([1, 1, -1])      # improper image
    assert kabsch_rmsd(P, mirrored) > 0.1    # proper rotation cannot fit a reflection
    assert kabsch_rmsd(-P, mirrored) < 1e-9  # inverting coordinates first does


def test_kabsch_single_atom_is_pure_translation():
    R, t = kabsch(np.array([[1.0, 2, 3]]), np.array([[4.0, 6, 8]]))
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, [3, 4, 5])


def test_radius_of_gyration_known_value():
    pts = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    assert abs(radius_of_gyration(pts) - 1.0) < 1e-12
