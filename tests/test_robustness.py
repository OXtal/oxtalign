"""Robustness tests: the metric must degrade smoothly under noise and reject scrambled packings."""
import numpy as np

from oxtalign import compare
from oxtalign.compare import load
from oxtalign.model import Molecule, Structure


def _rand_rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _jitter(s: Structure, sigma: float, seed: int) -> Structure:
    """Add isotropic Gaussian noise to every atom (intramolecular distortion)."""
    rng = np.random.default_rng(seed)
    mols = [Molecule(m.elements, m.coords + rng.normal(scale=sigma, size=m.coords.shape),
                     m.labels, m.bonds, m.species_key) for m in s.molecules]
    return Structure(mols, None, s.source + "_jit")


def _scramble(s: Structure, seed: int, spread: float = 12.0) -> Structure:
    """Keep molecules intact but give each a random rigid pose — destroys the packing."""
    rng = np.random.default_rng(seed)
    mols = []
    for m in s.molecules:
        c = m.centroid
        moved = (m.coords - c) @ _rand_rot(rng).T + rng.uniform(-spread, spread, size=3)
        mols.append(Molecule(m.elements, moved, m.labels, m.bonds, m.species_key))
    return Structure(mols, None, s.source + "_scr")


def test_atom_jitter_is_continuous(pred):
    """Small atomic noise -> still matches, RMSD grows smoothly with the noise level."""
    s = load(str(pred / "CAPRYL_seed0_sample_0.cif"))
    last = 0.0
    for sigma in (0.05, 0.1, 0.2):
        r = compare(s, _jitter(s, sigma, seed=1), n=15)
        assert r.n_matched == 15, f"jitter sigma={sigma} should still match"
        assert sigma * 0.3 < r.rmsd_n < sigma * 3.0, (sigma, r.rmsd_n)  # tracks noise level
        assert r.rmsd_n >= last - 1e-9                                   # monotone in noise
        last = r.rmsd_n


def test_scrambled_packing_does_not_match(pred):
    """Randomly re-posed (intact) molecules must NOT yield a full/passing packing match."""
    s = load(str(pred / "XAFQON_seed0_sample_0.cif"))
    r = compare(s, _scramble(s, seed=2), n=15)
    assert r.status != "full" and r.n_matched < 8


def test_determinism_under_repeat(pred):
    s = load(str(pred / "RUSXEM_seed0_sample_0.cif"))
    scr = _scramble(s, seed=4)
    a = compare(s, scr, n=15)
    b = compare(s, scr, n=15)
    assert a.n_matched == b.n_matched and (a.rmsd_n == b.rmsd_n or
                                           (np.isnan(a.rmsd_n) and np.isnan(b.rmsd_n)))
