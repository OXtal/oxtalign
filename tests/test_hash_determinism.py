from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HASH_SEED_PROBE = r"""
import json

import numpy as np

from oxtalign.matching import compare_structures
from oxtalign.model import Molecule, Structure
from oxtalign.overlay import _pick_central


def molecule(species_key, offset):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1]]) + np.asarray(offset)
    return Molecule(("C", "N"), coords, ("A", "B"), ((0, 1),), species_key)


a = Structure(
    [molecule("zeta", (0.0, 0.0, 0.0)), molecule("alpha", (4.0, 1.0, 0.0))],
    None,
    "a",
)
b = Structure(
    [molecule("zeta", (2.0, 3.0, 1.0)), molecule("alpha", (6.0, 4.0, 1.0))],
    None,
    "b",
)
result = compare_structures(a, b, n=2)
print(json.dumps({
    "central_a": result.central_a,
    "n_matched": result.n_matched,
    "overlay_central": _pick_central(a),
    "rmsd_n": result.rmsd_n,
    "status": result.status,
}, sort_keys=True))
"""


def _probe_hash_seed(seed: int) -> dict:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", _HASH_SEED_PROBE],
        cwd=_REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_equal_size_primary_species_is_hash_seed_deterministic():
    results = [_probe_hash_seed(seed) for seed in (0, 1, 2, 3, 4, 5, 17, 42)]

    assert results == [results[0]] * len(results)
    assert results[0]["central_a"] == 1
    assert results[0]["overlay_central"] == 1
    assert results[0]["status"] == "full"
    assert results[0]["n_matched"] == 2
    assert results[0]["rmsd_n"] == pytest.approx(0.0, abs=1e-12)
