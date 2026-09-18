"""Fixture directories.

The CSD-derived CIFs are not redistributed with this repository; `tests/data/README.md` lists the
refcodes and how to fetch them. Tests that need a directory skip when it is absent.
"""
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"


def _require(directory: Path, what: str) -> Path:
    if not directory.is_dir() or not next(directory.rglob("*.cif"), None):
        pytest.skip(f"{what} not present under {directory} (see tests/data/README.md)")
    return directory


@pytest.fixture
def exp():
    """Experimental small-molecule CIFs (fractional coordinates + symmetry); CSD-derived."""
    return _require(DATA / "experimental", "experimental CIF fixtures")


@pytest.fixture
def pred():
    """Predicted mmCIF clusters (Cartesian coordinates, explicit bonds, no cell)."""
    return _require(DATA / "predicted", "predicted-cluster fixtures")


@pytest.fixture
def csd_families():
    """CSD refcode families (redeterminations / polymorphs) used by the COMPACK benchmark."""
    return _require(DATA / "csd_families", "CSD refcode-family fixtures")
