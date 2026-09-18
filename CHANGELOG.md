# Changelog

## 0.1.0

First public release.

- RMSD_n engine: deterministic COMPACK-style packing comparison for experimental CIFs, predicted mmCIF
  clusters, and in-memory structures (`from_arrays`, `from_ase`, `from_pymatgen`).
- CLI verbs `compare`, `matrix`, `dedup`, `score`, `overlay`; parallel batch API with balanced dispatch.
- Interactive HTML overlay viewer and PDB export.
- Benchmarks against CCDC COMPACK bundled under `benchmark/`, figures under `docs/figures/`.
