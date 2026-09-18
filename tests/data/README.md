# Test fixtures

## CSD-derived CIFs are not included

The 153 experimental CIFs that the tests and the COMPACK benchmark read are Cambridge Structural
Database entries, so they are not redistributed here. `csd_manifest.csv` lists every one of them:
the path it belongs at, its CSD refcode, and its CCDC deposition number. Fetch them yourself from
[CCDC Access Structures](https://www.ccdc.cam.ac.uk/structures/), or with the CSD Python API if you
have a licence, and save each file at the path the manifest gives.

| Directory | Contents |
|---|---|
| `experimental/` | 45 CIFs: single structures (CAPRYL, MUZKOL, OBEQIX, RUSXEM, XAFQON) and polymorph families (HOLSUX, NOFJEX, NOFKAU, ROY/QAXMEH, galunisertib/DORDUM, ritonavir/YIGPIO, FOYNEO) |
| `csd_families/` | 25 refcode families, up to 5 redeterminations or polymorphs each, as sampled by `benchmark/fetch_and_compack.py` |
| `predicted/` | 32 predicted mmCIF clusters (Cartesian coordinates, no cell, explicit `_chem_comp_bond`), our own model outputs, committed here |

The files under `csd_families/` were written by the CSD Python API, so a fetch reproduces them as
they were. Those under `experimental/` came from Access Structures and from published supplementary
material, so a CSD copy can differ in detail from the one used to record the reference numbers in
`benchmark/results/`.

## Without the CSD files

`pytest` skips the tests that need a missing fixture directory and runs the rest: the
synthetic-geometry, in-memory-input, predicted-cluster, CLI and determinism tests all pass on a
checkout with no CSD data. That is what CI runs.
