# COMPACK cross-validation

Reference numbers come from CCDC's `PackingSimilarity` (COMPACK) and require a CSD licence; the
comparison against them does not.

| File | What |
|---|---|
| `fetch_and_compack.py` | (CCDC env) builds the whole reference: the curated families plus random CSD ones, exactly 2,000 pairs -> `tests/data/csd_families/` + `results/compack_reference.csv`. Skips polymeric entries (CSD's `is_polymeric` misses catena structures, so the `)n` formula marker is used) and any family that does not assemble into whole molecules |
| `compare_to_compack.py` | (this env) runs oxtalign on the same pairs, prints the side-by-side table, RMSD_15 correlation on mutually matched pairs, and same/different-form agreement |
| `disagreements.py` | (this env) lists every disagreeing pair and buckets the cause |
| `results/compack_reference.csv` | the single reference set: 2,000 pairs, 163 curated + 1,837 random, `source` column distinguishes them |
| `compack_bench.py` | (CCDC env) per-pair timing, parallel scaling and shell-size sweep of COMPACK -> `results/compack_*.csv` |
| `oxtalign_bench.py` | (this env) the same three runs for oxtalign -> `results/oxtalign_*.csv`; `select-sweep-pairs` picks the sweep subset (`--count`; the bundled results use 200 pairs) |
| `deformation_sweep.py` | (this env) rigid per-molecule perturbations of ten crystals -> `results/deformation_sweep.csv` |
| `csp_demo.py` | (this env) 30 predicted CAPRYL clusters all-against-all -> `results/csp_demo_*.csv` |
| `plot_figures.py` | renders `docs/figures/*.png` from `results/` (needs the `viz` extra) |

COMPACK settings mirror the usual CSP-evaluation driver: shell size 15, distance/angle tolerance
50 % / 75 deg (20 % / 20 deg for size 1; the API's distance tolerance is a fraction), hydrogens and bond counts/types ignored, artificial
inversion allowed, 15-20 s timeout. "Matched" means >= 8 of 15 molecules (COMPACK's match fraction 0.5).

```bash
uv run python benchmark/compare_to_compack.py                 # all 2,000 pairs
SOURCE=curated uv run python benchmark/compare_to_compack.py   # or SOURCE=random
ALLOW_PARTIAL=1 uv run python benchmark/compare_to_compack.py
uv run python benchmark/disagreements.py benchmark/results/compack_reference.csv
```

CIF paths inside the CSVs are repository-relative. The CIFs themselves are CSD entries and are not
shipped with the repository; `tests/data/csd_manifest.csv` lists their refcodes.

```bash
# regenerate the README figures' data (the COMPACK half needs the CCDC env)
micromamba run -n ccdc python benchmark/fetch_and_compack.py --families 330 --jobs 32   # rebuild the reference
uv run python benchmark/oxtalign_bench.py select-sweep-pairs
uv run python benchmark/oxtalign_bench.py timing && uv run python benchmark/oxtalign_bench.py scaling --workers 1,2,4,8,16,32,64 --repeat 3
uv run python benchmark/oxtalign_bench.py sweep --workers 32
micromamba run -n ccdc python benchmark/compack_bench.py timing
micromamba run -n ccdc python benchmark/compack_bench.py scaling --workers 2,4,8,16,32,64
micromamba run -n ccdc python benchmark/compack_bench.py sweep --workers 32
uv run python benchmark/deformation_sweep.py && uv run python benchmark/csp_demo.py
uv run --extra viz python benchmark/plot_figures.py
```
