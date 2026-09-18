# OXtalign

[![CI](https://github.com/OXtal/oxtalign/actions/workflows/ci.yml/badge.svg)](https://github.com/OXtal/oxtalign/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Fast, open RMSD_n packing comparisons for crystals.**

![OXtalign compares 2,000 crystal pairs in 21.3 s on one core; COMPACK reaches 84 in the same time](https://raw.githubusercontent.com/OXtal/oxtalign/main/docs/figures/overlay.gif)

## What it does

RMSD_n is the standard similarity metric in crystal structure prediction (CSP), as implemented in CCDC's
COMPACK (Chisholm & Motherwell, *J. Appl. Cryst.* **38**, 228–231, 2005): take a molecule in crystal A with
its *n* nearest neighbours, and find the rigid motion that best superimposes that cluster onto crystal B.
The comparison yields two numbers: **how many of the *n* molecules land within 1 Å of a counterpart**, and
**RMSD_n**, the root-mean-square atomic deviation over the molecules that do.

OXtalign is a fast, open Python implementation (gemmi + numpy + scipy) from the team behind
[OXtal](https://github.com/OXtal/OXtal). Why use it:

- 🎯 **Same accuracy as COMPACK.** On 2,000 CSD comparisons the RMSD₁₅ values agree to 0.004 Å. Match decisions
  agree on 99% of curated polymorph comparisons and 88% of random CSD comparisons, and most of the rest are COMPACK timeouts.
- ⚡ **Fast.** Per comparison on one core: 7.8 ms median, 10.7 ms mean, against COMPACK's 69 ms and 2.0 s. This is 9×
  on a typical comparison, and ~200× over a batch (due to COMPACK timeouts). 4,700 comparisons per second on 64 cores,
  and cost grows roughly linearly with *n*.
- 🔁 **Deterministic.** No distance or angle tolerances, no timeouts, no random seeds. The same inputs give
  the same numbers on any machine or worker count.
- 📦 **Any input.** Experimental CIFs (symmetry, disorder, Z′ > 1, solvates, salts, organometallics), predicted
  clusters without a unit cell, PDB and XYZ files, in-memory arrays, ASE and pymatgen objects. No
  preprocessing needed.
- 🧩 **Easy to use.** `pip install oxtalign`, then a function call. Apache-2.0 license.

## Install

```bash
pip install oxtalign                                  # latest release (PyPI)
pip install git+https://github.com/OXtal/oxtalign     # latest development version (GitHub main)
```

Needs Python ≥ 3.10; the only runtime dependencies are gemmi, numpy and scipy. To work on the code itself,
clone the repo and run `uv sync --extra dev`.

## Quick start

```bash
oxtalign compare formI.cif formII.cif              # RMSD_15 = 0.0925 Å   matched 15/15   status: full
oxtalign compare formI.cif formII.cif --profile    # coverage at tol = 1, 2, 3 Å: how much packing is shared
oxtalign overlay formI.cif formII.cif -o overlay.html --pdb overlay.pdb
```

```python
from oxtalign import compare

res = compare("formI.cif", "formII.cif", n=15)
res.rmsd_n, res.n_matched, res.status      # 0.0925, 15, "full"   (status: full | partial | failed)
res.rmsd_1, res.inverted, res.transform    # conformer RMSD, mirror-image flag, 4x4 overlay B -> A
```

`status` is `full` when all *n* molecules overlay within the per-molecule tolerance (default 1 Å), `partial`
when at least half do, and `failed` otherwise. This is the match definition used in the OXtal paper. The two
structures must be the same compound unless `--allow-partial` is set.

## For crystallographers

**Example: are two determinations the same polymorph?** `compare` gives the numbers COMPACK would: 15 of 15
molecules within ~0.1–0.3 Å means the same packing.

**Visualization.** `oxtalign overlay` writes a self-contained interactive HTML page: drag to rotate, scroll
to zoom, toggle A / B / matched-only, with matched molecules in green and unit cells drawn. Add `--pdb` to
also get the overlay as a two-chain PDB for Mercury, ChimeraX or VESTA (chain A = reference shell, chain B =
overlaid, B-factor = each molecule's RMSD).

**Details.**

- Hydrogens are dropped; `--include-hydrogens` keeps them.
- Disorder sites with occupancy below 0.5 are dropped. `oxtalign.io_cif.read(..., disorder="dominant")`
  selects the dominant configuration from `_atom_site_disorder_group` instead.
- Already-expanded P1 cells, Z′ > 1, co-crystals, salts and bare molecular clusters work out of the box.
- `--allow-partial` tolerates two determinations that perceive slightly different heavy-atom counts
  (disorder drift), or a shared cation with a different counter-ion.
- Polymeric systems are out of scope.
- For a centrosymmetric packing the direct and inverted fits are numerically the same solution, so the
  reported `inverted` flag can fall either way on rounding; `rmsd_n` and the other numbers are unaffected.

## For (ML) CSP pipelines

```bash
oxtalign score preds/ --truth truths/ --jobs 32 --csv scores.csv      # which predictions hit an experimental form
oxtalign dedup landscape/*.cif --threshold 0.25 --jobs 32 --csv clusters.csv   # distinct packings in a landscape
oxtalign matrix *.cif --jobs 32 --csv matrix.csv                        # all-pairs RMSD_15
```

`score` maps predictions to truths by refcode prefix (`CAPRYL_seed3.cif` ↔ `CAPRYL.cif`, `CAPRYL01.cif`, …)
and reports the usual CSP-evaluation columns (passed = at least half the shell matched and no steric clash,
nmatched/RMSD at 1 and *n*, best truth). `dedup` groups structures whose full shells overlay below a
threshold and names a representative per group.

```python
from oxtalign import compare, from_arrays, from_ase, from_pdb, from_pymatgen, from_xyz
from oxtalign.batch import distance_matrix, score_predictions

s = from_arrays(elements, frac_coords, cell=(a, b, c, alpha, beta, gamma))   # or a 3x3 lattice; symops=...
c = from_arrays(elements, xyz, bonds=bonds)                                  # a cluster without a cell
p = from_pdb("form.pdb")                        # CRYST1 expands the asymmetric unit; no CRYST1 -> cluster
x = from_xyz("form.xyz")                        # extended-XYZ Lattice="..." if present, else a cluster
compare(s, from_ase(atoms), n=15)                                            # ASE / pymatgen: no import needed

M, results, structures = distance_matrix(paths, n=15, n_jobs=32)            # inf where no match
rows = score_predictions("preds/", {"CAPRYL": ["CAPRYL.cif"]}, n_jobs=32)
```

Molecular clusters without a cell (Cartesian mmCIF with `_chem_comp_bond`, as OXtal writes them) are
compared exactly like periodic cells. Every comparison is independent, so `n_jobs` scales across processes;
the largest pairs go first so a slow pair never strands a worker. On a dedicated node, `pin_cores=True` /
`--pin-cores` binds one worker per physical core.

## Options

| Python | CLI | default | meaning |
|---|---|---|---|
| `n` | `--n` | 15 | molecules in the coordination shell |
| `mol_rmsd_tol` | `--tol` | 1.0 Å | a molecule counts as matched below this overlay RMSD |
| `allow_inversion` | `--no-inversion` | on | also try the mirror image; `inverted` reports it |
| `include_hydrogens` | `--include-hydrogens` | off | heavy atoms only by default |
| `allow_partial` | `--allow-partial` | off | tolerate small atom-count / counter-ion differences |
| `refine_schedule` | — | `(1.5,)` | larger values widen the packing search a bit; `(2.0,)` may find more matches |

## Benchmarks

All numbers come from one machine (AMD EPYC 9654) and the scripts in `benchmark/`. COMPACK was run as CCDC's
`PackingSimilarity` with the usual CSP-evaluation settings: shell 15, 50% distance and 75° angle tolerances,
hydrogens and bond counts ignored, inversion allowed, 20 s timeout. Its outputs are bundled as CSVs under
`benchmark/results/`, so the figures regenerate without a CSD licence:
`uv run --extra viz python benchmark/plot_figures.py`.

COMPACK is proprietary CCDC software, run here under our own CSD licence, and OXtalign is an independent
implementation that is not affiliated with or endorsed by CCDC.

![Agreement with COMPACK](https://raw.githubusercontent.com/OXtal/oxtalign/main/docs/figures/agreement_with_compack.png)

**Agreement with COMPACK.** 2,000 polymorph comparisons: 163 from seven curated polymorph families (ROY,
galunisertib, ritonavir, …) and 1,837 from random CSD refcode families. Where both engines match all 15
molecules, RMSD₁₅ is the same number: 1,243 comparisons, mean |Δ| 0.004 Å. Where they disagree it is often a
COMPACK timeout, which hit 118 of the 1,837 random comparisons and 9 of the 163 curated ones. Disagreements are not
always our errors: COMPACK's search has tolerances and a timeout, so a disagreement can be a COMPACK miss as
well as ours (OXtalign matches 120 comparisons that COMPACK misses).

![Throughput vs cores](https://raw.githubusercontent.com/OXtal/oxtalign/main/docs/figures/throughput_vs_cores.png)

**Throughput.** On one core the median comparison takes 7.8 ms against COMPACK's 69 ms. The means are 10.7 ms and
2.0 s, because COMPACK has a tail of 20 s timeouts. Across processes OXtalign reaches 4,689 comparisons/s on 64 workers
and 7,126 on 128. COMPACK parallelises too, but from a much higher per-comparison cost: 24.6 comparisons/s on 64 workers, 190×
slower.

![Shell-size scaling](https://raw.githubusercontent.com/OXtal/oxtalign/main/docs/figures/shell_size_scaling.png)

**Shell size.** Up to n ≈ 50 the median cost is linear (fixed seeding cost + a per-molecule cost),
4.1 ms + 0.31 ms × n (R² 0.99, 200 comparisons). Beyond that it turns superlinear, since both the shell and the
candidate images grow with n (75 to 150 costs 2.6× rather than 2×). COMPACK is faster for a single molecule.
From three molecules on OXtalign pulls ahead, by 9× at n = 15 (62× on the mean) and ~400× at n = 30, as
COMPACK's distance-graph search grows combinatorially and starts timing out (19 of 200 comparisons at n = 30).
OXtalign and COMPACK agree on 84–92% of comparisons across n.

![Predicted clusters and perturbations](https://raw.githubusercontent.com/OXtal/oxtalign/main/docs/figures/beyond_experimental_cifs.png)

**Beyond experimental CIFs.** Left: 30 predicted CAPRYL clusters (no unit cell) compared all-against-all;
near-identical clusters show up as diagonal blocks. Middle and right: crystals with every molecule perturbed by a set
amount. RMSD₁₅ tracks the perturbation while it is small; past ~1 Å the packing is no longer recognised as
the same one, as expected.

## How it works

Both structures are assembled into whole molecules: symmetry is expanded, duplicates removed, and bonds
perceived from covalent radii or taken from the file. A's reference molecule and its *n* closest-contact
neighbours form the shell. For every candidate central pairing, and both chiralities, the two central
molecules are overlaid from their principal axes and from every near-isometric atom map between them, so a
symmetric molecule such as ferrocene is covered. Each overlay is then grown by trimmed ICP with a briefly
widened inlier tolerance: molecules are assigned by the Hungarian algorithm, atoms within each molecule by a
signature-grouped Hungarian assignment, and the rigid motion is the closed-form Kabsch fit over the inlier
molecules. The overlay with the most matched molecules, then the lowest RMSD, wins. Every step is closed-form
or an exact assignment, so the result is deterministic. [docs/METHOD.md](docs/METHOD.md) has the details and
the honest limits (RMSD_n is a bounded-shell similarity, not a complete invariant); that document was drafted
almost entirely by an LLM.

## Layout

```
src/oxtalign/    io_cif · io_arrays · assemble · cluster · matching · compare · batch · overlay · cli
tests/           pytest suite; CSD fixtures are listed by refcode, not shipped (tests/data/README.md)
benchmark/       COMPACK reference results and the scripts behind the figures
docs/            METHOD.md, the figures above, and the script that renders the overlay GIF
```

## License and citation

Apache License 2.0 (see `LICENSE`). If you use OXtalign, please cite OXtal and this software:

```bibtex
@inproceedings{jin2026oxtal,
  title     = {{OXtal}: An All-Atom Diffusion Model for Organic Crystal Structure Prediction},
  author    = {Jin, Emily and Nica, Andrei Cristian and Galkin, Mikhail and Rector-Brooks, Jarrid and
               Lee, Kin Long Kelvin and Miret, Santiago and Arnold, Frances H. and Bronstein, Michael and
               Bose, Avishek Joey and Tong, Alexander and Liu, Cheng-Hao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2512.06987}
}

@software{oxtalign2026,
  title   = {{OXtalign}: fast, open RMSD$_n$ packing similarity for molecular crystals},
  author  = {Liu, Cheng-Hao},
  year    = {2026},
  url     = {https://github.com/OXtal/oxtalign},
  version = {0.1.0}
}
```
