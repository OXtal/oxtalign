# How oxtalign computes RMSD_n

A concise account of the algorithm, where it improves on COMPACK, and where it falls short.

## What we compute

The packing-similarity metric **RMSD_n** is the root-mean-square deviation between the atoms of the
*n* closest-contact whole molecules around a reference molecule in crystal A and the corresponding molecules
in crystal B, after the best rigid overlay. Formally, with A's shell fixed as molecules
`{a₁…aₙ}`, we minimise over a rigid motion `g` (rotation + translation, optionally a mirror), a
molecule pairing `σ` (A-molecule → B-molecule), and per-molecule atom correspondences `πₖ`:

```
RMSD_n = min_{g, σ, π}  sqrt( mean_k mean_i | a_k[i] − g(b_{σ(k)}[π_k(i)]) |² )
```

It is a single number in Ångström with a direct physical reading: ≲0.1 Å over n=15 means "same
packing"; growing RMSD / falling `n_matched` means the motifs diverge. This targets the metric COMPACK
popularised through an independent implementation.

The objective couples one **continuous** problem (the rigid motion) with two **discrete** ones
(which molecule maps to which, and which atom maps to which). The design below solves each part with
a method that is *exact given the others*, then alternates to a fixed point and reseeds to avoid
local optima.

## Pipeline

**1. Parse → atoms.** `gemmi` reads both small-molecule CIFs (fractional coords + symmetry operators,
e.s.d.'s stripped) and mmCIF/predicted clusters (Cartesian, explicit bonds). Symmetry is expanded and
**deduplicated by wrapped fractional position**, so an asymmetric-unit input and an already-expanded
full-cell input (e.g. HOLSUX, listed 4× under P2₁/c) yield the *same* atoms — no 4× blow-up.

**2. Atoms → molecules.** Bonds are perceived by a covalent-radius cutoff `d ≤ r_cov(i)+r_cov(j)+0.40 Å`
(Cordero radii, metals included) or taken from explicit `_chem_comp_bond`. Connected components are
whole molecules, unwrapped across cell boundaries. Hydrogens are dropped by default (heavy-atom RMSD,
the COMPACK convention). Molecular species are keyed by
element formula (robust to bond-perception noise in flexible molecules); minor-occupancy disorder
components (occupancy < 0.5) are dropped by default, or a joint disorder configuration can be selected
explicitly through `io_cif.read(..., disorder="dominant")` / `selected_labels=`.

**3. Build A's shell.** Anchor on A's primary (largest) species; order its molecules from the cell
centre outward; dedup conformers. Tile lattice images out to a radius sized from the cell's *narrowest*
perpendicular width and the maximum molecular radius (so thin cells and elongated molecules are tiled
enough), and take the *n* images with the closest interatomic contact to the central molecule. **B is
never given an independent top-n**: B's shell is whatever the overlay maps onto A's.

**4. Match (the core).** For each candidate central pairing and each chirality, collect candidate
overlays from the seed sources below, refine them (trimmed ICP) in order of a cheap packing score, and
keep the result with the most matched molecules, then lowest RMSD.

Outputs: `rmsd_n`, `n_matched`, per-molecule RMSD, the 4×4 transform, radius of gyration of each
shell, `rmsd_1` (central-molecule conformer/lattice check), the chirality relationship (`inverted`),
and `status ∈ {full, partial, failed}`.

## The matching core

**Atom correspondence (discrete, exact).** Within a molecule pair, atoms are grouped by a local
*signature* — `(element, sorted tuple of bonded elements)` — and matched group-by-group with the
Hungarian algorithm (optimal min-cost assignment) on the distance matrix. Atoms whose signature
counts disagree (flexible-bond noise) fall back to element-only matching; unequal counts leave a
rectangular assignment with the surplus unmatched. RMSD is always taken over matched atoms only.

**Seeding (escape local optima).** Three sources of central-molecule overlays, each tried for both
chiralities (`allow_inversion`, via coordinate inversion, not an improper rotation):

- *Permutations.* For small or symmetric molecules, every signature-consistent atom permutation
  (e.g. the 3! orderings of a phosphonate's oxygens), each Kabsch-fitted.
- *Principal axes.* For larger molecules, the 8 sign flips of the principal frame, each refined by a
  per-molecule ICP. All distinct converged overlays are kept, not just the best: a 2-fold-symmetric
  arene has several near-equal central fits and only some bootstrap the packing.
- *Near-isometries.* A molecule with a C3 or higher axis (ferrocene, cubane, coronene,
  hexamethylenetetramine, cages) defeats both: its permutation count is past the cap and its inertia
  tensor is degenerate, so the sign flips reach only a few of its orientations and the packing is
  missed. Any correspondence between two determinations of the same rigid molecule is a near-isometry,
  and an isometry is fixed by four points, so we take four well-spread anchor atoms of A, enumerate the
  same-element quadruples of B whose six mutual distances agree (within 0.3 Å; 0.6 Å only if that
  yields fewer than two maps), and for each quadruple, proper and improper, predict every other atom's
  image as the nearest same-element atom under the implied rigid motion, keeping the bijective maps
  with every atom within 1.2 Å. One Kabsch fit per map is an exact overlay to seed from. A molecule so
  regular that more than 512 quadruples qualify gets no near-isometry seeds and relies on the two
  sources above.

**Refinement order.** Permutation and axis seeds are always refined. Near-isometry seeds are scored
by how many of A's shell centroids land within 2 Å of a same-species B image under the seed pose,
refined in descending score, and skipped when that score is below the best `n_matched` so far or,
once every molecule is matched, below the score that achieved it (a tie can only lower the RMSD). The
skip is a measured shortcut, not a proven bound. On 6,082 CSD pairs it cost no matched molecules
and 7 slightly higher RMSDs (by at most 0.05 Å), while avoiding ~3% of all refinement at 0.4% of the
comparison's cost.

**Trimmed ICP (alternate to convergence).** From each seed we alternate: (a) Hungarian assignment of
A-shell molecules to same-species B images by centroid cost, then per-molecule atom correspondence;
(b) closed-form SVD **Kabsch** fit over the *inlier* molecules only, with a proper-rotation reflection
correction. Iterate until ΔRMSD < 1e-5. Trimming to inliers gives a principled `n_matched` and stops
one edge molecule from inflating the score.

The inlier tolerance follows `refine_schedule`: one iteration per entry (default `(1.5,)`, a single
iteration at 1.5 Å), then `mol_rmsd_tol` for the remaining iterations, and the reported assignment is
always recomputed at `mol_rmsd_tol` under the final pose. Without the wider first step a seed whose
central molecule is exact but whose neighbours sit 1–2 Å off has a single inlier, and a Kabsch fit
over one molecule reproduces its own pose, so the iteration cannot move. The schedule changes only the
search path, never the definition of a match: a full match keeps its RMSD (its inlier set is the
whole shell, so the iteration ends at the same fixed point), while a partial match may settle in a
neighbouring optimum. On 6,082 CSD pairs 407 results improved (more molecules matched, or a lower
RMSD), 9 partial matches worsened slightly, and no full match was lost. `refine_schedule=()`
restores the single-tolerance iteration.

Every core step is closed-form (Kabsch) or polynomial-and-optimal (Hungarian); enumeration is sorted and
tie-breaks are fixed, so the result is independent of `PYTHONHASHSEED`, process count, and wall clock.
`batch.distance_matrix` adds an optional per-pair wall-clock guard for defensive use only.

## How this improves on COMPACK (algorithmically)

COMPACK matches two molecular clusters by comparing a graph of *intermolecular distances and angles*
within fixed tolerances (default ~20% / 20°), via backtracking search under a wall-clock timeout.

1. **No packing-geometry tolerances.** Our *only* tolerance is the covalent-radius bond cutoff —
   chemistry, not packing. This removes COMPACK's central fragility, where RMSD_n shifts with the
   distance/angle tolerances and can even return identical numbers for every shell size.
2. **Determinism.** Closed-form Kabsch + optimal Hungarian + numerical convergence replace
   backtracking-with-timeout, so results are exactly reproducible — not subject to search order or a
   clock.
3. **Globally optimal sub-steps.** Kabsch is the exact least-squares rotation; Hungarian is the exact
   minimum-cost matching. COMPACK's cluster matching is a heuristic graph search.
4. **Multi-seed robustness.** Trying every distinct central overlay (both chiralities, all symmetry
   orderings, and the exact near-isometric maps of a symmetric molecule) and ranking by *packing*
   match defeats the local-optimum trap on symmetric / near-degenerate molecules that single-seed
   greedy matching falls into.
5. **Explicit species identity + geometric correspondence.** Molecules are gated by species (element
   formula) before any atom correspondence is resolved geometrically, so unlike molecules never enter
   the expensive packing search.
6. **A-anchored shell + trimmed inliers.** Defining B's shell as the overlay image of A's (never an
   independent top-n) compares the same physical molecules both ways and yields a stable `n_matched`.
7. **Free and fast.** No CCDC licence; near-linear in *n*, process-parallel across pairs.

See the README for the measured agreement with CCDC COMPACK on the bundled benchmark sets
(`benchmark/`).

## What we fall short of

Honest limits, separated into *fundamental to the metric* vs *fixable in our framework*.

**Fundamental to RMSD_n (shared with COMPACK).**
- **Finite-shell incompleteness.** A fixed *n*-molecule comparison is not a complete invariant: one
  can construct non-isometric crystals that agree on any bounded shell (Anosova–Widdowson–Kurlin,
  arXiv:2205.15298). RMSD_n is a tunable, interpretable *similarity*, not a universal isometry
  classifier — by design — but the limit is real.
- **Not globally continuous.** RMSD_n is only piecewise-continuous: at near-ties the *n*-shell
  membership or an atom correspondence can switch, producing a small but nonzero jump. Fine for
  thresholded near-duplicate detection; a provably Lipschitz invariant (e.g. PDD/AMD) is better for
  smooth ranking and clustering.

**Fixable / not yet done.**
- **No tie-aware shell.** We take exactly the *n* nearest with a stable tie-break and do not currently
  flag a borderline cutoff. A "include all within ε of the n-th distance, keep the best subset" rule
  (and a `boundary_degenerate` flag, or a distance-weighted companion RMSD) would remove the residual
  discontinuity cheaply.
- **Disorder handling is a filter, not a search.** The default loader keeps major-occupancy sites
  (>= 0.5) and drops the rest; `io_cif` can parse `_atom_site_disorder_assembly/group` into joint
  configurations, but `compare` does not yet search over configurations on both sides.
- **Heuristic partial gate.** `allow_partial` accepts a close-formula species within ≤15% heavy-atom
  difference and overlays the shared scaffold — useful for salts / disorder drift, but a threshold, not
  a principled maximum-common-substructure match.
- **Ground truth is COMPACK.** Our accuracy gate is agreement with COMPACK on a finite set; we have no
  independent oracle beyond that and the geometric invariance/golden-anchor tests.

