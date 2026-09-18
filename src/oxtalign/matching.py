"""The matching core: RMSD_n by progressive coordinate superposition (PAC-style) + trimmed ICP.

Design choices that make RMSD_n stable and robust:

* Exact species identity uses each molecule's persisted ``species_key``. When ``allow_partial`` is
  enabled, fallback pairing is selected from the representative molecules' element-count multisets,
  never by parsing the key; keys may be opaque global conformer identities. Atom correspondence is
  then found geometrically (element-wise Hungarian assignment after alignment).
* Seeding tries every distinct conformer pairing (central of A x central of B), both chiralities,
  several principal-axis orientations, and — when the two central molecules are near-isometric —
  every atom map between them, then ICP-refines each, avoiding the local optima a single greedy
  seed falls into.
* Refinement alternates Hungarian molecule-assignment with SVD-Kabsch over the *inlier* molecules
  (per-molecule RMSD < tol), to numerical convergence (trimmed ICP); the first iterations use a
  wider inlier tolerance so a seed with a single inlier is not frozen. No packing distance/angle
  tolerances or timeouts: same inputs -> identical outputs.

RMSD_n is the RMSD over the atoms of the matched molecules in A's n-molecule shell.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .align import kabsch, radius_of_gyration, rmsd
from .cluster import materialize, molecule_images, nearest_shell_images
from .model import ComparisonResult, Structure

# The core calls cdist ~5x10^5 times per comparison on 2-30 row inputs, where the wrapper spends
# ~0.7 us on validation and dispatch. Bind the kernel it ends up calling, checked equal on import.
try:
    from scipy.spatial import _distance_pybind as _dpb

    _cdist = _dpb.cdist_euclidean
    _probe_a, _probe_b = np.zeros((1, 3)), np.ones((1, 3))
    if not np.array_equal(_cdist(_probe_a, _probe_b), cdist(_probe_a, _probe_b)):
        raise ImportError("cdist kernel disagrees with scipy.spatial.distance.cdist")
except Exception:                                          # pragma: no cover - scipy layout change
    _cdist = cdist

# Proper + improper sign flips of the 3 principal axes (kabsch re-imposes a proper rotation anyway).
_SIGN_SEEDS = [np.diag(d) for d in
               [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]]

# Near-isometry seeding of the central pair (see _isometry_maps / _isometry_seeds).
_ANCHOR_DELTAS = (0.3, 0.6)   # A; anchor pairwise-distance tolerance: tight pass, then loose fallback
_PREDICT_R = 1.2              # A; acceptance radius for a predicted atom image
_SEED_RMSD_CAP = 1.5          # A; a map seeds a chirality branch only if its overlay is this good
_MAX_QUADS = 512              # anchor quadruples per pass (an icosahedral cage needs 240)
_SCREEN_R = 2.0               # A; a shell centroid counts as hit within this distance


def _apply(coords, sign, R, t):
    return (sign * coords) @ R.T + t


def _signatures(mol) -> list[tuple]:
    """Per-atom signature (element, sorted bonded-element tuple).

    Discriminates atom roles (e.g. a quaternary C from a methyl C) for a precise correspondence,
    while being local enough that flexible-molecule bond noise perturbs only a few atoms.
    """
    nbr: list[list[str]] = [[] for _ in range(mol.n_atoms)]
    for a, b in mol.bonds:
        nbr[a].append(mol.elements[b])
        nbr[b].append(mol.elements[a])
    return [(mol.elements[i], tuple(sorted(nbr[i]))) for i in range(mol.n_atoms)]


def _groups(mol) -> dict[tuple, np.ndarray]:
    """signature -> atom-index array, precomputed once per molecule (kept out of the hot loop)."""
    g: dict[tuple, list[int]] = {}
    for i, s in enumerate(_signatures(mol)):
        g.setdefault(s, []).append(i)
    return {s: np.array(idx) for s, idx in g.items()}


@dataclass(frozen=True, slots=True)
class _CorrespondencePlan:
    """Index bookkeeping for the atom correspondence between one A molecule and one B molecule.

    Built once per (A molecule, B molecule) pair and reused across every ICP iteration, seed and
    central pairing that overlays those two molecules -- thousands of times per comparison.
    """

    n_atoms: int
    single_a: np.ndarray    # A atoms whose signature class has exactly one member in both molecules
    single_b: np.ndarray    # ... and the B atom each one maps to
    exact: tuple            # (ai, bi) index arrays of equal-count signature classes with > 1 atom
    fallback: tuple         # (ai, bi) element-only classes for atoms whose signature counts differ
    complete: bool          # every A atom is covered by single/exact classes (perm never holds -1)
    # Derived, for the hot path only (see _correspondence): the single-member classes already
    # scattered into a template permutation, and every multi-atom class' atoms laid out back to
    # back so the two coordinate gathers happen once per call instead of once per class.
    template: np.ndarray    # perm prefilled with the single-member classes, -1 elsewhere
    order_a: np.ndarray     # concatenation of every class' A indices (exact classes first)
    order_b: np.ndarray     # ... and of their B indices
    classes: tuple          # (ai, bi, sa, ea, sb, eb) slices into the gathered coordinates


def _plan(a_groups, b_groups, n_atoms) -> _CorrespondencePlan:
    single_a, single_b, exact = [], [], []
    left_a: dict[str, list[int]] = {}
    for s, ai in a_groups.items():
        bi = b_groups.get(s)
        if bi is not None and len(bi) == len(ai):
            if len(ai) == 1:
                single_a.append(ai[0])
                single_b.append(bi[0])
            else:
                exact.append((ai, bi))
        else:
            left_a.setdefault(s[0], []).extend(ai.tolist())
    fallback = []
    if left_a:                                            # element-only fallback for the remainder
        left_b: dict[str, list[int]] = {}
        for s, bi in b_groups.items():
            ai = a_groups.get(s)
            if ai is None or len(ai) != len(bi):
                left_b.setdefault(s[0], []).extend(bi.tolist())
        for el, ai in left_a.items():
            ai_a, bi_a = np.array(ai), np.array(left_b.get(el, []))
            if len(bi_a) == 0:                            # no B atoms of this element: leave unmatched
                continue
            fallback.append((ai_a, bi_a))
    sa_arr, sb_arr = np.array(single_a, dtype=int), np.array(single_b, dtype=int)
    template = np.full(n_atoms, -1, dtype=int)
    template[sa_arr] = sb_arr
    classes, order_a, order_b, pa, pb = [], [], [], 0, 0
    for ai, bi in (*exact, *fallback):
        classes.append((ai, bi, pa, pa + len(ai), pb, pb + len(bi)))
        order_a.append(ai)
        order_b.append(bi)
        pa += len(ai)
        pb += len(bi)
    return _CorrespondencePlan(
        n_atoms, sa_arr, sb_arr, tuple(exact), tuple(fallback), complete=not left_a,
        template=template,
        order_a=np.concatenate(order_a) if order_a else np.empty(0, dtype=int),
        order_b=np.concatenate(order_b) if order_b else np.empty(0, dtype=int),
        classes=tuple(classes))


class _Plans:
    """Per-comparison cache of signature groupings and correspondence plans, keyed by molecule index."""

    def __init__(self, A: Structure, B: Structure):
        self.A, self.B = A, B
        self.A_groups = [_groups(m) for m in A.molecules]
        self.B_groups = [_groups(m) for m in B.molecules]
        self._n_atoms = [m.n_atoms for m in A.molecules]
        self._cache: dict[tuple[int, int], _CorrespondencePlan] = {}
        self._perms: dict[tuple[int, int], list] = {}
        self._axes: dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray]] = {}
        self._elements: dict[str, int] = {}
        self._labels: dict[tuple[str, int], np.ndarray] = {}
        self._anchors: dict[int, np.ndarray] = {}
        self._maps: dict[tuple[int, int], list] = {}
        self.b_keys = [m.species_key for m in B.molecules]

    def get(self, ia: int, ib: int) -> _CorrespondencePlan:
        plan = self._cache.get((ia, ib))
        if plan is None:
            plan = self._cache[(ia, ib)] = _plan(self.A_groups[ia], self.B_groups[ib], self._n_atoms[ia])
        return plan

    def perms(self, ia: int, ib: int) -> list:
        """Signature-consistent start correspondences for the central pair (see _enumerate_perms)."""
        perms = self._perms.get((ia, ib))
        if perms is None:
            perms = self._perms[(ia, ib)] = _enumerate_perms(self.A_groups[ia], self.B_groups[ib],
                                                             self._n_atoms[ia])
        return perms

    def axes(self, side: str, index: int, invert: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(principal axes, centroid) of one molecule, optionally of its inverted coordinates."""
        key = (side, index, invert)
        axes = self._axes.get(key)
        if axes is None:
            coords = (self.A if side == "A" else self.B).molecules[index].coords
            if invert:
                coords = -coords
            axes = self._axes[key] = (_principal_axes(coords), coords.mean(axis=0))
        return axes

    def labels(self, side: str, index: int) -> np.ndarray:
        """One molecule's per-atom element codes, interned across both structures."""
        key = (side, index)
        lab = self._labels.get(key)
        if lab is None:
            mol = (self.A if side == "A" else self.B).molecules[index]
            lab = self._labels[key] = np.fromiter(
                (self._elements.setdefault(e, len(self._elements)) for e in mol.elements),
                dtype=np.intp, count=mol.n_atoms)
        return lab

    def maps(self, ia: int, ib: int) -> list:
        """Near-isometric overlays of central B[ib] onto central A[ia]; shared by both chiralities."""
        found = self._maps.get((ia, ib))
        if found is None:
            a, b = self.A.molecules[ia], self.B.molecules[ib]
            if b.n_atoms != a.n_atoms or a.n_atoms < 4:
                self._maps[(ia, ib)] = []
                return []
            lab_a = self.labels("A", ia)
            anchors = self._anchors.get(ia)
            if anchors is None:
                anchors = self._anchors[ia] = _anchors(a.coords, lab_a)
            found = self._maps[(ia, ib)] = _isometry_maps(a.coords, lab_a, anchors, b.coords,
                                                          self.labels("B", ib))
        return found


def _correspondence(a_coords, b_coords, plan: _CorrespondencePlan):
    """Geometric atom correspondence between two molecules. Returns (perm, rmsd): A atom i -> B atom perm[i].

    Atoms sharing a signature (precise) are matched by Hungarian assignment on their distance
    matrix; atoms whose signature count differs between the molecules (flexible-bond noise) fall
    back to element-only matching. When the molecules have unequal atom counts per element class
    (allow_partial / disorder drift), the rectangular Hungarian matches min(len_A, len_B) atoms per
    class and the surplus stays unmatched (perm == -1). RMSD is always computed over the matched
    atoms only, so a leftover -1 can never index B with a negative wrap-around.
    """
    perm = plan.template.copy()                           # singles are already scattered into it
    classes = plan.classes
    if classes:
        # One gather per molecule, not two per class: each class is then a contiguous slice of it.
        xa = a_coords[plan.order_a]
        xb = b_coords[plan.order_b]
        for ai, bi, sa, ea, sb, eb in classes:
            r, c = linear_sum_assignment(_cdist(xa[sa:ea], xb[sb:eb]))
            perm[ai[r]] = bi[c]                           # rectangular -> only min(len) get matched
    if plan.complete:
        return perm, rmsd(a_coords, b_coords[perm])
    m = perm >= 0
    return perm, (rmsd(a_coords[m], b_coords[perm[m]]) if m.any() else np.inf)


def _principal_axes(x):
    xc = x - x.mean(axis=0)
    if len(x) < 3:
        return np.eye(3)
    _, _, Vt = np.linalg.svd(xc, full_matrices=True)
    return Vt


def _enumerate_perms(a_groups, b_groups, n_atoms, cap=24):
    """All signature-consistent A->B atom correspondences, or [] if there are too many.

    For small/symmetric molecules this is a few permutations (e.g. a methyl/phosphonate's 3! = 6) and
    gives an exact correspondence to seed from; for large symmetric molecules the count blows past
    the cap and we fall back to principal-axis seeding instead.
    """
    if set(a_groups) != set(b_groups):
        return []
    count = 1
    for s, ai in a_groups.items():
        if len(b_groups[s]) != len(ai):
            return []
        count *= math.factorial(len(ai))
        if count > cap:
            return []
    classes = list(a_groups.items())
    orderings = [list(permutations(b_groups[s])) for s, _ in classes]
    perms = []
    for combo in product(*orderings):
        perm = np.empty(n_atoms, dtype=int)
        for (_, ai), b_order in zip(classes, combo, strict=True):
            perm[ai] = b_order
        perms.append(perm)
    return perms


def _icp(a_coords, b, plan, perm, iters, fits=None, steps=None):
    """Trimmed ICP of one molecule onto another, from a starting atom correspondence.

    ``fits`` and ``steps`` are optional memos shared by every seed of one central pairing. With
    ``a_coords``, ``b`` and ``plan`` fixed for that pairing, both the Kabsch fit of a correspondence
    and the correspondence that fit induces are pure functions of the correspondence, so two
    starting permutations whose trajectories meet share the whole remaining path -- about a third of
    all ICP steps are such re-visits. Replaying them from the memo returns exactly the values the
    unmemoized loop computes, bit for bit.
    """
    if fits is None:
        fits, steps = {}, {}

    def _kabsch(perm, key):                  # fit over matched atoms only (all matched if not partial)
        rt = fits.get(key)
        if rt is None:
            if plan.complete:
                rt = kabsch(b[perm], a_coords)
            else:
                m = perm >= 0
                rt = kabsch(b[perm[m]], a_coords[m])
            fits[key] = rt
        return rt

    prev = np.inf
    key = perm.tobytes()
    for _ in range(iters):
        R, t = _kabsch(perm, key)
        step = steps.get(key)
        if step is None:
            new_perm, r = _correspondence(a_coords, b @ R.T + t, plan)
            step = steps[key] = (new_perm, new_perm.tobytes(), r)
        new_perm, new_key, r = step
        if new_key == key:
            # Fixed point: one more iteration would recompute exactly this (R, t) and correspondence
            # and then stop, so the converged result is already in hand. With every atom matched, so
            # is its RMSD: _correspondence measured it over the same atom pairs of the same
            # transformed B, and gathering rows commutes with the rigid transform of all of them.
            if plan.complete:
                return R, t, r
            break
        perm, key = new_perm, new_key
        if abs(prev - r) < 1e-6:
            R, t = _kabsch(perm, key)
            break
        prev = r
    else:
        R, t = _kabsch(perm, key)
    if plan.complete:
        return R, t, rmsd(b[perm] @ R.T + t, a_coords)
    m = perm >= 0
    return R, t, rmsd(b[perm[m]] @ R.T + t, a_coords[m])


def _mol_align_seeds(plans: _Plans, ia: int, ib: int, invert: bool, iters=6, max_seeds=16):
    """Candidate rigid superpositions of central B onto central A, as distinct (R, t, rmsd) seeds.

    For a symmetric central molecule several near-equal central overlays exist, but only some
    bootstrap the full packing match (e.g. a 2-fold-symmetric arene). We therefore return *all*
    distinct converged seeds (deduplicated by transform) for the caller to grow and rank by packing
    match, rather than collapsing to the single best central-RMSD overlay. Asymmetric molecules
    yield exactly one seed, so this is free in the common case.
    """
    a_coords = plans.A.molecules[ia].coords
    b = plans.B.molecules[ib].coords
    if invert:
        b = -b
    plan = plans.get(ia, ib)
    perms = plans.perms(ia, ib)
    if not perms:                                          # large molecule: principal-axis seeds
        (Va, ac), (Vb, bc) = plans.axes("A", ia), plans.axes("B", ib, invert)
        bc_b = b - bc                                      # the same for all eight sign flips
        perms = [_correspondence(a_coords, bc_b @ (Va.T @ S @ Vb).T + ac, plan)[0]
                 for S in _SIGN_SEEDS]
    distinct = {}
    started = set()
    fits, steps = {}, {}                                   # ICP memo shared by these trajectories
    for perm in perms:
        start = perm.tobytes()
        if start in started:                               # same start -> same ICP trajectory; skip
            continue
        started.add(start)
        R, t, r = _icp(a_coords, b, plan, perm, iters, fits, steps)
        key = (round(r, 3), *np.round(R, 1).ravel().tolist())
        if key not in distinct:
            distinct[key] = (R, t, r)
    return sorted(distinct.values(), key=lambda s: s[2])[:max_seeds]


def _anchors(x, lab) -> np.ndarray:
    """Four well-spread atoms: a rarest-element atom, the farthest from it, the farthest from their
    line, the farthest from their plane.

    Four non-coplanar points determine a rigid motion, so mapping just these onto B enumerates the
    candidate overlays; spreading them out keeps that motion well conditioned and the distance
    filter in _isometry_maps selective.
    """
    n = len(x)
    counts = np.bincount(lab)
    a0 = int(np.argmin(counts[lab] * n + np.arange(n)))          # rarest element, lowest index
    rel = x - x[a0]
    d0 = np.linalg.norm(rel, axis=1)
    a1 = int(np.argmax(d0))
    u = (x[a1] - x[a0]) / (d0[a1] + 1e-12)
    dl = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    dl[[a0, a1]] = -1.0
    a2 = int(np.argmax(dl))
    nrm = np.cross(u, x[a2] - x[a0])
    nrm /= np.linalg.norm(nrm) + 1e-12
    dp = np.abs(rel @ nrm)
    dp[[a0, a1, a2]] = -1.0
    return np.array([a0, a1, a2, int(np.argmax(dp))])


def _anchor_quads(db, lab_b, want, da, delta):
    """B quadruples whose six pairwise distances match A's anchors within delta, and a cap flag.

    Grown one anchor at a time so the distance filter prunes before the next level: an asymmetric
    molecule yields one or two quadruples, an icosahedral cage 240 (120 rotations x 2 chiralities).
    ``db`` is B's pairwise distance matrix (see _isometry_maps), shared by both tolerance passes.
    """
    quads = []
    # A level's element mask, and its distance tests against the anchors already fixed, depend only
    # on the outer indices: form each once per level instead of once per candidate of the level.
    el = [lab_b == w for w in want]

    for j0 in np.flatnonzero(el[0]):
        d0 = db[j0]
        m1 = el[1] & (np.abs(d0 - da[0, 1]) < delta)
        m1[j0] = False
        near2 = el[2] & (np.abs(d0 - da[0, 2]) < delta)
        near3 = el[3] & (np.abs(d0 - da[0, 3]) < delta)
        for j1 in np.flatnonzero(m1):
            d1 = db[j1]
            m2 = near2 & (np.abs(d1 - da[1, 2]) < delta)
            m2[j0] = m2[j1] = False
            near3b = near3 & (np.abs(d1 - da[1, 3]) < delta)
            for j2 in np.flatnonzero(m2):
                m3 = near3b & (np.abs(db[j2] - da[2, 3]) < delta)
                m3[j0] = m3[j1] = m3[j2] = False
                for j3 in np.flatnonzero(m3):
                    quads.append((j0, j1, j2, j3))
                    if len(quads) >= _MAX_QUADS:
                        return quads, True
    return quads, False


def _batch_kabsch(P, Q):
    """Proper rotations R (q,3,3) and translations t (q,3) with P[q] @ R[q].T + t[q] ~ Q.

    The batched twin of align.kabsch: one SVD call for thousands of tiny fits. Every fit in a batch
    targets the same point set Q (n,3) -- A's anchors, or A's central molecule -- so its centroid
    and centred coordinates are formed once rather than once per candidate, and V D U^T becomes a
    single contraction by scaling V's last row instead of materializing q copies of diag(1, 1, d).
    """
    cP = P.mean(axis=1, keepdims=True)
    cQ = Q.mean(axis=0)
    H = np.einsum("qni,nj->qij", P - cP, Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.ones((len(P), 3))
    d[:, 2] = np.sign(np.linalg.det(np.einsum("qji,qkj->qik", Vt, U)))      # reflection guard
    R = np.einsum("qji,qlj->qil", Vt * d[:, :, None], U)                    # V D U^T
    t = cQ - np.einsum("qij,qj->qi", R, cP[:, 0, :])
    return R, t


def _fits(xa, xb, perms):
    """Full-molecule Kabsch overlay of B onto A under each atom map, for both chiralities.

    The improper overlays are the proper overlays of -B and every fit in the stack is independent,
    so both chiralities are fitted in one batch: one SVD call instead of two.
    """
    m = len(perms)
    xp = xb[perms]
    P = np.concatenate((xp, -xp))
    R, t = _batch_kabsch(P, xa)
    r = np.sqrt(((np.einsum("mni,mji->mnj", P, R) + t[:, None, :] - xa) ** 2).sum(-1).mean(-1))
    rl = r.tolist()
    return [list(zip(R[k:k + m], t[k:k + m], rl[k:k + m], strict=True)) for k in (0, m)]


def _isometry_maps(xa, lab_a, anchors, xb, lab_b) -> list:
    """Element-preserving near-isometries A -> B as (proper fit, improper fit) overlay pairs.

    A correspondence between two determinations of the same rigid molecule is a near-isometry, and
    an isometry is pinned down by four points: enumerate the anchor images whose mutual distances
    agree (_anchor_quads), predict every remaining atom's image as the nearest same-element B atom
    under that rigid motion, and keep the maps that are bijective with every atom inside _PREDICT_R.
    One full Kabsch per map then gives an exact molecular overlay to seed from. The atom
    correspondence is recomputed downstream anyway, so a map only has to land (R, t) in the basin.
    """
    n = len(xa)
    k = int(max(lab_a.max(), lab_b.max())) + 1
    if not np.array_equal(np.bincount(lab_a, minlength=k), np.bincount(lab_b, minlength=k)):
        return []
    qa = xa[anchors]
    da = np.linalg.norm(qa[:, None, :] - qa[None, :, :], axis=2)
    # B's pairwise distances, in one pass rather than a column at a time per tolerance pass; this
    # is the same sqrt(sum of squared differences) the per-column norm computes, element for element.
    db = np.linalg.norm(xb[:, None, :] - xb[None, :, :], axis=2)
    # +inf on an element mismatch and an exact zero elsewhere: adding this to a block of predicted
    # distances is the boolean-indexed inf write, bit for bit, in one broadcast pass.
    block = np.where(lab_a[:, None] != lab_b[None, :], np.inf, 0.0)
    out, seen = [], set()
    for delta in _ANCHOR_DELTAS:                # tight first; loosen only if that found too little
        quads, capped = _anchor_quads(db, lab_b, lab_a[anchors], da, delta)
        if capped:
            # thousands of near-isometric anchor embeddings: a distance-ambiguous (very regular)
            # molecule whose alternatives are not worth the tail. Keep the existing pass,
            # and the axis/permutation seeds.
            return out
        if not quads:
            continue
        jq = np.array(quads)
        # Proper and improper anchor embeddings in one batch (the improper ones are -B's proper
        # ones); the fits are independent, so batching is bit for bit the same two passes.
        xq = xb[jq]
        P = np.concatenate((xq, -xq))
        R, t = _batch_kabsch(P, qa)
        fit = np.sqrt(((np.einsum("qni,qji->qnj", P, R) + t[:, None, :] - qa) ** 2)
                      .sum(-1).mean(-1))
        nq = len(jq)
        for lo, xs in ((0, xb), (nq, -xb)):                 # improper maps via inverted B
            ok = np.flatnonzero(fit[lo:lo + nq] <= delta) + lo
            for c0 in range(0, len(ok), 256):               # chunked: the (chunk, n, n) distances
                idx = ok[c0:c0 + 256]
                bt = np.einsum("nj,qij->qni", xs, R[idx]) + t[idx][:, None, :]
                # squared distances accumulated one coordinate at a time: the same arithmetic as
                # ((xa - bt) ** 2).sum(-1), without its (chunk, n, n, 3) temporary
                e = xa[None, :, None, 0] - bt[:, None, :, 0]
                d = e * e
                e = xa[None, :, None, 1] - bt[:, None, :, 1]
                d += e * e
                e = xa[None, :, None, 2] - bt[:, None, :, 2]
                d += e * e
                np.sqrt(d, out=d)
                d += block
                perms = []
                for ci in np.flatnonzero(d.min(axis=2).max(axis=1) <= _PREDICT_R):
                    pm = d[ci].argmin(axis=1)
                    key = pm.tobytes()
                    if key not in seen and np.bincount(pm, minlength=n).max() == 1:
                        seen.add(key)
                        perms.append(pm)
                if perms:
                    out += list(zip(*_fits(xa, xb, np.array(perms)), strict=True))
        if len(out) >= 2:
            break
    return out


def _isometry_seeds(plans: _Plans, ia: int, ib: int, invert: bool, base) -> list:
    """Central overlays from _isometry_maps that the axis/permutation seeds cannot reach.

    A molecule with a C3 or higher axis has a degenerate inertia tensor, so the 8 principal-axis
    sign flips sample only a few of its orientations and the packing match is missed outright
    (ferrocene, a cage, hexasubstituted arenes). Maps supply the rest exactly.
    """
    found = plans.maps(ia, ib)
    if not found:
        return []
    seeds = []
    known = [(Rs, ts) for Rs, ts, _ in base]
    stack = np.array([Rs for Rs, _ in known]) if known else np.empty((0, 3, 3))
    for proper, improper in found:
        R, t, r = improper if invert else proper
        if r >= _SEED_RMSD_CAP:
            continue
        # a map landing on an already-converged pose (~1 deg, 0.1 A) refines into the same basin.
        # The rotation test is the selective one, so it is taken against every known pose at once
        # and only the handful that pass it are measured for translation.
        close = np.abs(stack - R).max(axis=(1, 2)) < 0.02
        if any(np.linalg.norm(t - known[j][1]) < 0.1 for j in np.flatnonzero(close)):
            continue
        seeds.append((R, t, r))
        known.append((R, t))
        stack = np.concatenate((stack, R[None]))
    return seeds


def _screen_images(b_cent, b_spec, shell_spec, centre, radius):
    """B images a seed could bring onto A's shell, with a +inf block on species mismatches.

    Every seed maps B's central molecule onto A's, so only images within the shell's reach of that
    centroid can ever be assigned. Independent of the seed pose, hence computed once per pairing.
    """
    keep = np.flatnonzero(np.linalg.norm(b_cent - centre, axis=1) <= 2.0 * radius + 6.0)
    # +inf on a species mismatch, exact zero elsewhere; added, not written (as in _isometry_maps).
    return b_cent[keep], np.where(shell_spec[:, None] != b_spec[keep][None, :], np.inf, 0.0)


def _screen_score(shell_cent, cent, block, R, t, sign) -> int:
    """How many of A's shell centroids land on a same-species B image under the seed pose.

    A centroid-only proxy for n_matched, costing one cdist instead of a full shell refinement: it
    orders the enumerated alternatives and lets those that cannot overtake the best result so far
    skip refinement entirely. Measured on the benchmark sets, not proven lossless.
    """
    if not len(cent):
        return 0
    d = cdist(shell_cent, (sign * cent) @ R.T + t)
    d += block
    return int((d.min(axis=1) < _SCREEN_R).sum())


def _interior_order(struct: Structure, species: str) -> list[int]:
    """Indices of the given species' molecules, ordered from the structure's centre outward."""
    prim = [i for i, m in enumerate(struct.molecules) if m.species_key == species]
    center = np.array([m.centroid for m in struct.molecules]).mean(axis=0)

    def distance(i):
        return float(np.linalg.norm(struct.molecules[i].centroid - center))

    if struct.cell is None:
        # Equivalent cluster sites part by ~1e-15: round so they tie and the index picks the same one.
        return sorted(prim, key=lambda i: (round(distance(i), 6), i))
    return sorted(prim, key=distance)


_TRIU = {}


def _triu_flat(n) -> np.ndarray:
    """Flat indices of the strict upper triangle of an (n, n) array, cached per size."""
    idx = _TRIU.get(n)
    if idx is None:
        r, c = np.triu_indices(n, 1)
        idx = _TRIU[n] = r * n + c
    return idx


def _unique_conformers(mols, indices) -> list[int]:
    """Among molecule indices (same species), keep one representative per distinct conformer.

    The conformer signature is the sorted rounded interatomic-distance multiset. It is built from
    the same arithmetic as ``norm(c[:, None] - c[None, :], axis=2)`` accumulated one coordinate at
    a time -- no (n, n, 3) temporary -- and keyed by the rounded array's bytes, which for a set of
    non-negative distances is exactly the equality of the tuple of those floats.
    """
    seen, keep = set(), []
    for i in indices:
        c = mols[i].coords
        e = c[:, None, 0] - c[None, :, 0]
        d = e * e
        e = c[:, None, 1] - c[None, :, 1]
        d += e * e
        e = c[:, None, 2] - c[None, :, 2]
        d += e * e
        np.sqrt(d, out=d)
        sig = np.round(np.sort(d.reshape(-1)[_triu_flat(len(c))]), 2).tobytes()
        if sig not in seen:
            seen.add(sig)
            keep.append(i)
    return keep


def _centrals(struct: Structure, species: str, max_central: int) -> list[int]:
    """Central-molecule candidates of one species, memoized on the structure.

    A pure function of the structure, so a batch run that compares one structure against hundreds
    of others pays for the conformer signatures once rather than once per pair. Read-only for
    callers; the cache is dropped on pickling like every other derived table.
    """
    key = ("centrals", species, int(max_central))
    got = struct.cache.get(key)
    if got is None:
        got = struct.cache[key] = _unique_conformers(
            struct.molecules, _interior_order(struct, species))[:max_central]
    return got


def _element_count_distance(a, b) -> int:
    """Element-multiset L1 distance between two representative molecules."""
    ca, cb = Counter(a.elements), Counter(b.elements)
    return sum(abs(ca.get(e, 0) - cb.get(e, 0)) for e in set(ca) | set(cb))


def _primary_species(struct: Structure) -> str:
    """Choose the largest molecular species with a deterministic tie break."""
    atom_counts: dict[str, int] = {}
    for molecule in struct.molecules:
        atom_counts.setdefault(molecule.species_key, molecule.n_atoms)
    return min(atom_counts, key=lambda key: (-atom_counts[key], key))


def compare_structures(A: Structure, B: Structure, *, n: int = 15, allow_inversion: bool = True,
                       mol_rmsd_tol: float = 1.0, match_fraction: float = 0.5,
                       max_central: int = 6, max_icp: int = 8,
                       allow_partial: bool = False,
                       refine_schedule: tuple[float, ...] = (1.5,)) -> ComparisonResult:
    specB = {m.species_key for m in B.molecules}

    def representative(struct, key):
        return next(m for m in struct.molecules if m.species_key == key)

    # Anchor on A's main component; require only that it is present in B (not full multiset
    # equality). This tolerates minor-solvent / disorder perception drift between determinations
    # and supports same-cation/different-salt comparison, while still refusing to match two
    # compounds that merely share a small solvent. Molecules of species absent from B simply go
    # unmatched in the shell.
    primary = _primary_species(A)
    # spec_pair maps each B species to the A species it stands in for (identity for exact matches);
    # remapping b_spec through it lets the same-species assignment masking accept the paired species.
    spec_pair = {k: k for k in specB}
    primaryB = primary
    if primary not in specB:
        # allow_partial: pair the primary with the closest B species if its actual element multiset
        # differs by only a small fraction of the primary's heavy atoms (disorder / borderline-bond
        # drift, not a different compound). `species_key` may be an opaque conformer identity and
        # must never be parsed as a formula. Exact-key behavior above remains unchanged.
        if allow_partial:
            primary_molecule = representative(A, primary)
            primaryB = min(
                specB,
                key=lambda key: (
                    _element_count_distance(primary_molecule, representative(B, key)),
                    key,
                ),
            )
            if (
                _element_count_distance(primary_molecule, representative(B, primaryB))
                > 0.15 * primary_molecule.n_atoms
            ):
                return ComparisonResult(float("nan"), n, 0, "failed",
                                        message=f"main component {primary} not in other structure")
            spec_pair[primaryB] = primary
        else:
            return ComparisonResult(float("nan"), n, 0, "failed",
                                    message=f"main component {primary} not in other structure")
    centralsA = _centrals(A, primary, max_central)
    centralsB = _centrals(B, primaryB, max_central)

    plans = _Plans(A, B)
    b_cent, b_base, b_off = molecule_images(B, n, contact=True)
    # Species identity is only ever tested for equality, so intern the keys to small integers:
    # the per-iteration mismatch matrix then compares ints instead of numpy unicode strings.
    codes: dict[str, int] = {}

    def code(key: str) -> int:
        c = codes.get(key)
        if c is None:
            c = codes[key] = len(codes)
        return c

    base_spec = np.fromiter((code(spec_pair[m.species_key]) for m in B.molecules),
                            dtype=np.intp, count=len(B.molecules))
    b_spec = base_spec[b_base]
    b_cent_sq = np.einsum("ij,ij->i", b_cent, b_cent)   # for the squared-distance shell filter
    inversions = (False, True) if allow_inversion else (False,)

    schedule = tuple(float(x) for x in refine_schedule)
    best, best_r1 = None, np.inf
    best_n, full_score = 0, -1     # most molecules matched so far, and the score that first got n
    for ia in centralsA:
        images = nearest_shell_images(A, ia, n)
        shell = [materialize(A, base, offset) for base, offset in images]
        shell_bases = [base for base, _ in images]
        shell_cent = np.array([m.centroid for m in shell])
        shell_spec = np.fromiter((code(m.species_key) for m in shell), dtype=np.intp,
                                 count=len(shell))
        # Constant over every seed that grows from this central molecule.
        ctr = shell_cent.mean(axis=0)
        radius = float(np.linalg.norm(shell_cent - ctr, axis=1).max())
        shell_rg = radius_of_gyration(np.vstack([m.coords for m in shell]))
        for ib in centralsB:
            screen = None                 # built lazily: only a symmetric central pair needs it
            for invert in inversions:
                base = _mol_align_seeds(plans, ia, ib, invert)
                extra = _isometry_seeds(plans, ia, ib, invert, base)
                if extra and screen is None:
                    screen = _screen_images(b_cent, b_spec, shell_spec,
                                            B.molecules[ib].centroid, radius)
                sign = -1.0 if invert else 1.0
                extra = sorted(((_screen_score(shell_cent, *screen, R, t, sign), R, t, r1)
                                for R, t, r1 in extra), key=lambda c: (-c[0], c[3]))
                for score, R, t, r1 in [(None, *s) for s in base] + extra:
                    best_r1 = min(best_r1, r1)
                    # An enumerated alternative has to beat the best match so far to be worth a
                    # refinement, and once every molecule is matched it can only lower the RMSD,
                    # which takes the same shell score. The seeds above are always refined.
                    if score is not None and (score < best_n
                                              or (best_n >= n and score < full_score)):
                        continue
                    res = _refine(shell, shell_bases, shell_cent, shell_spec, B, plans,
                                  b_cent, b_cent_sq, b_base, b_off, b_spec, R, t, invert, n,
                                  mol_rmsd_tol, max_icp, r1, ctr, radius, shell_rg, schedule)
                    res.central_a = ia            # record which A central won, for overlay/QA
                    if res.n_matched > best_n:
                        best_n = res.n_matched
                        if best_n >= n and full_score < 0:
                            full_score = n if score is None else score
                    if _better(res, best):
                        best = res
    if best is None:  # central conformer never aligned a packing; still report the conformer RMSD
        return ComparisonResult(float("nan"), n, 0, "failed", rmsd_1=best_r1,
                                message="central conformer matched but packing did not")

    need = int(np.ceil(n * match_fraction))
    best.status = "full" if best.n_matched >= n else ("partial" if best.n_matched >= need else "failed")
    return best


def _assign(env, spec_mismatch, R, t, max_rmsd):
    """Hungarian molecule assignment (same species), with per-molecule geometric correspondence.

    Returns list of (shell_index, A_coords, signed_B_raw_permuted, per_molecule_rmsd), where the
    coords are restricted to the atoms actually matched (all of them unless atom counts differ).

    ``env`` is the constant bundle _refine builds once per refinement: everything used here that
    does not depend on (R, t) -- the sign-folded image centroids and lattice offsets, the shell's
    coordinate and species lists, the per-base sign-folded molecule coordinates -- is hoisted out
    of the ICP loop, which calls this once per iteration.
    """
    (shell_coords, shell_keys, shell_bases, plans, shell_cent, sb_cent, b_base, sb_off,
     sign, b_mols, b_keys, b_signed) = env
    cost = _cdist(shell_cent, sb_cent @ R.T + t)
    if spec_mismatch is not None:            # None == single species on both sides: nothing to mask
        cost[spec_mismatch] = 1e9
    rows, cols = linear_sum_assignment(cost)
    # One gather instead of a 2-D scalar index per row, and plain Python ints/floats in the loop.
    out = []
    for ri, ci, c in zip(rows.tolist(), cols.tolist(), cost[rows, cols].tolist(), strict=True):
        if c >= 1e9:
            continue
        base = b_base[ci]
        # With every atom present, molecular RMSD is bounded below by centroid distance. Avoid the
        # atom assignment -- and the image's coordinates -- when that bound already excludes the
        # pair. Different formulas may be compared under allow_partial; their matched-atom centroid
        # need not obey this bound.
        if c >= max_rmsd and shell_keys[ri] == b_keys[base]:
            continue
        # sign * (coords + offset) == (sign * coords) + (sign * offset) exactly for sign = +-1
        # (negation and round-to-nearest are both sign symmetric), so the per-base sign fold is
        # hoisted and only the 3-vector offset is added per image.
        sb = b_signed.get(base)
        if sb is None:
            sb = b_signed[base] = sign * b_mols[base].coords
        sb_raw = sb + sb_off[ci]
        plan = plans.get(shell_bases[ri], base)
        perm, r = _correspondence(shell_coords[ri], sb_raw @ R.T + t, plan)
        if plan.complete:
            out.append((ri, shell_coords[ri], sb_raw[perm], r))
            continue
        m = perm >= 0
        out.append((ri, shell_coords[ri][m], sb_raw[perm[m]], r))
    return out


def _refine(shell, shell_bases, shell_cent, shell_spec, B, plans, b_cent, b_cent_sq,
            b_base, b_off, b_spec, R, t, invert, n, tol, max_icp, rmsd_1,
            ctr, radius, shell_rg, schedule=()) -> ComparisonResult:
    sign = -1 if invert else 1
    # Restrict B images to the shell neighbourhood under the seed transform (cheap per-iteration cost).
    # R is orthogonal, so ||(sign c) R^T + t - ctr|| == ||sign c - q|| with q = (ctr - t) R: the test
    # needs no per-image matrix product, and squaring both sides removes the square root as well.
    # Only ~6% of the images survive, so gathering by index beats four boolean masks.
    q = (ctr - t) @ R
    limit = 2.0 * radius + 6.0
    d2 = b_cent_sq - (2.0 * sign) * (b_cent @ q) + float(q @ q)
    near = np.flatnonzero(d2 <= limit * limit)
    b_cent, b_base, b_off, b_spec = b_cent[near], b_base[near], b_off[near], b_spec[near]
    spec_mismatch = shell_spec[:, None] != b_spec[None, :]   # constant across the ICP iterations
    if not spec_mismatch.any():        # single-species pair: skip the full-matrix sentinel write
        spec_mismatch = None

    # Everything _assign needs that does not depend on (R, t), built once for the whole refinement.
    env = ([m.coords for m in shell], [m.species_key for m in shell], shell_bases, plans,
           shell_cent, sign * b_cent, b_base, sign * b_off, sign, B.molecules, plans.b_keys, {})

    # A seed whose central molecule is exact but whose neighbours sit 1-2 A off has a single inlier,
    # and a Kabsch fit over one molecule reproduces its own pose, so the trimmed ICP cannot move.
    # The first iterations therefore fit a wider set of molecules; the match itself is unchanged,
    # because the assignment below is always recomputed at `tol` under the final pose.
    prev = np.inf
    for step in schedule + (tol,) * max(1, max_icp - len(schedule)):
        step = max(step, tol)
        inliers = [(a, b) for _, a, b, r in _assign(env, spec_mismatch, R, t, step) if r < step]
        if not inliers:
            break
        # np.concatenate == np.vstack for (n, 3) blocks, without the atleast_2d preamble.
        A_stack = np.concatenate([a for a, _ in inliers])
        B_stack = np.concatenate([b for _, b in inliers])
        R, t = kabsch(B_stack, A_stack)
        cur = rmsd(B_stack @ R.T + t, A_stack)
        if step == tol and abs(prev - cur) < 1e-5:
            break
        prev = cur

    assigned = _assign(env, spec_mismatch, R, t, tol)
    A_in, B_in, per_mol = [], [], []
    for _, a, b, r in assigned:
        if r < tol:
            A_in.append(a)
            B_in.append(b @ R.T + t)     # == _apply(b, 1, R, t): the sign is already folded in
            per_mol.append(r)
    if not per_mol:
        return ComparisonResult(float("nan"), n, 0, "failed", rmsd_1=rmsd_1)
    A_stack, B_stack = np.concatenate(A_in), np.concatenate(B_in)
    return ComparisonResult(
        rmsd_n=rmsd(A_stack, B_stack), n=n, n_matched=len(per_mol), status="",
        per_molecule_rmsd=sorted(per_mol),
        transform=np.block([[R, t[:, None]], [np.zeros(3), 1.0]]),
        radius_of_gyration=(shell_rg, radius_of_gyration(B_stack)),
        rmsd_1=rmsd_1, inverted=invert)


def _better(res: ComparisonResult, best: ComparisonResult | None) -> bool:
    if res is None or np.isnan(res.rmsd_n):
        return False
    if best is None:
        return True
    if res.n_matched != best.n_matched:
        return res.n_matched > best.n_matched
    return res.rmsd_n < best.rmsd_n
