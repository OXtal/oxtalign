"""Coordination-shell construction: lattice images of molecules and nearest-contact selection.

The shell is always anchored to a chosen central molecule of structure A; B's shell is then
defined by spatial matching (see matching.py), never by an independent nearest-N in B.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .cell import lattice_shifts
from .model import Molecule, Structure


def base_centroids(struct: Structure) -> np.ndarray:
    return np.array([m.centroid for m in struct.molecules])


def molecular_radii(struct: Structure) -> np.ndarray:
    """Each molecule's bounding radius about its own centroid.

    Memoized on the structure: every central molecule's shell, and every comparison the structure
    takes part in, needs the same vector.
    """
    radii = struct.cache.get("radii")
    if radii is None:
        radii = struct.cache["radii"] = np.asarray(
            [
                np.linalg.norm(molecule.coords - molecule.centroid, axis=1).max(initial=0.0)
                for molecule in struct.molecules
            ],
            dtype=np.float64,
        )
    return radii


def stacked_atoms(struct: Structure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All molecule coordinates in one (N,3) array plus each molecule's (start, count) into it.

    Lets a batch of molecule images be built with a handful of array operations instead of one
    Python-level array addition per image. Memoized alongside the radii.
    """
    stacked = struct.cache.get("stacked")
    if stacked is None:
        counts = np.array([m.n_atoms for m in struct.molecules], dtype=np.intp)
        coords = (np.vstack([m.coords for m in struct.molecules]) if struct.molecules
                  else np.zeros((0, 3)))
        starts = np.cumsum(counts) - counts
        stacked = struct.cache["stacked"] = (coords, starts, counts)
    return stacked


def image_radius(struct: Structure, n: int, *, contact: bool = False) -> int:
    """How many cells out to tile so the n nearest molecules are available.

    Sized from the cell's *narrowest* perpendicular width: a sphere holding ~n molecules has radius
    R, so we need ceil(R / min_perp_width) cells (+1 pad). For a contact-nearest shell, a candidate
    centroid can be one central radius plus one candidate radius beyond that sphere; ``contact=True``
    adds the conservative maximum of that padding. An isotropic (n/Z)^(1/3) cell count without these
    geometric terms under-tiles elongated/thin cells and can silently drop a nearer contact.
    """
    cell, z = struct.cell, max(1, len(struct.molecules))
    volume = cell.volume                                  # a determinant; ask for it once
    R = (3.0 * n * volume / (4.0 * math.pi * z)) ** (1.0 / 3.0)
    if contact:
        R += 2.0 * float(molecular_radii(struct).max(initial=0.0))
    lat = cell.lattice_vectors
    perp = [volume / np.linalg.norm(np.cross(lat[(i + 1) % 3], lat[(i + 2) % 3])) for i in range(3)]
    return max(2, int(math.ceil(R / min(perp))) + 1)


def molecule_images(struct: Structure, n: int, *, contact: bool = False):
    """All molecule images out to the required radius.

    Returns (centroids (M,3), base_index (M,), offsets (M,3)). For a cell-less cluster the
    molecules are returned as-is. ``contact=True`` includes the molecular-radius padding required
    by a closest-interatomic-contact shell or by matching against one.
    """
    key = ("images", n, contact)
    images = struct.cache.get(key)
    if images is not None:                                # read-only for every caller
        return images
    base = base_centroids(struct)
    if struct.cell is None:
        z = len(struct.molecules)
        images = (base, np.arange(z), np.zeros((z, 3)))
    else:
        offsets = lattice_shifts(struct.cell, image_radius(struct, n, contact=contact))
        z = len(base)
        images = ((base[None, :, :] + offsets[:, None, :]).reshape(-1, 3),
                  np.tile(np.arange(z), len(offsets)),
                  np.repeat(offsets, z, axis=0))
    struct.cache[key] = images
    return images


def materialize(struct: Structure, base_index: int, offset: np.ndarray) -> Molecule:
    return struct.molecules[base_index].translated(offset)


def nearest_shell(struct: Structure, central_base: int, n: int) -> list[Molecule]:
    """The central molecule plus the ``n - 1`` images with the closest interatomic contact."""
    images = nearest_shell_images(struct, central_base, n)
    return [materialize(struct, base, offset) for base, offset in images]


def nearest_shell_images(struct: Structure, central_base: int, n: int) -> list[tuple[int, np.ndarray]]:
    """``(base_index, lattice_offset)`` of the central molecule plus its ``n - 1`` closest-contact images.

    COMPACK's packing shell is contact-nearest, which matters for elongated molecules: two touching
    molecules can have centroids farther apart than several non-touching images.  Centroid/radius
    lower bounds keep the atom-level work small while preserving the exact contact-distance order.

    Memoized per (central molecule, n): matching seeds and screens the same few shells repeatedly,
    and a structure compared against many others rebuilds them for every pair. Read-only for callers.
    """
    key = ("shell", int(central_base), int(n))
    cached = struct.cache.get(key)
    if cached is not None:
        return cached
    centroids, base_index, offsets = molecule_images(struct, n, contact=True)
    if not len(centroids):
        struct.cache[key] = []
        return []

    central = struct.molecules[central_base]
    radii = molecular_radii(struct)
    centroid_distance = np.linalg.norm(centroids - central.centroid, axis=1)
    lower_bound = np.maximum(
        0.0,
        centroid_distance - radii[central_base] - radii[base_index],
    )

    tree = cKDTree(central.coords)
    mol_coords, mol_starts, mol_counts = stacked_atoms(struct)
    evaluated_indices: list[int] = []
    evaluated_distances: list[float] = []
    cutoff = float("inf")
    required = min(int(n), len(lower_bound))
    # Evaluate candidates in lower-bound order, a chunk at a time (one batched KD-tree query per
    # chunk). Stopping once the next unevaluated candidate's lower bound exceeds the current n-th
    # smallest contact distance is exact: nothing skipped can enter the shell, and evaluating a few
    # extra candidates per chunk never changes which n are selected.
    chunk = max(4 * required, 16)
    # The loop almost always stops after the first chunk, so select that chunk with a partial sort
    # and only pay for a full ordering if a second one is really needed. argpartition puts the
    # chunk-th smallest bound in position with everything smaller before it -- exactly what the
    # stopping rule inspects -- and every candidate in a chunk is evaluated, so their relative order
    # inside it does not matter.
    tail_sorted = chunk >= len(lower_bound)
    bound_order = (np.argsort(lower_bound, kind="stable") if tail_sorted
                   else np.argpartition(lower_bound, chunk))
    position = 0
    while position < len(bound_order):
        if len(evaluated_indices) >= required and lower_bound[bound_order[position]] > cutoff:
            break
        if position and not tail_sorted:   # first chunk was not decisive
            # Sort only what is still unevaluated: re-sorting the whole array could reshuffle the
            # already-evaluated prefix across a tie and evaluate some candidate twice.
            tail = bound_order[position:]
            bound_order = np.concatenate(
                (bound_order[:position], tail[np.argsort(lower_bound[tail], kind="stable")]))
            tail_sorted = True
        batch = bound_order[position:position + chunk]
        position += len(batch)
        bases = base_index[batch]
        counts = mol_counts[bases]
        starts = np.cumsum(counts) - counts
        # Gather every image's atoms in one shot: atom k of image j lives at mol_starts[base] + k.
        atoms = np.repeat(mol_starts[bases] - starts, counts) + np.arange(int(counts.sum()))
        points = mol_coords[atoms] + np.repeat(offsets[batch], counts, axis=0)
        distances, _ = tree.query(points, k=1)
        evaluated_indices.extend(int(i) for i in batch)
        evaluated_distances.extend(float(d) for d in np.minimum.reduceat(distances, starts))
        if len(evaluated_indices) >= required:
            cutoff = float(np.partition(evaluated_distances, required - 1)[required - 1])

    order = [
        image_index
        for _, image_index in sorted(
            zip(evaluated_distances, evaluated_indices, strict=True),
            key=lambda item: (
                item[0],
                not (
                    base_index[item[1]] == central_base
                    and np.count_nonzero(offsets[item[1]]) == 0
                ),
                item[1],
            ),
        )[:required]
    ]
    shell = struct.cache[key] = [(int(base_index[i]), offsets[i]) for i in order]
    return shell
