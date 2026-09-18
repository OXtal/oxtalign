"""CIF reader producing a RawStructure.

Two input flavours are handled transparently:

* Small-molecule CIF (experimental): fractional coords + unit cell + symmetry operators.
  Parsed via ``gemmi.read_small_structure`` (strips e.s.d. ``(n)`` notation, exposes symops).
* mmCIF / PDBx (predicted): Cartesian coords, no cell, explicit ``_chem_comp_bond`` connectivity,
  molecule copies tagged by ``label_asym_id``.

Coordinates in the returned RawStructure are always CARTESIAN.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass

import gemmi
import numpy as np

from .model import RawStructure, UnitCell

_ESD = re.compile(r"\(\d+\)")


@dataclass(frozen=True)
class DisorderGroup:
    """One mutually exclusive core-CIF disorder group."""

    group_id: str
    labels: frozenset[str]
    weight: float


@dataclass(frozen=True)
class DisorderAssembly:
    """A set of mutually exclusive groups, ordered by decreasing occupancy."""

    assembly_id: str
    groups: tuple[DisorderGroup, ...]


@dataclass(frozen=True)
class DisorderModel:
    """Normalized joint disorder model parsed from one atom-site loop.

    Assemblies are independent choices. Sites without a disorder group are present in every configuration.
    A partial-occupancy site without a complete alternative choice cannot be resolved into discrete
    configurations and is flagged separately so downstream ingestion can keep it for audit while excluding
    it from training.
    """

    ordered_labels: frozenset[str]
    assemblies: tuple[DisorderAssembly, ...]
    has_ungrouped_partial: bool
    occupancy_fallback: bool

    @property
    def has_grouped_disorder(self) -> bool:
        return bool(self.assemblies)

    def labels_for(self, group_indices: tuple[int, ...]) -> frozenset[str]:
        if len(group_indices) != len(self.assemblies):
            raise ValueError("one group index is required for every disorder assembly")
        labels = set(self.ordered_labels)
        for assembly, index in zip(self.assemblies, group_indices, strict=True):
            labels.update(assembly.groups[index].labels)
        return frozenset(labels)


def _normalize_element(symbol: str) -> str:
    """'BR' -> 'Br', 'CO' -> 'Co', 'c' -> 'C'. Strips charge/ordinal suffixes."""
    s = re.sub(r"[^A-Za-z]", "", symbol)
    if not s:
        return symbol
    return s[0].upper() + s[1:].lower()


def _to_float(tok: str) -> float:
    return float(_ESD.sub("", tok))


def read(
    path: str,
    *,
    block: gemmi.cif.Block | None = None,
    disorder: str = "all",
    small_structure=None,
    selected_labels: Collection[str] | None = None,
) -> RawStructure:
    """Read a CIF, optionally reusing an already parsed gemmi block.

    Bulk ingestion needs both crystallographic metadata and coordinates.  Accepting
    ``block`` avoids parsing every CIF a second time solely to collect metadata.
    """
    if disorder not in {"all", "dominant"}:
        raise ValueError(f"unknown disorder policy: {disorder!r}")
    if selected_labels is not None and disorder != "all":
        raise ValueError("selected_labels and a non-'all' disorder policy are mutually exclusive")
    if block is None:
        block = gemmi.cif.read(str(path)).sole_block()
    if block.find_loop("_atom_site.Cartn_x") or block.find_loop("_atom_site_Cartn_x"):
        return _read_mmcif(block, str(path))
    if block.find_loop("_atom_site_fract_x"):
        return _read_small_molecule(
            block,
            str(path),
            disorder=disorder,
            small_structure=small_structure,
            selected_labels=selected_labels,
        )
    raise ValueError(
        f"{path}: no recognized atom-site coordinates (_atom_site.Cartn_x or _atom_site_fract_x)"
    )


def _present(value) -> bool:
    return str(value).strip() not in {"", ".", "?"}


def _occupancy(value) -> float | None:
    if not _present(value):
        return None
    try:
        return float(_ESD.sub("", str(value).strip().strip("'\"")))
    except ValueError:
        return None


def _token_key(value: str) -> tuple[int, int | str]:
    """Natural ordering for numeric group IDs, lexical ordering for arbitrary CIF codes."""
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def disorder_model(
    block: gemmi.cif.Block,
    *,
    occupancy_normalizer: Callable[[float], float] | None = None,
    excluded_indices: Collection[int] = (),
) -> DisorderModel:
    """Parse grouped and unresolved partial occupancy from a core-CIF atom-site loop.

    Group probability is the median site occupancy, avoiding a bias toward alternatives containing more
    atoms. Malformed probabilities make the entire affected assembly uniform and set ``occupancy_fallback``;
    mixing trustworthy and untrustworthy group probabilities would imply unsupported precision.
    """
    labels = [str(value) for value in block.find_loop("_atom_site_label")]
    groups = list(block.find_loop("_atom_site_disorder_group"))
    assemblies = list(block.find_loop("_atom_site_disorder_assembly"))
    occupancies = list(block.find_loop("_atom_site_occupancy"))

    excluded = set(excluded_indices)
    ordered: set[str] = set()
    grouped: dict[str, dict[str, dict[str, list]]] = {}
    has_ungrouped_partial = False
    for index, label in enumerate(labels):
        if index in excluded:
            continue
        group = str(groups[index]).strip() if index < len(groups) else "."
        occupancy = _occupancy(occupancies[index]) if index < len(occupancies) else 1.0
        if occupancy is not None and occupancy_normalizer is not None:
            occupancy = float(occupancy_normalizer(occupancy))
        if not _present(group):
            ordered.add(label)
            if occupancy is None or occupancy < 0.999 or occupancy > 1.001:
                has_ungrouped_partial = True
            continue
        assembly = str(assemblies[index]).strip() if index < len(assemblies) else "."
        assembly_id = assembly if _present(assembly) else "__unassigned__"
        record = grouped.setdefault(assembly_id, {}).setdefault(
            group,
            {"labels": [], "occupancies": []},
        )
        record["labels"].append(label)
        record["occupancies"].append(occupancy)

    parsed_assemblies = []
    any_fallback = False
    for assembly_id in sorted(grouped, key=_token_key):
        groups_by_id = grouped[assembly_id]
        raw_weights = []
        malformed = False
        for group_id in sorted(groups_by_id, key=_token_key):
            values = groups_by_id[group_id]["occupancies"]
            valid = [value for value in values if value is not None and 0.0 <= value <= 1.0]
            malformed |= len(valid) != len(values) or not valid
            raw_weights.append(float(np.median(valid)) if valid else np.nan)
        raw = np.asarray(raw_weights, dtype=float)
        fallback = malformed or not np.isfinite(raw).all() or float(raw.sum()) <= 0.0
        if len(groups_by_id) == 1:
            # A singleton full-occupancy group is only an annotation and is present in every state. A
            # singleton partial group has no deposited alternative, so it must not be promoted to a complete
            # probability-one configuration.
            only = next(iter(groups_by_id.values()))
            ordered.update(only["labels"])
            if fallback or not 0.999 <= float(raw[0]) <= 1.001:
                has_ungrouped_partial = True
            any_fallback |= fallback
            continue
        if fallback:
            weights = np.full(len(raw), 1.0 / len(raw), dtype=float)
            any_fallback = True
        else:
            weights = raw / raw.sum()
        parsed_groups = [
            DisorderGroup(
                group_id=group_id,
                labels=frozenset(groups_by_id[group_id]["labels"]),
                weight=float(weight),
            )
            for group_id, weight in zip(
                sorted(groups_by_id, key=_token_key),
                weights.tolist(),
                strict=True,
            )
        ]
        parsed_groups.sort(key=lambda group: (-group.weight, _token_key(group.group_id)))
        parsed_assemblies.append(DisorderAssembly(assembly_id, tuple(parsed_groups)))

    return DisorderModel(
        ordered_labels=frozenset(ordered),
        assemblies=tuple(parsed_assemblies),
        has_ungrouped_partial=has_ungrouped_partial,
        occupancy_fallback=any_fallback,
    )


def dominant_disorder_labels(block: gemmi.cif.Block) -> set[str] | None:
    """Labels in the dominant joint CIF disorder configuration, or ``None`` if ungrouped.

    Core CIF represents alternatives with ``_atom_site_disorder_assembly`` and
    ``_atom_site_disorder_group``. Occupancy is a per-site value, so group ranking uses the median rather
    than a sum that would favor alternatives containing more atoms. Ties use the group label to remain
    deterministic by retaining the first group encountered. Ordered sites are always retained.
    """
    model = disorder_model(block)
    if not model.assemblies:
        return None
    return set(model.labels_for((0,) * len(model.assemblies)))


def _read_small_molecule(
    block: gemmi.cif.Block,
    path: str,
    *,
    disorder: str,
    small_structure=None,
    selected_labels: Collection[str] | None = None,
) -> RawStructure:
    st = small_structure or gemmi.make_small_structure_from_block(block)
    g = st.cell
    if g.a == 0:
        raise ValueError(f"{path}: small-molecule CIF without a valid unit cell")
    orth = np.array(g.orth.mat.tolist())
    frac = np.array(g.frac.mat.tolist())
    symops = list(st.symops) or ["x,y,z"]
    cell = UnitCell(g.a, g.b, g.c, g.alpha, g.beta, g.gamma, symops, orth, frac)

    allowed_labels = (
        set(selected_labels)
        if selected_labels is not None
        else dominant_disorder_labels(block)
        if disorder == "dominant"
        else None
    )
    elements, fracs, labels, occs = [], [], [], []
    for s in st.sites:
        if allowed_labels is not None and s.label not in allowed_labels:
            continue
        elements.append(s.element.name)
        fracs.append([s.fract.x, s.fract.y, s.fract.z])
        labels.append(s.label)
        occs.append(float(s.occ))
    fracs = np.array(fracs, dtype=float)
    coords = fracs @ orth.T
    return RawStructure(elements, coords, labels, np.array(occs), cell, None, path)


def _read_mmcif(block: gemmi.cif.Block, path: str) -> RawStructure:
    cols = ["type_symbol", "label_atom_id", "label_asym_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    table = block.find("_atom_site.", cols)
    if not len(table):
        table = block.find("_atom_site_", cols)
    elements, coords, labels = [], [], []
    asym_ids, comp_ids, atom_ids = [], [], []
    index_of: dict[tuple[str, str], int] = {}
    for i, row in enumerate(table):
        elements.append(_normalize_element(row[0]))
        atom_id, asym_id, comp_id = row[1], row[2], row[3]
        labels.append(atom_id)
        coords.append([_to_float(row[4]), _to_float(row[5]), _to_float(row[6])])
        asym_ids.append(asym_id)
        comp_ids.append(comp_id)
        atom_ids.append(atom_id)
        index_of[(asym_id, atom_id)] = i
    coords = np.array(coords, dtype=float)
    n = len(elements)

    explicit_bonds = _read_bonds(block, asym_ids, comp_ids, atom_ids, index_of)
    return RawStructure(elements, coords, labels, np.ones(n), None, explicit_bonds, path)


def _read_bonds(block, asym_ids, comp_ids, atom_ids, index_of) -> list[tuple[int, int]]:
    """Expand per-species _chem_comp_bond rows onto every molecule copy (label_asym_id)."""
    bt = block.find("_chem_comp_bond.", ["comp_id", "atom_id_1", "atom_id_2"])
    if not len(bt):
        return []
    # bonds[comp_id] = list of (atom_id_1, atom_id_2)
    by_comp: dict[str, list[tuple[str, str]]] = {}
    for row in bt:
        by_comp.setdefault(row[0], []).append((row[1], row[2]))
    # asym_id -> comp_id (each copy is a single component)
    asym_comp = dict(zip(asym_ids, comp_ids, strict=True))
    bonds: list[tuple[int, int]] = []
    for asym, comp in asym_comp.items():
        for a1, a2 in by_comp.get(comp, []):
            i = index_of.get((asym, a1))
            j = index_of.get((asym, a2))
            if i is not None and j is not None:
                bonds.append((i, j))
    return bonds
