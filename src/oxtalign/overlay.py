"""Reconstruct a packing-similarity overlay for visualization (the "save outputs" flag).

This sits on top of the public comparison API. It re-runs ``compare`` to obtain the optimal
transform (B -> A), then reconstructs A's central-molecule coordination shell exactly the way the
matcher does (``_interior_order`` / ``_unique_conformers`` pick the primary central; then
``nearest_shell``), maps every B molecule image into A's frame, and pairs each A-shell molecule with
the nearest same-species mapped-B molecule. The result is a compact JSON describing both molecule
sets in A's coordinate frame, ready for a browser viewer.

Nothing here mutates the engine; it only imports the stable public helpers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .cluster import molecule_images, nearest_shell
from .compare import compare, load
from .matching import _interior_order, _primary_species, _unique_conformers
from .model import Structure


def _pick_central(struct) -> int:
    """The primary central molecule index in ``struct``, chosen exactly as the matcher does.

    The matcher anchors on the largest species (most atoms), orders its molecules from the
    structure centre outward, dedups conformers, and uses the first as the primary central. We
    reproduce that here so the reconstructed A-shell matches the one the comparison ran on.
    """
    primary = _primary_species(struct)
    ordered = _interior_order(struct, primary)
    uniq = _unique_conformers(struct.molecules, ordered)
    return uniq[0] if uniq else ordered[0]


def _atom_rmsd(a_coords: np.ndarray, a_elems, b_coords: np.ndarray, b_elems) -> float:
    """RMSD between two molecules using a nearest same-element atom assignment.

    Falls back to centroid distance when the atom multisets differ (e.g. different formulae or a
    partial overlay), so we always return a finite, comparable number.
    """
    a_coords = np.asarray(a_coords, dtype=float)
    b_coords = np.asarray(b_coords, dtype=float)
    if list(sorted(a_elems)) != list(sorted(b_elems)) or len(a_elems) == 0:
        ca = a_coords.mean(axis=0)
        cb = b_coords.mean(axis=0)
        return float(np.linalg.norm(ca - cb))

    a_elems = np.asarray(a_elems)
    b_elems = np.asarray(b_elems)
    cost = cdist(a_coords, b_coords)
    # Forbid cross-element assignment so the correspondence is chemically sensible.
    cost[a_elems[:, None] != b_elems[None, :]] = 1e9
    rows, cols = linear_sum_assignment(cost)
    if np.any(cost[rows, cols] >= 1e9):  # element multisets matched in total but not pairwise
        ca = a_coords.mean(axis=0)
        cb = b_coords.mean(axis=0)
        return float(np.linalg.norm(ca - cb))
    diff = a_coords[rows] - b_coords[cols]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _cell_frame(struct, central_idx: int):
    """The unit cell containing the central molecule, as (origin, a, b, c) cartesian vectors.

    Anchored at the cell the central molecule sits in so the box sits among the displayed shell.
    Returns None for cell-less inputs (predicted clusters).
    """
    cell = struct.cell
    if cell is None:
        return None
    lat = cell.lattice_vectors                       # rows a, b, c (cartesian)
    frac = cell.cart_to_frac(struct.molecules[central_idx].centroid)
    origin = np.floor(frac) @ lat
    return origin, lat[0], lat[1], lat[2]


def _cell_record(origin, a, b, c, decimals: int = 3) -> dict:
    def r(v):
        return np.round(np.asarray(v, dtype=float), decimals).tolist()
    return {"origin": r(origin), "a": r(a), "b": r(b), "c": r(c)}


def _mol_record(mol, role: str, shell_index: int, *, matched: bool,
                rmsd: float | None, coords: np.ndarray, decimals: int = 3) -> dict:
    return {
        "role": role,
        "shell_index": int(shell_index),
        "species": mol.species_key,
        "matched": bool(matched),
        "rmsd": None if rmsd is None else round(float(rmsd), 4),
        "elements": list(mol.elements),
        "coords": np.round(np.asarray(coords, dtype=float), decimals).tolist(),
        "bonds": [[int(i), int(j)] for (i, j) in mol.bonds],
    }


def build_overlay(a_cif: str | Structure, b_cif: str | Structure, *, n: int = 15, match_tol: float = 1.0,
                  **compare_kw) -> dict:
    """Compute the overlay payload (dict) without writing a file. Inputs are CIF paths or Structures."""
    def _prep(x):
        return x if isinstance(x, Structure) else load(
            x, include_hydrogens=compare_kw.get("include_hydrogens", False),
            drop_minor_occupancy=compare_kw.get("drop_minor_occupancy", True))

    A, B = _prep(a_cif), _prep(b_cif)

    # Align at the same tolerance used to flag matches, so a raised match_tol actually surfaces the
    # broader partial overlay (not just flag molecules against the tight default alignment).
    compare_kw.setdefault("mol_rmsd_tol", match_tol)
    result = compare(A, B, n=n, **compare_kw)

    # Anchor the reconstructed shell on the central molecule the match actually won on (Z'>1 /
    # multi-conformer cells can win on a non-first central); fall back to the matcher's first
    # candidate when the compare failed and recorded none.
    central = result.central_a if result.central_a >= 0 else _pick_central(A)
    shell = nearest_shell(A, central, n)

    rmsd_n = result.rmsd_n
    payload: dict = {
        "rmsd_n": None if (rmsd_n is None or np.isnan(rmsd_n)) else round(float(rmsd_n), 4),
        "n": int(n),
        "n_matched": int(result.n_matched),
        "status": result.status,
        "inverted": bool(result.inverted),
        "a_source": A.source,
        "b_source": B.source,
        "match_tol": float(match_tol),
        "molecules": [],
    }

    # Emit A's shell unconditionally so the user always sees something, even on a failed compare.
    a_records = [_mol_record(m, "A", i, matched=False, rmsd=None, coords=m.coords)
                 for i, m in enumerate(shell)]

    cell_a = _cell_frame(A, central)
    if cell_a is not None:
        payload["cell_a"] = _cell_record(*cell_a)

    if result.transform is None:
        payload["molecules"] = a_records
        payload["message"] = result.message
        return payload

    # Map every B molecule image into A's frame via the optimal transform.
    R = np.asarray(result.transform)[:3, :3]
    t = np.asarray(result.transform)[:3, 3]
    sign = -1 if result.inverted else 1

    # B's unit cell, anchored at B's central molecule and mapped into A's frame (origin as a point,
    # edges as directions) so the two lattices can be compared directly.
    cell_b = _cell_frame(B, _pick_central(B))
    if cell_b is not None:
        o, ea, eb, ec = cell_b
        payload["cell_b"] = _cell_record(_map(o, sign, R, t),
                                         (sign * ea) @ R.T, (sign * eb) @ R.T, (sign * ec) @ R.T)

    b_cent, b_base, b_off = molecule_images(B, n, contact=True)
    # A centroid maps like a point under the rigid+uniform transform (centroid of mapped = map of centroid).
    mapped_centroids = _map(b_cent, sign, R, t)
    b_species = np.array([B.molecules[i].species_key for i in b_base])

    b_records: list[dict] = []
    for ai, amol in enumerate(shell):
        a_centroid = amol.centroid
        # Candidate B images of the same species; for a partial (different-formula) match no exact
        # species exists, so fall back to all B images (the transform already aligns the main
        # component, so the nearest B image is the corresponding molecule).
        same = np.where(b_species == amol.species_key)[0]
        if len(same) == 0:
            same = np.arange(len(b_species))
        d = np.linalg.norm(mapped_centroids[same] - a_centroid, axis=1)
        pick = same[int(np.argmin(d))]
        base = int(b_base[pick])
        b_raw = B.molecules[base].coords + b_off[pick]
        b_mapped = _map(b_raw, sign, R, t)

        r = _atom_rmsd(amol.coords, amol.elements,
                       b_mapped, B.molecules[base].elements)
        matched = r < match_tol
        a_records[ai]["matched"] = matched
        a_records[ai]["rmsd"] = round(float(r), 4)

        b_records.append(_mol_record(
            B.molecules[base], "B", ai, matched=matched, rmsd=r, coords=b_mapped))

    payload["molecules"] = a_records + b_records
    return payload


def _map(coords: np.ndarray, sign: int, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map B-frame coords into A's frame: mapped = (sign*coords) @ R.T + t."""
    return (sign * np.asarray(coords, dtype=float)) @ R.T + t


def save_overlay(a_cif: str, b_cif: str, out_json: str, n: int = 15,
                 match_tol: float = 1.0, **compare_kw) -> dict:
    """Build the overlay payload and write it to ``out_json``. Returns the payload dict."""
    payload = build_overlay(a_cif, b_cif, n=n, match_tol=match_tol, **compare_kw)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return payload


# The interactive viewer is a single self-contained HTML template (no CDN/deps) that renders the
# overlay payload on a canvas. We inject the real payload between these markers in its <script>.
_VIEWER_TEMPLATE = Path(__file__).with_name("viewer_template.html")
_JSON_START, _JSON_END = "/*__OVERLAY_JSON__*/", "/*__END__*/"


def render_html(payload: dict) -> str:
    """Embed an overlay ``payload`` into the bundled viewer template -> a complete HTML document."""
    body = _VIEWER_TEMPLATE.read_text()
    i = body.index(_JSON_START) + len(_JSON_START)
    j = body.index(_JSON_END)
    data = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")  # neutralize literal </script>
    body = body[:i] + " " + data + " " + body[j:]
    return ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            + body + "\n</html>\n")


def save_html(a_cif: str, b_cif: str, out_html: str, *, n: int = 15,
              match_tol: float = 1.0, **compare_kw) -> dict:
    """Compute a match overlay and write a self-contained interactive 3D viewer to ``out_html``.

    Open the file in any browser: drag to rotate, scroll to zoom, toggle A / B / matched-only.
    ``compare_kw`` accepts the usual ``compare`` options (allow_inversion, allow_partial,
    include_hydrogens, ...). Returns the overlay payload dict.
    """
    payload = build_overlay(a_cif, b_cif, n=n, match_tol=match_tol, **compare_kw)
    Path(out_html).write_text(render_html(payload))
    return payload


def pdb_text(payload: dict) -> str:
    """The overlay as one PDB model: A's coordination shell is chain A, the overlaid B molecules chain B,
    one residue per molecule, B-factor = that molecule's overlay RMSD (colour by B-factor in a viewer),
    CONECT records from the perceived bonds. Opens in Mercury, ChimeraX, VESTA, PyMOL, ...
    """
    rmsd_n = payload["rmsd_n"]
    lines = [f"REMARK   1 oxtalign overlay: RMSD_{payload['n']} = {rmsd_n if rmsd_n is not None else 'nan'}  "
             f"matched {payload['n_matched']}/{payload['n']}  status {payload['status']}",
             f"REMARK   1 chain A = {payload['a_source']}",
             f"REMARK   1 chain B = {payload['b_source']} mapped into A's frame"]
    conect: list[str] = []
    serial = 1
    for mol in payload["molecules"]:
        chain, resseq = mol["role"], (mol["shell_index"] + 1) % 10000
        bfactor = min(float(mol["rmsd"]) if mol["rmsd"] is not None else 0.0, 999.99)
        start = serial
        for el, (x, y, z) in zip(mol["elements"], mol["coords"], strict=True):
            name = f"{el:>2s}  " if len(el) == 2 else f" {el:<3s}"
            lines.append(f"HETATM{serial % 100000:5d} {name} MOL {chain}{resseq:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{bfactor:6.2f}          {el:>2s}")
            serial += 1
        conect.extend(f"CONECT{(start + i) % 100000:5d}{(start + j) % 100000:5d}" for i, j in mol["bonds"])
    return "\n".join(lines + conect + ["END"]) + "\n"


def save_pdb(a_cif, b_cif, out_pdb: str, *, n: int = 15, match_tol: float = 1.0, **compare_kw) -> dict:
    """Compute a match overlay and write it as a two-chain PDB (see :func:`pdb_text`)."""
    payload = build_overlay(a_cif, b_cif, n=n, match_tol=match_tol, **compare_kw)
    Path(out_pdb).write_text(pdb_text(payload))
    return payload
