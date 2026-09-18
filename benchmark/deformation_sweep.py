"""Response to controlled perturbation: rigidly displace every molecule of a crystal (random rotation about
its centroid + random translation), scaled so the heavy-atom RMS displacement hits a target d, and compare
the deformed P1 cell back to the original. RMSD_15 should track d while the packing is recognisable, and
n_matched should fall once it is not.

    uv run python benchmark/deformation_sweep.py            # -> benchmark/results/deformation_sweep.csv
"""
import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

import oxtalign

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data"
OUT = REPO / "benchmark" / "results" / "deformation_sweep.csv"
SOURCES = ["experimental/CAPRYL.cif", "experimental/HOLSUX/HOLSUX.cif", "experimental/ROY/QAXMEH24.cif",
           "experimental/NOFJEX/NOFJEX.cif", "experimental/OBEQIX.cif",
           "experimental/galunisertib/Galunisertib_FORM_I.cif", "csd_families/YUFZOQ/YUFZOQ01.cif",
           "csd_families/QIHBEO/QIHBEO.cif", "csd_families/TOJHIK/TOJHIK.cif", "csd_families/ACEMID/ACEMID.cif"]
TARGETS = [0.0] + [(k + 0.5) * 0.2 for k in range(10) for _ in range(2)]      # 0 and 2 draws per 0.2 A bin


def _rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _deform(mols, scale, draws):
    out = []
    for m, (axis, angle, shift) in zip(mols, draws, strict=True):
        c = m.centroid
        out.append((m.coords - c) @ _rotation(axis, scale * angle).T + c + scale * shift)
    return out


def _rms(mols, new):
    d = np.concatenate([np.linalg.norm(n - m.coords, axis=1) for m, n in zip(mols, new, strict=True)])
    return float(np.sqrt(np.mean(d * d)))


def write_p1_cif(path, name, cell, elements, cart):
    """Write one full unit cell (all molecules explicit) as a P1 CIF, coordinates wrapped into [0, 1)."""
    frac = np.mod(cell.cart_to_frac(cart), 1.0)
    lines = [f"data_{name}", f"_cell_length_a {cell.a:.5f}", f"_cell_length_b {cell.b:.5f}",
             f"_cell_length_c {cell.c:.5f}", f"_cell_angle_alpha {cell.alpha:.4f}",
             f"_cell_angle_beta {cell.beta:.4f}", f"_cell_angle_gamma {cell.gamma:.4f}",
             "_symmetry_space_group_name_H-M 'P 1'", "_symmetry_Int_Tables_number 1",
             "loop_", "_symmetry_equiv_pos_as_xyz", "x,y,z",
             "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
             "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy"]
    for i, (e, f) in enumerate(zip(elements, frac, strict=True)):
        lines.append(f"{e}{i + 1} {e} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f} 1.0")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    rng = np.random.default_rng(2026)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for src in SOURCES:
            s = oxtalign.load(str(DATA / src))
            mols, cell, tag = s.molecules, s.cell, Path(src).stem
            elements = [e for m in mols for e in m.elements]
            ref = Path(tmp) / f"{tag}_ref.cif"
            write_p1_cif(ref, tag, cell, elements, np.vstack([m.coords for m in mols]))
            for j, d in enumerate(TARGETS):
                draws = [(rng.normal(size=3), abs(rng.normal()) * 0.35, rng.normal(size=3) * 0.7) for _ in mols]
                if d == 0:
                    new = [m.coords.copy() for m in mols]
                else:
                    lo, hi = 0.0, 8.0
                    for _ in range(40):                       # bisection on the perturbation scale
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if _rms(mols, _deform(mols, mid, draws)) < d else (lo, mid)
                    new = _deform(mols, 0.5 * (lo + hi), draws)
                var = Path(tmp) / f"{tag}_d{j:02d}.cif"
                write_p1_cif(var, f"{tag}_d{j}", cell, elements, np.vstack(new))
                try:
                    r = oxtalign.compare(str(ref), str(var), n=15)
                    rec = {"n_matched": r.n_matched, "rmsd_15": r.rmsd_n, "status": r.status}
                except ValueError as e:                      # molecules pushed into one bonded network
                    rec = {"n_matched": 0, "rmsd_15": float("nan"), "status": f"unassemblable: {e}"[:60]}
                rows.append({"source": tag, "z": len(mols), "heavy_atoms": len(elements), "target_d": d,
                             "rms_displacement": _rms(mols, new), **rec})
            print(f"{tag}: done", file=sys.stderr)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
