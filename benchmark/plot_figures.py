"""Render the README figures from benchmark/results/*.csv into docs/figures/.

    uv run --extra viz python benchmark/plot_figures.py [--only agreement,throughput,shell,beyond]

Inputs (produced by oxtalign_bench.py / compack_bench.py / deformation_sweep.py / csp_demo.py):
  oxtalign_timing_n15.csv, compack_timing_n15.csv, oxtalign_scaling_n15.csv, compack_scaling_n15.csv,
  oxtalign_shell_sweep.csv, compack_shell_sweep.csv, sweep_pairs.csv, deformation_sweep.csv, csp_demo_matrix.csv
A figure is skipped (with a note) when its inputs are missing.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "benchmark" / "results"
OUT = REPO / "docs" / "figures"

# Palette (validated light-surface set): series 1 = oxtalign, series 2 = COMPACK; grays for agreement/ink.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
MARK = dict(markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 10, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200, "legend.frameon": True, "legend.framealpha": 1.0,
    "legend.facecolor": SURFACE, "legend.edgecolor": "none", "legend.fontsize": 9,
    "text.color": INK,
})


def _rows(name):
    p = RES / name
    return list(csv.DictReader(open(p))) if p.exists() else None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / name)


def _full(n, rmsd, size=15):
    return n == size and not math.isnan(rmsd) and rmsd < 1.0


# ----------------------------------------------------------------------------- 1. agreement
def fig_agreement():
    ours = _rows("oxtalign_timing_n15.csv")
    cc = _rows("compack_timing_n15.csv")
    if ours is None or cc is None:
        return print("agreement: missing timing CSVs")
    curated = {(r["a_path"], r["b_path"]) for r in csv.DictReader(open(RES / "compack_reference.csv"))
               if r["source"] == "curated"}
    cc = {(r["a"], r["b"]): r for r in cc}
    sets = {"Curated polymorph\nfamilies": [], "Random CSD\nfamilies": []}
    both = []
    for r in ours:
        key = (r["a"], r["b"])
        c = cc[key]
        o_full = _full(int(r["nmatched"]), _f(r["rmsd"]))
        c_full = _full(int(c["nmatched"]), _f(c["rmsd"]))
        sets["Curated polymorph\nfamilies" if key in curated else "Random CSD\nfamilies"].append(
            (o_full, c_full, bool(c["error"])))
        if o_full and c_full:
            both.append((_f(c["rmsd"]), _f(r["rmsd"])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.45})
    x, y = np.array(both).T
    # An identical redetermination overlays to machine precision (~1e-15), which would stretch the
    # log axis over fifteen empty decades. Below 1e-4 A the engines simply agree the pair is the
    # same structure, so those points are dropped from the panel rather than plotted.
    floor = 1e-4
    keep = (x >= floor) & (y >= floor)
    dropped = int((~keep).sum())
    x, y = x[keep], y[keep]
    lim = (floor * 0.7, max(x.max(), y.max()) * 1.4)
    ax1.plot(lim, lim, color=AXIS, lw=1, zorder=1)
    # 1,200+ points sit almost on top of each other along the diagonal, where a plain scatter shows
    # only the handful of outliers. Bin them instead, so the panel reads as "nearly all pairs agree"
    # with the disagreements still visible as single-count cells.
    hb = ax1.hexbin(x, y, xscale="log", yscale="log", gridsize=34, bins="log", mincnt=1,
                    extent=(np.log10(lim[0]), np.log10(lim[1]), np.log10(lim[0]), np.log10(lim[1])),
                    cmap=LinearSegmentedColormap.from_list("agree", ["#dce9fa"] + BLUE_RAMP[2:]),
                    linewidths=0.2, edgecolors=SURFACE, zorder=3)
    cb = fig.colorbar(hb, ax=ax1, pad=0.02, fraction=0.046)
    cb.set_label("pairs per cell", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.outline.set_visible(False)
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_xlabel("COMPACK RMSD$_{15}$ (Å)")
    ax1.set_ylabel("OXtalign RMSD$_{15}$ (Å)")
    r = np.corrcoef(np.log(x), np.log(y))[0, 1]
    ax1.set_title("RMSD$_{15}$, OXtalign vs COMPACK")
    note = f"\n({dropped} identical pairs omitted)" if dropped else ""
    ax1.text(0.03, 0.97, f"{len(x)} pairs{note}\nPearson r (log) = {r:.3f}\n"
             f"mean |Δ| = {np.mean(np.abs(x - y)):.3f} Å",
             transform=ax1.transAxes, va="top", ha="left", color=INK2, fontsize=9)

    labels = list(sets)
    cats = [("Both match", MUTED), ("Neither matches", AXIS), ("OXtalign only", BLUE), ("COMPACK only", ORANGE)]
    for i, name in enumerate(labels):
        v = sets[name]
        n = len(v)
        counts = [sum(o and c for o, c, _ in v), sum((not o) and (not c) for o, c, _ in v),
                  sum(o and not c for o, c, _ in v), sum(c and not o for o, c, _ in v)]
        left = 0.0
        for (label, color), cnt in zip(cats, counts, strict=True):
            w = 100.0 * cnt / n
            ax2.barh(i, w, left=left, height=0.3, color=color, edgecolor=SURFACE, linewidth=2,
                     label=label if i == 0 else None)
            if w >= 7:
                ax2.text(left + w / 2, i, str(cnt), ha="center", va="center", fontsize=9,
                         color=SURFACE if color in (BLUE, ORANGE, MUTED) else INK)
            left += w
        errs = sum(e for _, _, e in v)
        agree = counts[0] + counts[1]
        ax2.text(101, i, f"agree {100 * agree / n:.0f}%\n(n = {n}; COMPACK\nfailed on {errs})", va="center",
                 ha="left", fontsize=8.5, color=INK2)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_ylim(1.7, -0.7)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("share of pairs (%)")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Match decisions")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, handlelength=1.2, columnspacing=1.4)
    _save(fig, "agreement_with_compack.png")


# ----------------------------------------------------------------------------- 2. throughput
def fig_throughput():
    ours = _rows("oxtalign_scaling_n15.csv")
    cc = _rows("compack_scaling_n15.csv")
    cc_t = _rows("compack_timing_n15.csv")
    if ours is None or cc is None or cc_t is None:
        return print("throughput: missing scaling CSVs")
    ow = [int(r["workers"]) for r in ours if int(r["workers"]) <= 64]
    oy = [float(r["pairs_per_s"]) for r in ours if int(r["workers"]) <= 64]
    cw = [1] + [int(r["workers"]) for r in cc]
    cy = [len(cc_t) / sum(_f(r["t"]) for r in cc_t)] + [float(r["pairs_per_s"]) for r in cc]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ideal = np.array([1, max(ow)])
    ax.plot(ideal, oy[0] * ideal, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=1, label="linear scaling")
    ax.plot(ideal, cy[0] * ideal, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.plot(ow, oy, "-o", color=BLUE, lw=2, zorder=3, label="OXtalign", **MARK)
    ax.plot(cw, cy, "-o", color=ORANGE, lw=2, zorder=3, label="COMPACK", **MARK)
    ax.text(ow[-1] * 1.12, oy[-1], f"{oy[-1]:,.0f} pairs/s", va="center", fontsize=9, color=INK2)
    ax.text(cw[-1] * 1.12, cy[-1], f"{cy[-1]:,.0f} pairs/s", va="center", fontsize=9, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ow)
    ax.set_xticklabels([str(w) for w in ow])
    ax.set_xlim(0.85, max(ow) * 1.9)
    ax.set_xlabel("worker processes")
    ax.set_ylabel("comparisons per second")
    ax.set_title("Throughput vs worker processes")
    ax.legend(loc="upper left")
    _save(fig, "throughput_vs_cores.png")


# ----------------------------------------------------------------------------- 3. shell size
def fig_shell():
    ours = _rows("oxtalign_shell_sweep.csv")
    cc = _rows("compack_shell_sweep.csv")
    if ours is None or cc is None:
        return print("shell: missing sweep CSVs")
    npairs = len({(r["a"], r["b"]) for r in ours})

    def by_shell(rows):
        out = {}
        for r in rows:
            out.setdefault(int(r["shell"]), []).append(r)
        return dict(sorted(out.items()))

    # p10-p90 rather than the IQR: COMPACK's per-pair cost is bimodal -- fast on most pairs,
    # catastrophic on a minority -- and an IQR band hides the tail that decides what a batch pays.
    def stats(rows):
        ns, med, lo, hi, done = [], [], [], [], []
        for n, rs in rows.items():
            ok = [r for r in rs if not r.get("error")]
            t = np.array([_f(r["t"]) for r in ok]) if ok else np.array([np.nan])
            ns.append(n)
            med.append(np.nanmedian(t))
            lo.append(np.nanpercentile(t, 10))
            hi.append(np.nanpercentile(t, 90))
            done.append(len(ok) / len(rs))
        return np.array(ns), np.array(med), np.array(lo), np.array(hi), np.array(done)

    o, c = by_shell(ours), by_shell(cc)
    on, om, ol, oh, _ = stats(o)
    cn, cm, cl, ch, cdone = stats(c)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"wspace": 0.32})

    # (a) oxtalign alone, linear axes: a fixed seeding cost plus a per-molecule cost
    # Fit the linear regime only. Past n ~ 50 the cost turns superlinear, because both the shell and
    # the candidate-image set it is matched against grow with n; extending the dashed line over the
    # whole range shows that departure rather than averaging it away.
    LINEAR_TO = 50
    fit_mask = (on > 0) & (on <= LINEAR_TO)
    slope, intercept = np.polyfit(on[fit_mask], 1000 * om[fit_mask], 1)
    xs = np.array([0, on.max()])
    ax1.plot(xs, intercept + slope * xs, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=1)
    ax1.fill_between(on, 1000 * ol, 1000 * oh, color=BLUE, alpha=0.10, lw=0)
    ax1.plot(on, 1000 * om, "-o", color=BLUE, lw=2, zorder=3, **MARK)
    ax1.set_xlim(0, on.max() * 1.05)
    ax1.set_ylim(0, 1000 * oh.max() * 1.1)
    ax1.set_xlabel("shell size n (molecules)")
    ax1.set_ylabel("OXtalign time per comparison (ms), median and p10–p90")
    ax1.set_title(f"OXtalign cost: {intercept:.1f} ms + {slope:.2f} ms × n  (n ≤ {LINEAR_TO})")
    ax1.text(0.04, 0.96, f"{npairs} pairs; linear fit R² = "
             f"{1 - np.sum((1000 * om[fit_mask] - (intercept + slope * on[fit_mask])) ** 2) / np.sum((1000 * om[fit_mask] - np.mean(1000 * om[fit_mask])) ** 2):.3f}"
             f" over n ≤ {LINEAR_TO};\nsuperlinear beyond",
             transform=ax1.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)

    # (b) both engines on COMPACK's range, log y
    keep = on <= cn.max()
    ax2.fill_between(on[keep], ol[keep], oh[keep], color=BLUE, alpha=0.10, lw=0)
    ax2.plot(on[keep], om[keep], "-o", color=BLUE, lw=2, label="OXtalign", zorder=3, **MARK)
    ax2.fill_between(cn, cl, ch, color=ORANGE, alpha=0.10, lw=0)
    ax2.plot(cn, cm, "-o", color=ORANGE, lw=2, label="COMPACK", zorder=3, **MARK)
    for n, m, d in zip(cn, cm, cdone, strict=True):
        if d < 1.0:
            ax2.annotate(f"{round((1 - d) * len(c[n]))} of {len(c[n])} failed", (n, m), textcoords="offset points",
                         xytext=(-6, 8), ha="right", fontsize=8, color=INK2)
    ax2.set_yscale("log")
    ax2.set_xlim(0, cn.max() * 1.08)
    ax2.set_xlabel("shell size n (molecules)")
    ax2.set_ylabel("time per comparison (s), median and p10–p90")
    ax2.set_title("Cost vs shell size, both engines")
    ax2.legend(loc="upper left")

    # (c) agreement with COMPACK by n
    ns, agree = [], []
    for n in cn:
        cc_ok = {(r["a"], r["b"]): r for r in c[n] if not r.get("error")}
        oo = {(r["a"], r["b"]): r for r in o[n]}
        same = [_full(int(oo[k]["nmatched"]), _f(oo[k]["rmsd"]), n) == _full(int(v["nmatched"]), _f(v["rmsd"]), n)
                for k, v in cc_ok.items()]
        ns.append(n)
        agree.append(100 * np.mean(same))
    ax3.plot(ns, agree, "-o", color=BLUE, lw=2, **MARK)
    ax3.set_ylim(0, 105)
    ax3.set_xlim(0, cn.max() * 1.08)
    ax3.set_xlabel("shell size n (molecules)")
    ax3.set_ylabel("pairs with the same match decision (%)")
    ax3.set_title("Agreement with COMPACK vs n")
    _save(fig, "shell_size_scaling.png")


# ----------------------------------------------------------------------------- 4. beyond experimental CIFs
def fig_beyond():
    mat = _rows("csp_demo_matrix.csv")
    dfm = _rows("deformation_sweep.csv")
    if mat is None or dfm is None:
        return print("beyond: missing csp/deformation CSVs")
    names = [r["sample"] for r in mat]
    M = np.array([[_f(r[n]) if r[n] != "inf" else np.inf for n in names] for r in mat])
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    D = np.where(np.isfinite(M), M, 3.0)
    order = leaves_list(linkage(squareform(D, checks=False), "average"))
    Mo = M[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.05, 1, 1], "wspace": 0.38})
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("blue_rev", BLUE_RAMP[::-1])
    shown = np.where(np.isfinite(Mo), Mo, np.nan)
    ax.set_facecolor(GRID)
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1.0, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i].replace("CAPRYL_seed", "s") for i in order], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, shrink=0.9)
    cb.set_label("RMSD$_{15}$ (Å), darker = more similar; grey = no match", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    n_sim = int(np.sum(np.triu(np.isfinite(M), 1)))
    ax.set_title(f"30 predicted clusters, all pairs\n({n_sim} of 435 similar)", pad=8)

    ax = axes[1]
    full = [(float(r["rms_displacement"]), _f(r["rmsd_15"])) for r in dfm if r["n_matched"] == "15" and 0 < float(r["rms_displacement"])]
    part = [(float(r["rms_displacement"]), _f(r["rmsd_15"])) for r in dfm if 8 <= int(r["n_matched"]) < 15]
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
    if part:
        px, py = np.array(part).T
        ax.plot(px, py, "o", markerfacecolor=SURFACE, markeredgecolor=BLUE, markersize=8, markeredgewidth=1.5,
                zorder=2, label="8–14 of 15 matched (RMSD over matched)")
    fx, fy = np.array(full).T
    ax.plot(fx, fy, "o", color=BLUE, zorder=3, label="all 15 matched", **MARK)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("applied RMS displacement per molecule (Å)")
    ax.set_ylabel("RMSD$_{15}$ over matched molecules (Å)")
    ax.set_title("RMSD$_{15}$ vs applied perturbation")
    ax.legend(loc="upper right")

    ax = axes[2]
    bins = np.arange(0.0, 2.01, 0.2)
    xs, share = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        rows = [r for r in dfm if lo < float(r["rms_displacement"]) <= hi]
        if rows:
            xs.append(0.5 * (lo + hi))
            share.append(100 * np.mean([r["n_matched"] == "15" for r in rows]))
    ax.plot(xs, share, "-o", color=BLUE, lw=2, **MARK)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 2.05)
    ax.set_xlabel("applied RMS displacement per molecule (Å)")
    ax.set_ylabel("variants with all 15 molecules matched (%)")
    ax.set_title("Full matches vs applied perturbation")
    _save(fig, "beyond_experimental_cifs.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="agreement,throughput,shell,beyond")
    args = ap.parse_args()
    for name in args.only.split(","):
        {"agreement": fig_agreement, "throughput": fig_throughput, "shell": fig_shell, "beyond": fig_beyond}[name]()


if __name__ == "__main__":
    main()
