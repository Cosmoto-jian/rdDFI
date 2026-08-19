#!/usr/bin/env python3
"""
xyz_length_scan.py — 单蛋白 ρ(r) 对比图：不同 RC 截断半径（8/10/12/15 Å）。

固定 K=1000，读取 raw/ 目录下 4 个 rc_*.h5 文件，
对 xy 和 z 方向各出一张图，对比 4 条 RC 曲线。

直接从 HDF5 读取 bin_r / bin_rho 原始数据。
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── 参数 ──────────────────────────────────────────────────────────────────
RAW_DIR    = "src/rc_scan/process"
OUT_DIR    = "results/rc_cor_length_results"
S_DEFAULT  = 16
K_FIXED    = 1000
N_RES_MIN  = 375
N_RES_MAX  = 475
CATEGORIES = ["GPCR"]       # 只选取 GPCR（7TM）类别

RC_VALUES  = [8, 10, 12, 15]

RC_COLORS  = {8: "#457B9D", 10: "#2A9D8F", 12: "#F4A261", 15: "#E63946"}
RC_MARKERS = {8: "o", 10: "s", 12: "^", 15: "D"}
RC_LABELS  = {8:  r"$r_c = 8$ Å",
              10: r"$r_c = 10$ Å",
              12: r"$r_c = 12$ Å",
              15: r"$r_c = 15$ Å"}

C_NULL = "#999999"
MM = 1.0 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.handlelength": 1.4,
    "axes.spines.top": True, "axes.spines.right": True,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

DIR_LABELS = {
    "xy": r"$\phi_{xy}$",
    "z":  r"$\phi_{z}$",
}
DIR_TITLES = {
    "xy": "In-plane (xy)",
    "z":  "Normal (z)",
}


# ═══════════════════════════════════════════════════════════════════════════
#  数据提取：从 raw/ 目录读取 rc_*.h5 文件
# ═══════════════════════════════════════════════════════════════════════════
def load_rc_data(raw_dir, s, K, direction, rc_values):
    """从 raw/rc_*.h5（批量结构）读取分箱关联数据，按 RC 值分组。

    H5 结构: /{category}/{protein_id}/s{s}/K{K}/{direction}

    只取 N_residues ∈ [N_RES_MIN, N_RES_MAX] 的蛋白。

    返回:
      curve  : {rc: {r_key: [rho_values]}}
      counts : {rc: n_proteins}
    """
    curve  = {rc: defaultdict(list) for rc in rc_values}
    counts = {rc: 0 for rc in rc_values}

    for rc in rc_values:
        h5_path = raw_dir / f"rc_{rc}.h5"
        if not h5_path.exists():
            print(f"   ⚠ File not found: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as hf:
            for cat in sorted(hf.keys()):
                if cat not in CATEGORIES:
                    continue
                cat_grp = hf[cat]
                if not isinstance(cat_grp, h5py.Group):
                    continue
                for pid in sorted(cat_grp.keys()):
                    prot = cat_grp[pid]
                    if not isinstance(prot, h5py.Group):
                        continue
                    # 跳过非蛋白 group（如 s*/K* 目录）
                    if f"s{s}" not in prot:
                        continue

                    # N_residues 优先，缺失时用 N_beads 代替
                    N_res = int(prot.attrs.get("N_residues",
                              prot.attrs.get("N_beads", -1)))
                    if N_res < N_RES_MIN or N_res > N_RES_MAX:
                        continue

                    try:
                        dg = prot[f"s{s}"][f"K{K}"][direction]
                    except KeyError:
                        continue

                    r   = dg["bin_r"][:]
                    rho = dg["bin_rho"][:]
                    cnt_arr = dg["bin_count"][:] if "bin_count" in dg else np.ones(len(r))

                    valid = np.isfinite(r) & np.isfinite(rho) & (cnt_arr >= 2)
                    r_v, rho_v = r[valid], rho[valid]

                    if len(r_v) == 0:
                        continue

                    counts[rc] += 1
                    for ri, rhoi in zip(r_v, rho_v):
                        r_key = round(ri, 1)
                        curve[rc][r_key].append(rhoi)

    return curve, counts


# ═══════════════════════════════════════════════════════════════════════════
#  绘图：单方向 RC 对比（K=1000 固定）
# ═══════════════════════════════════════════════════════════════════════════
def plot_direction(curve, counts, direction, s, out_dir, ylim=None, show_title=True):
    """一个方向的 ρ(r) RC 值对比均值曲线（4 条曲线，K=1000 固定）。"""
    total = max(counts.values())
    if total == 0:
        print(f"   ⚠ No data for '{direction}', skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(50 * MM, 36 * MM))

    all_r = set()
    for rc in RC_VALUES:
        if counts[rc] == 0:
            continue
        bins = sorted(curve[rc].keys())
        r_mean, rho_mean = [], []
        for r_key in bins:
            vals = np.array(curve[rc][r_key])
            n = len(vals)
            if n < 1:
                continue
            r_mean.append(r_key)
            rho_mean.append(vals.mean())

        r_mean = np.array(r_mean)
        rho_mean = np.array(rho_mean)

        ax.plot(r_mean, rho_mean, "-",
                marker=RC_MARKERS[rc],
                lw=1.4,
                color=RC_COLORS[rc],
                markersize=4.0,
                markeredgewidth=0.3,
                markeredgecolor="black",
                zorder=4,
                label=RC_LABELS[rc])

        all_r.update(r_mean.tolist())

    # ── 参考线 ──
    ax.axhline(0, color=C_NULL, ls=":", lw=0.5, zorder=0)

    # ── x 轴范围 ──
    r_vals = sorted(all_r)
    rmax = r_vals[-1] if r_vals else 85.0
    rmax = min(np.ceil(rmax / 10) * 10, 90)
    ax.set_xlim(0, rmax)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel(r"$r_{ij}$ (Å)")

    # ── y 轴范围（仅基于 x 轴可见区间） ──
    vis_vals = []
    for rc in RC_VALUES:
        if counts[rc] == 0:
            continue
        for r_key in sorted(curve[rc].keys()):
            if r_key > rmax:
                break
            vals = np.array(curve[rc][r_key])
            if len(vals) < 1:
                continue
            vis_vals.extend(vals.tolist())

    if vis_vals:
        y_lo = np.nanmin(vis_vals)
        y_hi = max(np.nanmax(vis_vals), 0.95)
    else:
        y_lo, y_hi = -0.3, 1.05
    pad = (y_hi - y_lo) * 0.06
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel(DIR_LABELS[direction], rotation=0, ha="right", va="center")
    if show_title:
        ax.set_title(f"{DIR_TITLES.get(direction, direction)}    "
                     f"(s = {s},  K = {K_FIXED})",
                     pad=4, fontsize=10)

    ax.legend(loc="upper right", fontsize=6, frameon=True,
              handlelength=1.8, markerscale=0.8,
              borderpad=0.3, labelspacing=0.3)

    fname = f"corr_{direction}_s{s}.png"
    fig.savefig(out_dir / fname, dpi=600)
    plt.close(fig)
    return fname


# ═══════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default=RAW_DIR,
                    help="Directory containing rc_*.h5 files")
    ap.add_argument("--out", type=str, default=OUT_DIR)
    ap.add_argument("--s", type=int, default=S_DEFAULT)
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 图 1 & 2：xy / z 方向 RC 值对比曲线 ──
    for direction in ["xy", "z"]:
        print(f"== {direction} (s={args.s}, K={K_FIXED}) ==")
        curve, counts = load_rc_data(
            raw_dir, args.s, K_FIXED, direction, RC_VALUES)

        for rc in RC_VALUES:
            n_pts = sum(len(v) for v in curve[rc].values())
            print(f"   {RC_LABELS[rc]:30s}: {counts[rc]:4d} files, "
                  f"{n_pts:7d} points")

        fname = plot_direction(curve, counts, direction, args.s, out_dir,
                               ylim=(-0.5, 1) if direction == "xy" else (-0.3, 1),
                               show_title=False)
        if fname:
            print(f"   -> {fname}")
        else:
            print(f"   ⚠ skipped (no data)")

    print(f"\n   Figures -> {out_dir}/")


if __name__ == "__main__":
    main()
