#!/usr/bin/env python3
"""
plot_N_distribution.py — N_res 窗口 ξ/L 概率分布对比图。

固定 s=16, K=1000，针对不同 N_res 窗口分别绘制 KDE 曲线：
  [275, 375], [325, 425], [375, 475], [425, 525]

两组图：GPCR 分类 + 全部分类，各含 ξ_xy/L_xy 和 ξ_z/L_z 两张图。

Usage:
  python plot_N_distribution.py [--bins 25]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde

# ── 固定参数 ──────────────────────────────────────────────────────────────────
H5_PATH = "data/process/tm_plddt70.h5"
CSV_PATH = "data/process/xi_master_table_100_800.csv"
OUT_DIR = "results/S16_p_distribution"

K_FIXED = 1000
S_FIXED = 16
BINS_DEFAULT = 25
N_RES_WINDOWS = [
    (275, 375),
    (325, 425),
    (375, 475),
    (425, 525),
]
ALL_CATEGORIES = ["GPCR", "large", "medium", "single_pass"]

# ── N_res 窗口视觉配置 ───────────────────────────────────────────────────────
WINDOW_COLORS = {
    (275, 375): "#0072B2",
    (325, 425): "#009E73",
    (375, 475): "#D55E00",
    (425, 525): "#CC79A7",
}   # Wong 4-color 色盲友好 / Nature 风格

# ── matplotlib rc ────────────────────────────────────────────────────────────
MM = 1.0 / 25.4

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": True,
        "legend.handlelength": 1.4,
        "lines.linewidth": 1.2,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
#  从 H5 读取蛋白元数据
# ═══════════════════════════════════════════════════════════════════════════════
def load_protein_meta(h5_path: str, category: str) -> pd.DataFrame:
    """读取 H5 中指定 category 下所有蛋白的 attrs。"""
    rows = []
    with h5py.File(h5_path, "r") as f:
        if category not in f:
            raise KeyError(f"Category '{category}' not found in H5 file.")
        for pid in f[category]:
            obj = f[category][pid]
            if not isinstance(obj, h5py.Group):
                continue
            attrs = dict(obj.attrs)
            if "N_residues" not in attrs:
                continue
            rows.append(
                {
                    "pid": pid,
                    "category": category,
                    "N_residues": int(attrs["N_residues"]),
                    "L_xy": float(attrs["L_xy"]),
                    "L_z": float(attrs["L_z"]),
                }
            )
    return pd.DataFrame(rows)


def load_all_protein_meta(h5_path: str, categories: list) -> pd.DataFrame:
    """读取 H5 中所有 categories 的蛋白 attrs。"""
    frames = []
    for cat in categories:
        try:
            df = load_protein_meta(h5_path, cat)
            if len(df) > 0:
                frames.append(df)
        except KeyError:
            print(f"  WARNING: category '{cat}' not found, skipping.")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  从 CSV 读取 ξ，按 N_res 窗口返回 η 数组
# ═══════════════════════════════════════════════════════════════════════════════
def load_eta_by_window(
    csv_path: str,
    pids: set,
    windows: list,
    s_val: int,
    K: int,
    category: str = None,
) -> dict:
    """返回 {(direction, (lo, hi)): eta array}。"""
    df = pd.read_csv(csv_path)
    mask = (
        (df["pid"].isin(pids))
        & (df["s"] == s_val)
        & (df["K"] == K)
    )
    if category is not None:
        mask = mask & (df["category"] == category)
    sub = df[mask].copy()

    eta_dict = {}
    for direction in ["xy", "z"]:
        sub_d = sub[sub["direction"] == direction]
        L_col = f"L_{direction}"
        for lo, hi in windows:
            win = sub_d[sub_d["N_res"].between(lo, hi)]
            if len(win) == 0:
                eta_dict[(direction, (lo, hi))] = np.array([])
                continue
            eta = win["xi_zero"].values / win[L_col].values
            eta_dict[(direction, (lo, hi))] = eta

    return eta_dict


# ═══════════════════════════════════════════════════════════════════════════════
#  画单张 N_res 窗口对比概率分布图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_N_distribution(
    eta_dict: dict,
    direction: str,
    windows: list,
    out_dir: Path,
    bins: int,
    title_tag: str = "GPCR",
    fname_tag: str = "gpcr",
    xlabel: str = None,
    hide_legend: bool = False,
) -> str:
    """画一张 η = ξ/L 的 N_res 窗口对比概率密度图。"""
    fig, ax = plt.subplots(figsize=(50 * MM, 36 * MM))

    phi_label = {"xy": r"\phi_{xy}", "z": r"\phi_z"}

    for lo, hi in windows:
        eta_raw = eta_dict.get((direction, (lo, hi)), np.array([]))
        eta = eta_raw[np.isfinite(eta_raw)]
        if len(eta) == 0:
            print(f"  WARNING: direction={direction}, N∈[{lo},{hi}]: no valid data")
            continue

        color = WINDOW_COLORS[(lo, hi)]
        mean_val = np.mean(eta)

        # KDE 曲线
        kde = gaussian_kde(eta)
        x_grid = np.linspace(eta.min() * 0.85, eta.max() * 1.15, 300)
        ax.plot(
            x_grid,
            kde(x_grid),
            color=color,
            linestyle="-",
            lw=1.4,
            label=rf"${lo} < N \leq {hi}$",
        )

        # 标注均值虚线
        ax.axvline(
            mean_val,
            color=color,
            linestyle=":",
            lw=1.2,
            alpha=0.5,
        )

    # 轴标签
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f"${phi_label[direction]}$")
    ax.set_ylabel(r"$p$", rotation=0, labelpad=10)

    # 刻度
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    # x 轴刻度
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    # 标题
    dir_name = {"xy": "in-plane", "z": "normal"}
    ax.set_title(
        rf"{title_tag}  ${phi_label[direction]}$  ($s={S_FIXED}$, K = 1000, {dir_name[direction]})",
        pad=4,
        fontsize=10,
    )

    if not hide_legend:
        ax.legend(loc="upper right", fontsize=5, frameon=True)

    # 保存
    fname_stem = f"eta_{direction}_N_windows_{fname_tag}"
    fpath = out_dir / f"{fname_stem}.png"
    fig.savefig(fpath, dpi=600)
    print(f"  -> {fpath}")

    plt.close(fig)
    return fname_stem


# ═══════════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Plot ξ/L probability distribution across N_res windows."
    )
    ap.add_argument("--bins", type=int, default=BINS_DEFAULT)
    ap.add_argument("--h5", type=str, default=H5_PATH)
    ap.add_argument("--csv", type=str, default=CSV_PATH)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parameters: s={S_FIXED}, K={K_FIXED}")
    print(f"N_res windows: {N_RES_WINDOWS}")

    # ══════════════════════════════════════════════════════════
    #  Part A: GPCR
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("  Part A: GPCR only")
    print("=" * 50)

    meta_gpcr = load_protein_meta(args.h5, "GPCR")
    print(f"\n[1] GPCR total proteins: {len(meta_gpcr)}")

    # 对所有 N_res 窗口取并集 pids（CSV 中再按窗口筛选）
    # 取宽范围并集以涵盖所有窗口
    all_lo = min(w[0] for w in N_RES_WINDOWS)
    all_hi = max(w[1] for w in N_RES_WINDOWS)
    mask = meta_gpcr["N_residues"].between(all_lo, all_hi)
    pids_gpcr = set(meta_gpcr[mask]["pid"])
    print(f"   N_res ∈ [{all_lo}, {all_hi}] union: {len(pids_gpcr)} proteins")

    print(f"\n[2] Loading ξ (s={S_FIXED}, K={K_FIXED}) [GPCR]")
    eta_gpcr = load_eta_by_window(
        args.csv, pids_gpcr, N_RES_WINDOWS, S_FIXED, K_FIXED, category="GPCR"
    )
    for direction in ["xy", "z"]:
        for lo, hi in N_RES_WINDOWS:
            eta = eta_gpcr.get((direction, (lo, hi)), np.array([]))
            valid = eta[np.isfinite(eta)]
            n_bad = len(eta) - len(valid)
            if len(valid) > 0:
                print(
                    f"   dir={direction}, N∈[{lo},{hi}]: "
                    f"n={len(valid)}, mean={np.mean(valid):.3f}, std={np.std(valid):.3f}"
                    + (f"  (dropped {n_bad} NaN)" if n_bad > 0 else "")
                )

    print(f"\n[3] Plotting GPCR")
    plot_N_distribution(
        eta_gpcr, "xy", N_RES_WINDOWS, out_dir, args.bins,
        title_tag="GPCR", fname_tag="gpcr",
        xlabel=r"$\rho_{xy}^*$", hide_legend=True,
    )
    plot_N_distribution(
        eta_gpcr, "z", N_RES_WINDOWS, out_dir, args.bins,
        title_tag="GPCR", fname_tag="gpcr",
        xlabel=r"$\rho_{z}^*$",
    )

    # ══════════════════════════════════════════════════════════
    #  Part B: All categories
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("  Part B: All categories")
    print("=" * 50)

    meta_all = load_all_protein_meta(args.h5, ALL_CATEGORIES)
    print(f"\n[4] All categories total: {len(meta_all)}")
    mask_all = meta_all["N_residues"].between(all_lo, all_hi)
    pids_all = set(meta_all[mask_all]["pid"])
    print(f"   N_res ∈ [{all_lo}, {all_hi}] union: {len(pids_all)} proteins")

    print(f"\n[5] Loading ξ (s={S_FIXED}, K={K_FIXED}) [All categories]")
    eta_all = load_eta_by_window(
        args.csv, pids_all, N_RES_WINDOWS, S_FIXED, K_FIXED, category=None
    )
    for direction in ["xy", "z"]:
        for lo, hi in N_RES_WINDOWS:
            eta = eta_all.get((direction, (lo, hi)), np.array([]))
            valid = eta[np.isfinite(eta)]
            n_bad = len(eta) - len(valid)
            if len(valid) > 0:
                print(
                    f"   dir={direction}, N∈[{lo},{hi}]: "
                    f"n={len(valid)}, mean={np.mean(valid):.3f}, std={np.std(valid):.3f}"
                    + (f"  (dropped {n_bad} NaN)" if n_bad > 0 else "")
                )

    print(f"\n[6] Plotting All categories")
    plot_N_distribution(
        eta_all, "xy", N_RES_WINDOWS, out_dir, args.bins,
        title_tag="All categories", fname_tag="all",
        xlabel=r"$\rho_{xy}^*$", hide_legend=True,
    )
    plot_N_distribution(
        eta_all, "z", N_RES_WINDOWS, out_dir, args.bins,
        title_tag="All categories", fname_tag="all",
        xlabel=r"$\rho_{z}^*$",
    )

    print(f"\n   Done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
