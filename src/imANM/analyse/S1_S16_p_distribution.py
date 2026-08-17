#!/usr/bin/env python3
"""
plot_xi_distribution.py — 绘制 GPCR 蛋白 ξ/L 概率分布图。

从 tm_plddt70.h5 读取蛋白元数据（L_xy, L_z, N_residues），
从 xi_master_table_100_800.csv 读取关联长度 ξ，
筛选 N_res ∈ [375, 475] 的 GPCR 蛋白，
画出 ξ_xy/L_xy 和 ξ_z/L_z 的概率分布，对比 s=1 和 s=16。

Usage:
  python plot_xi_distribution.py [--K 200] [--bins 30]
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
from scipy.stats import gaussian_kde, rankdata

# ── 路径默认值 ────────────────────────────────────────────────────────────────
H5_PATH = "data/process/tm_plddt70.h5"
CSV_PATH = "data/process/xi_master_table_100_800.csv"
OUT_DIR = "results/S1_S16_p_distribution"

K_DEFAULT = 1000
BINS_DEFAULT = 25
S_PAIR = (1, 16)
N_RES_LO, N_RES_HI = 375, 475
CATEGORY = "GPCR"

# ── matplotlib rc ────────────────────────────────────────────────────────────
MM = 1.0 / 25.4

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "Helvetica",
            "DejaVu Sans",
        ],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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

S_COLORS = {1: "#0072B2", 16: "#D55E00"}      # Wong/Nature 色盲友好


# ═══════════════════════════════════════════════════════════════════════════════
#  从 H5 读取蛋白元数据
# ═══════════════════════════════════════════════════════════════════════════════
def load_protein_meta(h5_path: str, category: str) -> pd.DataFrame:
    """读取 H5 中指定 category 下所有蛋白的 attrs，返回 DataFrame。"""
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
    """读取 H5 中所有 categories 的蛋白 attrs，返回合并 DataFrame。"""
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
#  从 CSV 读取 ξ 并计算 η = ξ / L
# ═══════════════════════════════════════════════════════════════════════════════
def load_eta(
    csv_path: str,
    pids: set,
    s_values: tuple,
    K: int,
    category: str = None,
) -> dict:
    """返回 {(direction, s): array of eta values}。

    category=None 表示不筛选分类，取所有类别。
    """
    df = pd.read_csv(csv_path)
    mask = (
        (df["pid"].isin(pids))
        & (df["s"].isin(s_values))
        & (df["K"] == K)
        & (df["N_res"].between(N_RES_LO, N_RES_HI))
    )
    if category is not None:
        mask = mask & (df["category"] == category)
    sub = df[mask].copy()

    eta_dict = {}
    for s_val in s_values:
        for direction in ["xy", "z"]:
            sub_d = sub[(sub["s"] == s_val) & (sub["direction"] == direction)]
            if len(sub_d) == 0:
                eta_dict[(direction, s_val)] = np.array([])
                continue
            L_col = f"L_{direction}"
            eta = sub_d["xi_zero"].values / sub_d[L_col].values
            eta_dict[(direction, s_val)] = eta

    return eta_dict


def load_eta_wide(csv_path, pids, s_values, K, category=None):
    """index=pid, 列=(direction, s) 的宽表，自动按 pid 对齐配对。"""
    df = pd.read_csv(csv_path)
    m = (
        df["pid"].isin(pids) & df["s"].isin(s_values) & (df["K"] == K)
        & df["N_res"].between(N_RES_LO, N_RES_HI)
    )
    if category is not None:
        m &= df["category"] == category
    sub = df[m].copy()
    sub["L_dir"] = np.where(sub["direction"] == "xy", sub["L_xy"], sub["L_z"])
    sub["eta"] = sub["xi_zero"] / sub["L_dir"]
    return sub.pivot_table(index="pid", columns=["direction", "s"], values="eta")


def paired_rbc(x1, x2, n_boot=5000, seed=0):
    """配对 rank-biserial 效应量 + bootstrap 95% CI。x1, x2 已对齐等长。

    d = x1 - x2：r_rb > 0 表示 x1 倾向于大于 x2。
    """
    d = np.asarray(x1, float) - np.asarray(x2, float)
    d = d[np.isfinite(d)]

    def rbc(diff):
        diff = diff[diff != 0]
        if len(diff) == 0:
            return np.nan
        ranks = rankdata(np.abs(diff))
        pos, neg = ranks[diff > 0].sum(), ranks[diff < 0].sum()
        return (pos - neg) / (pos + neg)

    r = rbc(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = np.array([rbc(d[i]) for i in idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return r, lo, hi


# ═══════════════════════════════════════════════════════════════════════════════
#  画单张概率分布图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_distribution(
    eta_dict: dict,
    direction: str,
    s_values: tuple,
    out_dir: Path,
    K: int,
    bins: int,
    wide=None,
    title_tag: str = "GPCR",
    fname_tag: str = "gpcr",
    xlabel: str = None,
    legend_no_mean: bool = False,
    hide_legend: bool = False,
    swap_rrb: bool = False,
    y_major_locator: float = None,
    ylim: tuple = None,
    show_title: bool = True,
) -> str:
    """画一张 η = ξ/L 的概率密度图（KDE + 直方图叠加），左上角标注配对效应量。"""
    fig, ax = plt.subplots(figsize=(50 * MM, 36 * MM))

    phi_label = {"xy": r"\phi_{xy}", "z": r"\phi_z"}

    for s_val in s_values:
        eta_raw = eta_dict.get((direction, s_val), np.array([]))
        # 剔除 NaN
        eta = eta_raw[np.isfinite(eta_raw)]
        if len(eta) == 0:
            print(f"  WARNING: direction={direction}, s={s_val}: no valid data")
            continue

        # 直方图 (density=True → 概率密度)
        ax.hist(
            eta,
            bins=bins,
            density=True,
            alpha=0.25,
            color=S_COLORS[s_val],
            edgecolor="white",
            linewidth=0.3,
        )

        mean_val = np.mean(eta)

        # KDE 平滑曲线
        kde = gaussian_kde(eta)
        x_grid = np.linspace(eta.min() * 0.85, eta.max() * 1.15, 300)
        if legend_no_mean:
            label_str = f"$s = {s_val}$"
        else:
            label_str = (
                f"$s = 1$     Mean = {mean_val:.2f}"
                if s_val == 1 else
                f"$s = 16$    Mean = {mean_val:.2f}"
            )
        ax.plot(
            x_grid,
            kde(x_grid),
            color=S_COLORS[s_val],
            linestyle="-",
            lw=1.4,
            label=label_str if not hide_legend else None,
        )

        # 标注均值虚线
        ax.axvline(
            mean_val,
            color=S_COLORS[s_val],
            linestyle=":",
            lw=1.2,
            alpha=0.7,
        )

    # ── 配对效应量：rank-biserial + 95% CI ──
    if wide is not None and (direction, 1) in wide.columns and (direction, 16) in wide.columns:
        pair = wide[[(direction, 1), (direction, 16)]].dropna()
        if len(pair) >= 3:
            x1 = pair[(direction, 1)].values
            x16 = pair[(direction, 16)].values
            # s=16 相对 s=1：r_rb > 0 表示 η(s=16) 倾向于大于 η(s=1)
            r, lo, hi = paired_rbc(x16, x1)
            rrb_y = 0.95 if swap_rrb else 0.68
            ax.text(
                0.97, rrb_y,
                rf"$r_{{\mathrm{{rb}}}} = {r:.2f}$"
                f"\n95% CI [{lo:.2f}, {hi:.2f}]",
                transform=ax.transAxes, fontsize=6, color="0.30",
                ha="right", va="top", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#333333", lw=0.5, alpha=0.85),
            )

    # 轴标签（纵轴改为密度）
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f"${phi_label[direction]}$")
    ax.set_ylabel(r"$p$", rotation=0, labelpad=10)

    # x 轴刻度
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    # y 轴刻度
    if y_major_locator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))

    # 标题
    K_label = "K = 1000" if K >= 1000 else f"$K = {K}$"
    dir_name = {"xy": "in-plane", "z": "normal"}
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim_auto = ax.get_ylim()
        ax.set_ylim(ylim_auto[0], ylim_auto[1] * 1.25)

    if show_title:
        ax.set_title(
            rf"{title_tag}  ${phi_label[direction]}$  distribution"
            f"    ({K_label}, {dir_name[direction]})",
            pad=4,
            fontsize=10,
        )

    if not hide_legend:
        if swap_rrb:
            ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.68),
                      fontsize=6, frameon=True)
        else:
            ax.legend(loc="upper right", fontsize=6, frameon=True)

    # 保存
    fname_stem = f"eta_{direction}_distribution_{fname_tag}_K{K}"
    fpath = out_dir / f"{fname_stem}.png"
    fig.savefig(fpath)
    print(f"  -> {fpath}")

    plt.close(fig)
    return fname_stem


# ═══════════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Plot ξ/L probability distribution for GPCR proteins."
    )
    ap.add_argument("--K", type=int, default=K_DEFAULT, help="模态截断数")
    ap.add_argument("--bins", type=int, default=BINS_DEFAULT, help="直方图 bin 数")
    ap.add_argument("--h5", type=str, default=H5_PATH, help="H5 文件路径")
    ap.add_argument("--csv", type=str, default=CSV_PATH, help="xi_master_table.csv 路径")
    ap.add_argument("--out", type=str, default=OUT_DIR, help="输出目录")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取 H5 蛋白元数据 (GPCR) ──
    print("== [1] Loading protein metadata from H5 ==")
    meta = load_protein_meta(args.h5, CATEGORY)
    print(f"   {CATEGORY} proteins total: {len(meta)}")

    # ── 2. 筛选 N_res ∈ [375, 475] ──
    mask = meta["N_residues"].between(N_RES_LO, N_RES_HI)
    meta_f = meta[mask]
    pids = set(meta_f["pid"])
    print(
        f"   N_res ∈ [{N_RES_LO}, {N_RES_HI}]: "
        f"{len(pids)} proteins"
    )

    # ── 3. 读取 CSV，计算 η (GPCR) ──
    print(f"== [2] Loading ξ from CSV (K={args.K}) [GPCR] ==")
    eta_dict_gpcr = load_eta(args.csv, pids, S_PAIR, args.K, category=CATEGORY)
    for (direction, s_val), eta in eta_dict_gpcr.items():
        valid = eta[np.isfinite(eta)]
        n_bad = len(eta) - len(valid)
        print(
            f"   direction={direction}, s={s_val}: "
            f"n={len(valid)}, mean={np.mean(valid):.3f}, std={np.std(valid):.3f}"
            + (f"  (dropped {n_bad} NaN)" if n_bad > 0 else "")
        )

    # ── 4. 画图 (GPCR) ──
    print(f"\n== [3] Plotting GPCR ==")
    wide_gpcr = load_eta_wide(args.csv, pids, S_PAIR, args.K, category=CATEGORY)
    # xy: 删除 s=1/16 图注，r_rb 移到右上角，横轴 ρ_xy^*，y 范围 [0, 1.5]
    plot_distribution(
        eta_dict_gpcr, "xy", S_PAIR, out_dir, args.K, args.bins,
        wide=wide_gpcr, title_tag="GPCR", fname_tag="gpcr",
        xlabel=r"$\rho_{xy}^*$", hide_legend=True, swap_rrb=True,
        ylim=(0, 1.5), show_title=False,
    )
    # z: 保留 s=1/16 标签（去 Mean），与 r_rb 互换位置，横轴 ρ_z^*，y 主刻度 0.5，y 范围 [0, 1.8]
    plot_distribution(
        eta_dict_gpcr, "z", S_PAIR, out_dir, args.K, args.bins,
        wide=wide_gpcr, title_tag="GPCR", fname_tag="gpcr",
        xlabel=r"$\rho_{z}^*$", legend_no_mean=True, swap_rrb=True,
        y_major_locator=0.5, ylim=(0, 1.8),
    )

    # ── 5. 读取所有类别蛋白 ⟮N_res ∈ [375, 475]⟯ ──
    ALL_CATEGORIES = ["GPCR", "large", "medium", "single_pass"]
    print(f"\n== [4] Loading all-category protein metadata ==")
    meta_all = load_all_protein_meta(args.h5, ALL_CATEGORIES)
    print(f"   All categories total: {len(meta_all)}")
    mask_all = meta_all["N_residues"].between(N_RES_LO, N_RES_HI)
    meta_all_f = meta_all[mask_all]
    pids_all = set(meta_all_f["pid"])
    print(
        f"   N_res ∈ [{N_RES_LO}, {N_RES_HI}]: "
        f"{len(pids_all)} proteins"
    )

    # ── 6. 计算 η (All categories) ──
    print(f"== [5] Loading ξ from CSV (K={args.K}) [All categories] ==")
    eta_dict_all = load_eta(args.csv, pids_all, S_PAIR, args.K, category=None)
    for (direction, s_val), eta in eta_dict_all.items():
        valid = eta[np.isfinite(eta)]
        n_bad = len(eta) - len(valid)
        print(
            f"   direction={direction}, s={s_val}: "
            f"n={len(valid)}, mean={np.mean(valid):.3f}, std={np.std(valid):.3f}"
            + (f"  (dropped {n_bad} NaN)" if n_bad > 0 else "")
        )

    # ── 7. 画图 (All categories) ──
    print(f"\n== [6] Plotting All categories ==")
    wide_all = load_eta_wide(args.csv, pids_all, S_PAIR, args.K, category=None)
    # xy: 删除 s=1/16 图注，r_rb 移到右上角，横轴 ρ_xy^*
    plot_distribution(
        eta_dict_all, "xy", S_PAIR, out_dir, args.K, args.bins,
        wide=wide_all, title_tag="All categories", fname_tag="all",
        xlabel=r"$\rho_{xy}^*$", hide_legend=True, swap_rrb=True,
    )
    # z: 保留 s=1/16 标签（去 Mean），与 r_rb 互换位置，横轴 ρ_z^*
    plot_distribution(
        eta_dict_all, "z", S_PAIR, out_dir, args.K, args.bins,
        wide=wide_all, title_tag="All categories", fname_tag="all",
        xlabel=r"$\rho_{z}^*$", legend_no_mean=True, swap_rrb=True,
    )

    print(f"\n   Done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
