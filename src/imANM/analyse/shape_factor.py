#!/usr/bin/env python3
"""
shape_factor.py — η = L_z/L_xy 与形状指标 (b, κ²) 散点图。

从 tm_plddt70.h5 读取蛋白元数据（L_xy, L_z, b, c, kappa2, N_residues），
从 xi_master_table.csv 读取 ξ_xy / ξ_z (s=1, K=1000) 用于蛋白筛选，
筛选 N_res ∈ [375, 475] 的所有蛋白，
按类别着色绘制 2 张散点图。

Usage:
  python shape_factor.py [--K 1000] [--s 1]
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

# ── 默认路径 ──────────────────────────────────────────────────────────────────
H5_PATH = "data/process/tm_plddt70.h5"
CSV_PATH = "data/process/xi_master_table_100_800.csv"
OUT_DIR = "results/shape_factor"

K_DEFAULT = 1000
S_DEFAULT = 1
N_RES_LO, N_RES_HI = 375, 475
ALL_CATEGORIES = ["GPCR", "large", "medium", "single_pass"]

# ── 视觉配置 ──────────────────────────────────────────────────────────────────
CAT_COLORS = {
    "GPCR": "#D62728",
    "large": "#1F77B4",
    "medium": "#2CA02C",
    "single_pass": "#9467BD",
}
CAT_MARKERS = {
    "GPCR": "v",
    "large": "s",
    "medium": "^",
    "single_pass": "*",
}
CAT_LABELS = {
    "GPCR": "7 TM(GPCR)",
    "large": "large",
    "medium": "medium",
    "single_pass": "single-pass",
}

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
        "legend.handlelength": 1.2,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
#  读取 H5 蛋白元数据（所有类别）
# ═══════════════════════════════════════════════════════════════════════════════
def load_all_meta(h5_path: str, categories: list) -> pd.DataFrame:
    """读取所有类别的蛋白 attrs，返回合并 DataFrame。"""
    rows = []
    with h5py.File(h5_path, "r") as f:
        for cat in categories:
            if cat not in f:
                print(f"  WARNING: category '{cat}' not found, skipping.")
                continue
            for pid in f[cat]:
                obj = f[cat][pid]
                if not isinstance(obj, h5py.Group):
                    continue
                attrs = dict(obj.attrs)
                if "N_residues" not in attrs:
                    continue
                rows.append(
                    {
                        "pid": pid,
                        "category": cat,
                        "N_residues": int(attrs["N_residues"]),
                        "L_xy": float(attrs["L_xy"]),
                        "L_z": float(attrs["L_z"]),
                        "b": float(attrs.get("b", np.nan)),
                        "c": float(attrs.get("c", np.nan)),
                        "kappa2": float(attrs.get("kappa2", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  合并 CSV 中的 ξ 数据
# ═══════════════════════════════════════════════════════════════════════════════
def merge_xi(
    meta: pd.DataFrame, csv_path: str, K: int, s_val: int
) -> pd.DataFrame:
    """合并 ξ_xy/ξ_z 及形状衍生量 (sqrt_b, shape_eta) 到 meta DataFrame。"""
    df_csv = pd.read_csv(csv_path)
    # 只取 N_res 在范围内的
    df_csv = df_csv[df_csv["N_res"].between(N_RES_LO, N_RES_HI)]

    for direction, col_name in [("xy", "xi_xy"), ("z", "xi_z")]:
        sub = df_csv[
            (df_csv["direction"] == direction)
            & (df_csv["K"] == K)
            & (df_csv["s"] == s_val)
        ][["pid", "xi_zero"]].copy()
        sub = sub.rename(columns={"xi_zero": col_name})
        meta = meta.merge(sub, on="pid", how="left")

    # 计算形状衍生量（仅对正值开根号，负值置 NaN）
    with np.errstate(invalid="ignore"):
        meta["sqrt_b"] = np.where(meta["b"] > 0, np.sqrt(meta["b"]), np.nan)
    meta["shape_eta"] = meta["L_z"] / meta["L_xy"]  # L_z / L_xy

    return meta


# ═══════════════════════════════════════════════════════════════════════════════
#  画单张散点图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    fname: str,
    out_dir: Path,
    K: int,
):
    """按类别着色画散点图。"""
    fig, ax = plt.subplots(figsize=(60 * MM, 50 * MM))

    for cat in ALL_CATEGORIES:
        sub = df[df["category"] == cat].dropna(subset=[x_col, y_col])
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            c=CAT_COLORS[cat],
            marker=CAT_MARKERS[cat],
            s=12,
            edgecolors="none",
            alpha=0.7,
            zorder=3,
            label=CAT_LABELS[cat],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, rotation=0, labelpad=12)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_title(title + f"    ($K = {K}$)", pad=4, fontsize=10)
    ax.legend(loc="upper left", fontsize=6, frameon=True)

    fpath = out_dir / f"{fname}.png"
    fig.savefig(fpath, dpi=600)
    print(f"  -> {fpath}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Scatter plots: η vs shape descriptors (b, κ²)."
    )
    ap.add_argument("--K", type=int, default=K_DEFAULT, help="模态截断数")
    ap.add_argument("--s", type=int, default=S_DEFAULT, help="粗粒化层级 s")
    ap.add_argument("--h5", type=str, default=H5_PATH)
    ap.add_argument("--csv", type=str, default=CSV_PATH)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取 H5 蛋白元数据 ──
    print("== [1] Loading protein metadata from H5 ==")
    meta = load_all_meta(args.h5, ALL_CATEGORIES)
    print(f"   Total proteins: {len(meta)}")

    # ── 2. 筛选 N_res ∈ [375, 475] ──
    meta_f = meta[meta["N_residues"].between(N_RES_LO, N_RES_HI)].copy()
    print(f"   N_res ∈ [{N_RES_LO}, {N_RES_HI}]: {len(meta_f)} proteins")
    for cat in ALL_CATEGORIES:
        n = (meta_f["category"] == cat).sum()
        print(f"      {cat}: {n}")

    # ── 3. 合并 ξ ──
    print(f"== [2] Merging ξ from CSV (K={args.K}, s={args.s}) ==")
    df = merge_xi(meta_f, args.csv, args.K, args.s)
    df = df.dropna(subset=["xi_xy", "xi_z"])
    print(f"   After dropping missing ξ: {len(df)} proteins")

    # ── 4. 画图 ──
    print(f"\n== [3] Plotting ==")

    # 图 1: L_z/L_xy vs b'
    plot_scatter(
        df,
        x_col="sqrt_b",
        y_col="shape_eta",
        x_label=r"$b'$ (nm)",
        y_label=r"$\eta$",
        title=r"$\eta$ vs $b'$",
        fname="Lz_Lxy_vs_sqrt_b",
        out_dir=out_dir,
        K=args.K,
    )

    # 图 2: L_z/L_xy vs κ²
    plot_scatter(
        df,
        x_col="kappa2",
        y_col="shape_eta",
        x_label=r"$\kappa^2$",
        y_label=r"$\eta$",
        title=r"$\eta$ vs $\kappa^2$",
        fname="Lz_Lxy_vs_kappa2",
        out_dir=out_dir,
        K=args.K,
    )

    print(f"\n   Done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
