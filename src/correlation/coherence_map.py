#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
coherence_map.py — 相干性变化的直观图示（相干块指标的可视化）

回答："面内小模块聚集成大模块、法向大模块瓦解" 长什么样。

一张图：
  fig1  方向分辨的相关矩阵热图（按残基序号排列，不重排）
        上排 xy（面内）| 下排 z（法向），每列一个 s 值，
        每格右上角标注接触对平均相干性 ⟨φ⟩_contact

用法:
    python src/correlation/coherence_map.py            # 随机抽一个蛋白
    python src/correlation/coherence_map.py P00403     # 指定蛋白
    python src/correlation/coherence_map.py P00403 --s 2 64
"""

import sys
import gzip
import random
import argparse
import zipfile
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

HERE     = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
H5_PATH  = Path("/Volumes/x23/临时/rdDFI/data/process/tm_plddt70.h5")
PDB_ZIP  = Path("/Volumes/x23/临时/rdDFI/data/raw/UP000005640_9606.zip")
OUT_DIR  = BASE_DIR.parent / "results" / "coherence_map"

RC       = 10.0   # ANM 接触截断 (Å)
SHORT    = RC     # 短程界限 = 接触截断：直接相连的残基对（与模型自身尺度一致）


def load_ca(pid, chain="A"):
    """从 zip 内读取 <pid>_tr.pdb.gz 并解析 Cα 坐标（链 A）。"""
    name = f"{pid}_tr.pdb.gz"
    with zipfile.ZipFile(PDB_ZIP) as zf:
        member = next((n for n in zf.namelist()
                       if n.rsplit("/", 1)[-1] == name), None)
        if member is None:
            raise FileNotFoundError(f"{name} not found in {PDB_ZIP}")
        text = gzip.decompress(zf.read(member)).decode("utf-8")

    coords, chains = [], []
    for line in text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]),
                           float(line[46:54])])
            chains.append(line[21])
    coords, chains = np.asarray(coords, float), np.asarray(chains)
    sel = chains == chain
    if sel.sum() < 3:
        sel = np.ones(len(coords), bool)
    coords = coords[sel]
    return coords - coords.mean(0)


def build_H(coords):
    from scipy.spatial import cKDTree
    N = len(coords)
    H = np.zeros((3 * N, 3 * N))
    for i, j in cKDTree(coords).query_pairs(RC, output_type="ndarray"):
        dr = coords[j] - coords[i]
        k = np.outer(dr, dr) / (dr @ dr)
        ii, jj = 3 * i, 3 * j
        H[ii:ii+3, jj:jj+3] -= k; H[jj:jj+3, ii:ii+3] -= k
        H[ii:ii+3, ii:ii+3] += k; H[jj:jj+3, jj:jj+3] += k
    return H


def direction_corr(coords, s, mode_tol=1e-6):
    """精确协方差 C_s 的面内/法向残基相关矩阵。"""
    N = len(coords)
    w, V = np.linalg.eigh(build_H(coords))
    nz = w > mode_tol
    lam, Q = w[nz], V[:, nz]
    M = len(lam)
    d = np.tile([np.sqrt(s), np.sqrt(s), 1.0], N)
    Ms = d[:, None] * Q
    pxy = np.tile([1.0, 1.0, 0.0], N)
    R = Q.T @ (pxy[:, None] * Q)
    Gi = np.linalg.inv(np.eye(M) + (s - 1.0) * R)
    C = Ms @ Gi @ np.diag(1.0 / lam) @ Gi @ Ms.T
    Cxy = C[0::3][:, 0::3] + C[1::3][:, 1::3]
    Cz  = C[2::3][:, 2::3]

    def norm(X):
        dd = np.sqrt(np.clip(np.diag(X), 1e-30, None))
        return np.clip(X / np.outer(dd, dd), -1, 1)
    return norm(Cxy), norm(Cz)


def coherent_profile(rho):
    """每个残基的相干块大小 n_i = Σ_j max(ρ_ij, 0)。"""
    return np.clip(rho, 0, None).sum(1)


def pick_random():
    with zipfile.ZipFile(PDB_ZIP) as zf:
        have = {n.rsplit("/", 1)[-1] for n in zf.namelist()
                if not n.startswith("__MACOSX")
                and not n.rsplit("/", 1)[-1].startswith("._")}
    with h5py.File(H5_PATH, "r") as h:
        pool = []
        for cat in h.keys():
            for pid in h[cat].keys():
                if pid == "protein_ids":
                    continue
                p = h[cat][pid]
                if isinstance(p, h5py.Group) and 150 <= p.attrs.get("N_residues", 0) <= 320 \
                        and f"{pid}_tr.pdb.gz" in have:
                    pool.append((pid, float(p.attrs.get("f", np.nan))))
    return random.choice(pool)


# ═══════════════════════════════════════════════════════════════════════
def run(pid, s_list=(1.0, 4.0, 16.0, 64.0)):
    s_list = [float(s) for s in s_list]
    coords = load_ca(pid)
    N = len(coords)

    mats = {}
    for s in s_list:
        mats[("xy", s)], mats[("z", s)] = direction_corr(coords, s)

    dm = squareform(pdist(coords))
    iu = np.triu_indices(N, 1)
    dist = dm[iu]
    contact = dist < SHORT

    print(f"\n{'='*66}\n  相干性图示   {pid}   N={N}   s = "
          f"{', '.join(f'{s:g}' for s in s_list)}\n{'='*66}")
    hdr = "  ".join(f"s={s:g}".rjust(8) for s in s_list)
    print(f"\n  接触对 (r<{SHORT:g}Å=Rc) 的平均相干性")
    print(f"  {'方向':>8} | {hdr}")
    print("  " + "-" * (11 + 10 * len(s_list)))
    for d, tag in [("xy", "面内 xy"), ("z", "法向 z")]:
        vals = [mats[(d, s)][iu][contact].mean() for s in s_list]
        arrow = "↑ 增强" if vals[-1] > vals[0] else "↓ 减弱"
        print(f"  {tag:>8} | " + "  ".join(f"{v:8.3f}" for v in vals) + f"   {arrow}")

    print(f"\n  相干块大小 n_i (残基数)")
    print(f"  {'方向':>8} | {hdr}")
    print("  " + "-" * (11 + 10 * len(s_list)))
    for d, tag in [("xy", "面内 xy"), ("z", "法向 z")]:
        vals = [coherent_profile(mats[(d, s)]).mean() for s in s_list]
        arrow = "↑ 聚集" if vals[-1] > vals[0] else "↓ 瓦解"
        print(f"  {tag:>8} | " + "  ".join(f"{v:8.1f}" for v in vals) + f"   {arrow}")

    f1 = _fig_matrices(pid, N, mats, s_list, dm)
    print(f"[图] {f1}")


def _fig_matrices(pid, N, mats, s_list, dm):
    """2 行(面内/法向) × len(s_list) 列，按残基序号排列，不重排。

    mats: {(direction, s): rho}  direction ∈ {"xy","z"}
    """
    ns = len(s_list)
    fig, ax = plt.subplots(2, ns, figsize=(4.3 * ns, 10.4))
    iu = np.triu_indices(N, 1)
    contact = dm[iu] < SHORT
    rows = [("xy", "$\\phi_{xy}$", "#B2182B"),
            ("z",  "$\\phi_{z}$",  "#2166AC")]
    FS = 20          # 通用字号
    FS_ANN = 16      # 图内标注字号
    ims = []

    for r, (d, name, col) in enumerate(rows):
        for c, s in enumerate(s_list):
            M = mats[(d, s)]
            im = ax[r, c].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1,
                                 interpolation="nearest")
            v = M[iu][contact].mean()
            ax[r, c].set_title(f"{name},   $s={s:g}$", fontsize=FS)
            # 去掉视觉上重复的坐标轴：只在最下一行留横轴、最左一列留纵轴
            if r == 1:
                ax[r, c].set_xlabel("residue index", fontsize=FS)
                ax[r, c].tick_params(axis="x", labelsize=FS)
            else:
                ax[r, c].set_xticks([])
            if c == 0:
                ax[r, c].set_ylabel("residue index", fontsize=FS)
                ax[r, c].tick_params(axis="y", labelsize=FS)
            else:
                ax[r, c].set_yticks([])
            # 标注移到右上角，只留 <phi>_contact
            ax[r, c].text(0.965, 0.965,
                          f"$\\langle\\phi\\rangle_{{contact}}$ = {v:.3f}",
                          transform=ax[r, c].transAxes, va="top", ha="right",
                          fontsize=FS_ANN, fontweight="bold", color=col,
                          bbox=dict(fc="white", ec="0.6", alpha=.85))
            ax[r, c].set_aspect("equal")
        ims.append(im)

    fig.suptitle(
        f"{pid} (N={N}): direction-resolved correlation vs membrane strength\n"
        f"residue order, no reordering;  "
        f"$\\langle\\phi\\rangle_{{contact}}$ = mean over pairs with $r<R_c={SHORT:g}$ A",
        fontsize=FS)
    fig.subplots_adjust(left=0.065, right=0.895, top=0.86, bottom=0.085,
                        wspace=0.07, hspace=0.16)
    # aspect="equal" 会在绘制时重排子图框，必须先 draw 再取真实位置
    fig.canvas.draw()
    for r in range(2):
        p0 = ax[r, ns - 1].get_position()
        cax = fig.add_axes([0.915, p0.y0, 0.013, p0.height])
        cb = fig.colorbar(ims[r], cax=cax)
        cb.ax.tick_params(labelsize=FS)
    tag = "_".join(f"{s:g}" for s in s_list)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"coh_matrix_{pid}_s{tag}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="相干性变化的热图/剖面图示")
    ap.add_argument("pid", nargs="?", default=None)
    ap.add_argument("--s", type=float, nargs="+", default=[1, 4, 16, 64],
                    help="膜强度列表，默认 1 4 16 64")
    a = ap.parse_args()
    if a.pid:
        run(a.pid, a.s)
    else:
        pid, f = pick_random()
        print(f"（随机抽取蛋白：{pid}，f={f:.2f}）")
        run(pid, a.s)
