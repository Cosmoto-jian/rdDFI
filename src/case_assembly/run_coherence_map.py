#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_coherence_map.py — 对 case_assembly 的 4dkl.pdb 直接运行 coherence_map.py 的相干性分析。

直接调用 src/correlation/coherence_map.py 的 run() 管线：
把坐标读取替换为 /Volumes/x23/临时/rdDFI/data/raw/case_assembly/4dkl.pdb（默认全部链，即 CXCR4 二聚体 A+B），
结果图输出到 results/case_assembly/coherence_map/。

4dkl 是二聚体：绘图后在每个热图块上叠加二聚体界面十字（深绿色虚线），
把矩阵分成 A-A / A-B / B-A / B-B 四个象限。

用法:
    python src/case_assembly/run_coherence_map.py                 # 默认整个二聚体（A+B），s = 1 4 16 64
    python src/case_assembly/run_coherence_map.py --chain B       # 只画单条链（无界面十字）
    python src/case_assembly/run_coherence_map.py --s 1 16 64
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "correlation"))
import coherence_map  # noqa: E402

PDB_PATH = Path("/Volumes/x23/临时/rdDFI/data/raw/case_assembly/4dkl.pdb")
PID = "4dkl"


def load_ca_from_pdb(pdb, chain=None):
    """从单个 PDB 文件解析 Cα 坐标（去质心）。

    与 coherence_map.load_ca 的解析方式一致，只是坐标来源换成本地 PDB。
    chain=None 时取全部链（4dkl 为二聚体 A+B）。
    """
    coords, chains = [], []
    with open(pdb) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                coords.append([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
                chains.append(line[21])
    coords, chains = np.asarray(coords, float), np.asarray(chains)
    if chain is not None:
        sel = chains == chain
        if sel.sum() < 3:
            sel = np.ones(len(coords), bool)
        coords = coords[sel]
    return coords - coords.mean(0)


def chain_split(pdb, chain=None):
    """图上第一条链的残基数（链切换位置，即二聚体界面在矩阵中的行/列号）。

    选择逻辑与 load_ca_from_pdb 一致；只含一条链时返回 None。
    """
    chains = []
    with open(pdb) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chains.append(line[21])
    chains = np.asarray(chains)
    if chain is not None:
        sel = chains == chain
        if sel.sum() < 3:
            sel = np.ones(len(chains), bool)
        chains = chains[sel]
    idx = np.flatnonzero(chains != chains[0])
    return int(idx[0]) if len(idx) else None


def _patch_fig_matrices(dimer_split):
    """包装 coherence_map._fig_matrices：在每个热图块上叠加二聚体界面十字。

    原函数在 savefig 后立即 plt.close(fig)，无法事后加工，
    因此拦截 plt.close 把 fig 截获下来，加线后重新保存。
    """
    orig = coherence_map._fig_matrices

    def wrapped(pid, N, mats, s_list, dm):
        captured = []
        orig_close = plt.close
        try:
            plt.close = lambda *a, **k: captured.extend(a)  # 截获未关闭的 fig
            out = orig(pid, N, mats, s_list, dm)
        finally:
            plt.close = orig_close
        if not captured:
            return out
        fig = captured[-1]
        # 界面位于第 dimer_split-1 与第 dimer_split 个残基之间（imshow 单元边界）
        pos = dimer_split - 0.5
        for ax in fig.axes:
            if not ax.get_images():      # 跳过 colorbar 轴
                continue
            xl, yl = ax.get_xlim(), ax.get_ylim()
            ax.axvline(pos, color="#1B5E20", ls="--", lw=1.4, zorder=5)
            ax.axhline(pos, color="#1B5E20", ls="--", lw=1.4, zorder=5)
            ax.set_xlim(xl); ax.set_ylim(yl)
        fig.savefig(out, dpi=140)
        orig_close(fig)
        return out

    return wrapped


def main():
    ap = argparse.ArgumentParser(description="对 4dkl.pdb 运行 coherence_map 相干性分析")
    ap.add_argument("--chain", default=None,
                    help="链 ID（默认 None = 全部链，即二聚体 A+B；A/B = 单链，无界面十字）")
    ap.add_argument("--s", type=float, nargs="+", default=[1, 4, 16, 64],
                    help="膜强度列表，默认 1 4 16 64")
    args = ap.parse_args()

    # 坐标来源换成 4dkl.pdb，输出目录指到 results/case_assembly/coherence_map/
    coherence_map.load_ca = lambda pid, chain=args.chain: load_ca_from_pdb(PDB_PATH, chain)
    coherence_map.OUT_DIR = REPO / "results" / "case_assembly" / "coherence_map"

    # ── 二聚体界面十字：叠加在每个热图块上 ──
    split = chain_split(PDB_PATH, args.chain)
    if split is None:
        print(f"   ⚠ 只绘制单条链（--chain {args.chain}），无链界面，跳过二聚体十字")
    else:
        coherence_map._fig_matrices = _patch_fig_matrices(split)
        print(f"   二聚体界面位于矩阵第 {split} 个残基之后（深绿色虚线十字）")

    coherence_map.run(PID, args.s)


if __name__ == "__main__":
    main()
