#!/usr/bin/env python3
"""
imANM_pipeline.py  —  Cα 粗粒化 + imANM s-参数扫描 + 方向分辨余弦关联

流程：
  1. PDB → Cα
  2. 对 s ∈ {1,2,4,8,16,32,64} 逐一建 Hessian、对角化
  3. 对全部珠子计算按 5Å 分箱的归一化余弦关联：总的 / 面内(xy) / 法向(z)
     特征值和特征向量仅做临时中间量，算完即释放

输出：
  批量模式: 回写 data/raw/case_assembly/case_assembly.h5
            （"single_chain" / "assembly" 两个类别）
  单蛋白:   <out_dir>/results.h5

Usage:
    批量模式:  python src/case_assembly/imANM_assembly.py --batch
    单蛋白:    python src/case_assembly/imANM_assembly.py <input.pdb> <out_dir>
               [ca_per_group=1]
"""

import sys
import gzip
from pathlib import Path

import numpy as np
import h5py
from Bio import PDB
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

# ── 物理参数 ──────────────────────────────────────────────────────────────
RC       = 10.0
GAMMA_Z  = 1.0
S_VALUES = [1, 2, 4, 8, 16, 32, 64]
K_VALUES = [20, 50, 100, 200, 1000]
BIN_WIDTH = 5   # Å

# ── 批量模式路径（case_assembly）─────────────────────────────────────────
CASE_H5_PATH = Path("/Volumes/x23/临时/rdDFI/data/raw/case_assembly/case_assembly.h5")
# (类别, H5 组名, 输入 PDB)：全部按全珠子关联计算
CASE_PDBS = [
    ("single_chain", "4dkl",
     Path("/Volumes/x23/临时/rdDFI/data/raw/case_assembly/4dkl_chainA.pdb")),
    ("assembly", "4dkl",
     Path("/Volumes/x23/临时/rdDFI/data/raw/case_assembly/4dkl.pdb")),
]


# ═══════════════════════════════════════════════════════════════════════════
#  PDB → Cα 粗粒化
# ═══════════════════════════════════════════════════════════════════════════
def _parse_ca(handle, ca_per_group=1, label="input"):
    """从已打开的文件句柄解析结构并提取 Cα（公共实现）。"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", handle)
    model = next(iter(structure))

    ca_info = []
    for chain in model:
        for residue in chain:
            for atom in residue:
                if atom.get_id() == "CA":
                    ca_info.append((atom.coord, chain.get_id(),
                                    residue.get_id()[1]))
    if not ca_info:
        raise ValueError(f"No CA atoms found in {label}")

    coords, chains, resids = [], [], []
    for i in range(0, len(ca_info), ca_per_group):
        grp = ca_info[i:i + ca_per_group]
        coords.append(np.mean([g[0] for g in grp], axis=0))
        chains.append(grp[0][1])
        resids.append(grp[0][2])

    coords = np.asarray(coords, dtype=float)
    print(f"   {len(ca_info)} CA -> {len(coords)} beads "
          f"(ca_per_group={ca_per_group})")
    return coords, chains, resids


def extract_ca(pdb_path, ca_per_group=1):
    """路径入口（单蛋白模式 / 目录文件）；.gz 走 gzip，其余直接解析。"""
    pdb_str = str(pdb_path)
    if pdb_str.endswith(".gz"):
        with gzip.open(pdb_str, "rt") as fh:
            return _parse_ca(fh, ca_per_group, label=pdb_str)
    return _parse_ca(pdb_str, ca_per_group, label=pdb_str)


# ═══════════════════════════════════════════════════════════════════════════
#  imANM Hessian
# ═══════════════════════════════════════════════════════════════════════════
def build_hessian(coords, rc, gamma_xy, gamma_z):
    N = len(coords)
    H = np.zeros((3 * N, 3 * N))
    g = np.array([gamma_xy, gamma_xy, gamma_z])
    gmat = np.sqrt(np.outer(g, g))

    pairs = cKDTree(coords).query_pairs(r=rc, output_type="ndarray")
    diffs = coords[pairs[:, 1]] - coords[pairs[:, 0]]
    dist2 = np.einsum("ij,ij->i", diffs, diffs)

    for (i, j), diff, d2 in zip(pairs, diffs, dist2):
        outer = np.outer(diff, diff) * gmat / d2
        ii, jj = 3 * i, 3 * j
        H[ii:ii+3, jj:jj+3]  = -outer
        H[jj:jj+3, ii:ii+3]  = -outer
        H[ii:ii+3, ii:ii+3] += outer
        H[jj:jj+3, jj:jj+3] += outer
    return H


# ═══════════════════════════════════════════════════════════════════════════
#  方向分辨余弦关联
# ═══════════════════════════════════════════════════════════════════════════
def chain_cov(vals, vecs, cidx, N, K, alphas, n_skip):
    n = len(cidx)
    C = np.zeros((n, n))
    for k in range(n_skip, min(K, 3 * N)):
        lam = vals[k]
        if lam < 1e-10:
            continue
        inv = 1.0 / lam
        for a in alphas:
            u = vecs[a::3, k][cidx]
            C += np.outer(u, u) * inv
    return C


def normalise(C):
    d = np.diag(C).copy()
    d[d < 1e-30] = 1e-30
    return C / np.sqrt(np.outer(d, d))


def bin_corr(dm, rho, bw):
    iu = np.triu_indices(len(dm), k=1)
    ds, rs = dm[iu], rho[iu]
    edges = np.arange(0, ds.max() + bw, bw)
    ctrs = (edges[:-1] + edges[1:]) / 2
    m   = np.full(len(ctrs), np.nan)
    sem = np.full(len(ctrs), np.nan)
    cnt = np.zeros(len(ctrs), int)
    bi = np.digitize(ds, edges) - 1
    for b in range(len(ctrs)):
        sel = bi == b
        nb = sel.sum()
        if nb == 0:
            continue
        v = rs[sel]
        cnt[b] = nb
        m[b] = v.mean()
        if nb >= 2:
            sem[b] = v.std() / np.sqrt(nb)
    ok = cnt > 0
    return ctrs[ok], m[ok], sem[ok], cnt[ok]


def compute_binned_corr(vals, vecs, cidx, N, K, n_skip, dm, bw):
    specs = {"total": [0, 1, 2], "xy": [0, 1], "z": [2]}
    result = {}
    for tag, alphas in specs.items():
        C = chain_cov(vals, vecs, cidx, N, K, alphas, n_skip)
        rho = normalise(C)
        r, m, sem, cnt = bin_corr(dm, rho, bw)
        result[tag] = {"r": r, "rho": m, "sem": sem, "count": cnt}
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  核心计算（不含文件 I/O，供单蛋白和批量模式共用）
# ═══════════════════════════════════════════════════════════════════════════
def compute_imANM(coords):
    """对已粗粒化的坐标执行 imANM 扫描，返回结果字典。

    协方差/关联一律对全部珠子计算（不做链筛选）。

    Returns
    -------
    results : dict
        {s: {"gamma_xy": ..., "n_zero_modes": ...,
             "K": {K: {"n_internal_modes": ..., "binned": {...}}}}}
    chain_used : str  (恒为 "ALL")
    N_beads : int
    N_beads_chain : int  (= N_beads)
    dm : ndarray
    """
    N = len(coords)
    cidx = np.arange(N)      # 全珠子关联
    chain_used = "ALL"
    print(f"   All chains: {N} beads")

    dm = squareform(pdist(coords[cidx]))

    results = {}
    for s in S_VALUES:
        gamma_xy = s * GAMMA_Z
        print(f"   s = {s}  (gamma_xy = {gamma_xy})", end="")

        H = build_hessian(coords, RC, gamma_xy, GAMMA_Z)
        vals, vecs = np.linalg.eigh(H)
        n_skip = int((np.abs(vals) < 1e-6).sum())
        print(f"  {N} beads, {n_skip} zero modes")

        s_data = {"gamma_xy": gamma_xy, "n_zero_modes": n_skip, "K": {}}
        for K in K_VALUES:
            n_int = max(min(K, 3 * N) - n_skip, 0)
            binned = compute_binned_corr(
                vals, vecs, cidx, N, K, n_skip, dm, BIN_WIDTH)
            s_data["K"][K] = {"n_internal_modes": n_int, "binned": binned}
            print(f"      K={K:<5d}  modes={n_int}")

        results[s] = s_data
        del H, vals, vecs

    return results, chain_used, N, len(cidx), dm


def _write_imANM_to_h5group(h5f, grp_path, results, N, chain_used,
                            ca_per_group):
    """将 imANM 结果写入 H5 中指定 group 路径下。"""
    grp = h5f.require_group(grp_path)
    for attr_name, attr_val in [
        ("rc", RC), ("gamma_z", GAMMA_Z), ("N_beads", N),
        ("ca_per_group", ca_per_group), ("s_values", S_VALUES),
        ("K_values", K_VALUES), ("bin_width", BIN_WIDTH),
        ("corr_chain", chain_used),
    ]:
        grp.attrs[attr_name] = attr_val

    for s, s_data in results.items():
        sg = grp.require_group(f"s{s}")
        sg.attrs["s_factor"] = s
        sg.attrs["gamma_xy"] = s_data["gamma_xy"]
        sg.attrs["n_zero_modes"] = s_data["n_zero_modes"]

        for K, k_data in s_data["K"].items():
            kg = sg.require_group(f"K{K}")
            kg.attrs["K"] = K
            kg.attrs["n_internal_modes"] = k_data["n_internal_modes"]
            for tag in ["total", "xy", "z"]:
                tg = kg.require_group(tag)
                b = k_data["binned"][tag]
                for ds_name, dict_key in [("bin_r", "r"), ("bin_rho", "rho"),
                                           ("bin_sem", "sem"), ("bin_count", "count")]:
                    if ds_name in tg:
                        del tg[ds_name]
                    tg.create_dataset(ds_name, data=b[dict_key])


# ═══════════════════════════════════════════════════════════════════════════
#  批量模式：对 case_assembly 的两个蛋白（single_chain / assembly）
#  逐一计算并写入 data/raw/case_assembly/case_assembly.h5
# ═══════════════════════════════════════════════════════════════════════════
def run_batch_pipeline(h5_path=None, case_pdbs=None, ca_per_group=1):
    h5_path = Path(h5_path or CASE_H5_PATH)
    case_pdbs = case_pdbs or CASE_PDBS

    n_total = len(case_pdbs)
    print(f"Output: {h5_path}")
    print(f"Cases:  {n_total} ({[c for c, _, _ in case_pdbs]})")

    with h5py.File(h5_path, "a") as h5f:
        for idx, (cat_key, pid, pdb_path) in enumerate(case_pdbs, 1):
            grp_path = f"{cat_key}/{pid}"
            # 已按全珠子关联计算过则跳过；旧数据（链筛选）删除重算
            if grp_path in h5f and "s1" in h5f[grp_path]:
                prev_chain = str(h5f[grp_path].attrs.get("corr_chain", "")).upper()
                if prev_chain == "ALL":
                    print(f"\n[{idx}/{n_total}] SKIP {pid}  ({cat_key}: already computed)")
                    continue
                print(f"\n[{idx}/{n_total}] {pid}  ({cat_key})  "
                      f"corr_chain '{prev_chain}' -> 'ALL': recompute")
                del h5f[grp_path]

            print(f"\n[{idx}/{n_total}] {pid}  ({cat_key})")
            print(f"   PDB: {pdb_path}")

            try:
                print("   Coarse-graining...", end=" ")
                coords, _, _ = extract_ca(pdb_path, ca_per_group)
                results, chain_used, N, _, _ = compute_imANM(coords)

                print("   Writing to H5...", end=" ")
                _write_imANM_to_h5group(
                    h5f, grp_path, results, N, chain_used, ca_per_group)
                h5f.flush()
                print("done.")
            except Exception as e:
                print(f"\n   ERROR: {e}")
                continue

    print(f"\n== Batch done. {n_total} cases processed. ==")


# ═══════════════════════════════════════════════════════════════════════════
#  单蛋白模式（保持向后兼容）
# ═══════════════════════════════════════════════════════════════════════════
def run_pipeline(pdb_in, out_dir, ca_per_group=1):
    pdb_in  = Path(pdb_in)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_out = out_dir / "results.h5"

    print("== Coarse-graining ==")
    coords, _, _ = extract_ca(pdb_in, ca_per_group)
    N = len(coords)

    cidx = np.arange(N)          # 全珠子关联
    print(f"   All chains: {N} beads")
    dm = squareform(pdist(coords[cidx]))

    with h5py.File(h5_out, "w") as hf:
        hf.attrs["rc"]           = RC
        hf.attrs["gamma_z"]      = GAMMA_Z
        hf.attrs["N_beads"]      = N
        hf.attrs["ca_per_group"] = ca_per_group
        hf.attrs["s_values"]     = S_VALUES
        hf.attrs["K_values"]     = K_VALUES
        hf.attrs["bin_width"]    = BIN_WIDTH
        hf.attrs["corr_chain"]   = "ALL"

        for s in S_VALUES:
            gamma_xy = s * GAMMA_Z
            print(f"\n== s = {s}  (gamma_xy = {gamma_xy}) ==")

            H = build_hessian(coords, RC, gamma_xy, GAMMA_Z)
            vals, vecs = np.linalg.eigh(H)
            n_skip = int((np.abs(vals) < 1e-6).sum())
            print(f"   {N} beads, {n_skip} zero modes")

            sg = hf.create_group(f"s{s}")
            sg.attrs["s_factor"]     = s
            sg.attrs["gamma_xy"]     = gamma_xy
            sg.attrs["n_zero_modes"] = n_skip

            for K in K_VALUES:
                n_int = max(min(K, 3 * N) - n_skip, 0)
                binned = compute_binned_corr(
                    vals, vecs, cidx, N, K, n_skip, dm, BIN_WIDTH)

                kg = sg.create_group(f"K{K}")
                kg.attrs["K"] = K
                kg.attrs["n_internal_modes"] = n_int
                for tag in ["total", "xy", "z"]:
                    tg = kg.create_group(tag)
                    tg.create_dataset("bin_r",     data=binned[tag]["r"])
                    tg.create_dataset("bin_rho",   data=binned[tag]["rho"])
                    tg.create_dataset("bin_sem",   data=binned[tag]["sem"])
                    tg.create_dataset("bin_count", data=binned[tag]["count"])

                print(f"   K={K:<5d}  modes={n_int}")

            del H, vals, vecs

    print(f"\n   -> {h5_out}")
    return h5_out


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--batch":
        run_batch_pipeline()
    elif len(sys.argv) < 3:
        print("Usage:")
        print("  Batch mode:  imANM_assembly.py --batch")
        print("  Single PDB:  imANM_assembly.py <input.pdb> <out_dir> "
              "[ca_per_group=1]")
        sys.exit(1)
    else:
        cg = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        run_pipeline(sys.argv[1], sys.argv[2], ca_per_group=cg)