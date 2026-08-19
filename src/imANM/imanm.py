#!/usr/bin/env python3
"""
imANM_pipeline.py  —  Cα 粗粒化 + imANM s-参数扫描 + 方向分辨余弦关联

流程：
  1. PDB → Cα
  2. 对 s ∈ {1,2,4,8,16,32,64} 逐一建 Hessian、对角化
  3. 对指定链计算按 5Å 分箱的归一化余弦关联：总的 / 面内(xy) / 法向(z)
     特征值和特征向量仅做临时中间量，算完即释放

输出：
  批量模式: 回写 <H5_DATA_PATH>（data/process/tm_plddt70.h5）的分箱关联数据
  单蛋白:   <out_dir>/results.h5

Usage:
    批量模式:  python src/imANM/imanm.py --batch
    单蛋白:    python src/imANM/imanm.py <input.pdb> <out_dir>
               [chain=A] [ca_per_group=1]
"""

import sys
import gzip
import io
import zipfile
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

# ── 批量模式路径 ──────────────────────────────────────────────────────────
PROJECT_DIR    = Path(__file__).resolve().parents[2]
H5_DATA_PATH   = PROJECT_DIR / "data" / "process" / "tm_plddt70.h5"
PDB_ZIP        = PROJECT_DIR / "data" / "raw" / "UP000005640_9606.zip"
N_RES_MIN      = 100
N_RES_MAX      = 800
CATEGORIES     = ["single_pass", "medium", "GPCR", "large"]


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
def compute_imANM(coords, chains_list, chain="A"):
    """对已粗粒化的坐标执行 imANM 扫描，返回结果字典。

    Returns
    -------
    results : dict
        {s: {"gamma_xy": ..., "n_zero_modes": ...,
             "K": {K: {"n_internal_modes": ..., "binned": {...}}}}}
    chain_used : str
    N_beads : int
    N_beads_chain : int
    dm : ndarray
    """
    N = len(coords)
    chains_arr = np.array(chains_list)

    cidx = np.where(chains_arr == chain)[0]
    if len(cidx) == 0:
        avail = sorted(set(chains_list))
        print(f"   Warning: chain '{chain}' not found. Available: {avail}")
        chain = avail[0]
        cidx = np.where(chains_arr == chain)[0]
        print(f"   -> Falling back to chain '{chain}' ({len(cidx)} beads)")
    else:
        print(f"   Chain {chain}: {len(cidx)} beads")

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

    return results, chain, N, len(cidx), dm


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
#  批量模式：遍历 H5 中 100 ≤ N ≤ 800 的蛋白，从 zip 读取 PDB，逐个计算并回写
# ═══════════════════════════════════════════════════════════════════════════
def run_batch_pipeline(h5_path=None, pdb_zip=None, n_min=N_RES_MIN,
                       n_max=N_RES_MAX, ca_per_group=1):
    h5_path = Path(h5_path or H5_DATA_PATH)
    pdb_zip = Path(pdb_zip or PDB_ZIP)

    # 0. 打开 zip，建立 basename → member 映射（剔除 __MACOSX/ 与 ._* 垃圾条目）
    with zipfile.ZipFile(pdb_zip) as zf:
        members = {
            name.rsplit("/", 1)[-1]: name
            for name in zf.namelist()
            if not name.startswith("__MACOSX")
            and not name.rsplit("/", 1)[-1].startswith("._")
            and name.endswith(".gz")
        }

    # 1. 扫描 H5，收集符合条件的蛋白
    print(f"Scanning: {h5_path}")
    print(f"Filter:  {n_min} ≤ N_residues ≤ {n_max}")
    proteins = []  # [(cat_key, pid, zip_member)]
    skipped_missing = 0
    with h5py.File(h5_path, "r") as h5f:
        for cat_key in CATEGORIES:
            if cat_key not in h5f:
                continue
            grp = h5f[cat_key]
            for pid in grp.keys():
                sub = grp[pid]
                if not isinstance(sub, h5py.Group):
                    continue
                n_res = sub.attrs.get("N_residues", 0)
                if n_min <= n_res <= n_max:
                    member = members.get(f"{pid}_tr.pdb.gz")
                    if member:
                        proteins.append((cat_key, pid, member))
                    else:
                        skipped_missing += 1

    print(f"  -> {len(proteins)} proteins matched "
          f"({skipped_missing} skipped: PDB not found in zip)")
    if not proteins:
        print("No proteins to process.")
        return

    # 2. 逐个处理
    n_total = len(proteins)
    with zipfile.ZipFile(pdb_zip) as zf, h5py.File(h5_path, "a") as h5f:
        for idx, (cat_key, pid, member) in enumerate(proteins, 1):
            grp_path = f"{cat_key}/{pid}"
            # 跳过已计算过的
            if grp_path in h5f and "s1" in h5f[grp_path]:
                print(f"\n[{idx}/{n_total}] SKIP {pid}  (already computed)")
                continue

            print(f"\n[{idx}/{n_total}] {pid}  ({cat_key})")
            print(f"   PDB: {member}")

            try:
                print("   Coarse-graining...", end=" ")
                raw = zf.read(member)
                text = gzip.decompress(raw).decode("utf-8")
                coords, chains_list, resids = _parse_ca(
                    io.StringIO(text), ca_per_group, label=member)
                results, chain_used, N, _, _ = compute_imANM(
                    coords, chains_list)

                print("   Writing to H5...", end=" ")
                _write_imANM_to_h5group(
                    h5f, grp_path, results, N, chain_used, ca_per_group)
                h5f.flush()
                print("done.")
            except Exception as e:
                print(f"\n   ERROR: {e}")
                continue

    print(f"\n== Batch done. {n_total} proteins processed. ==")


# ═══════════════════════════════════════════════════════════════════════════
#  单蛋白模式（保持向后兼容）
# ═══════════════════════════════════════════════════════════════════════════
def run_pipeline(pdb_in, out_dir, chain="A", ca_per_group=1):
    pdb_in  = Path(pdb_in)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_out = out_dir / "results.h5"

    print("== Coarse-graining ==")
    coords, chains_list, resids = extract_ca(pdb_in, ca_per_group)
    N = len(coords)
    chains_arr = np.array(chains_list)

    cidx = np.where(chains_arr == chain)[0]
    if len(cidx) == 0:
        avail = sorted(set(chains_list))
        print(f"   Warning: chain '{chain}' not found. Available: {avail}")
        chain = avail[0]
        cidx = np.where(chains_arr == chain)[0]
        print(f"   -> Falling back to chain '{chain}' ({len(cidx)} beads)")
    else:
        print(f"   Chain {chain}: {len(cidx)} beads")

    dm = squareform(pdist(coords[cidx]))

    with h5py.File(h5_out, "w") as hf:
        hf.attrs["rc"]           = RC
        hf.attrs["gamma_z"]      = GAMMA_Z
        hf.attrs["N_beads"]      = N
        hf.attrs["ca_per_group"] = ca_per_group
        hf.attrs["s_values"]     = S_VALUES
        hf.attrs["K_values"]     = K_VALUES
        hf.attrs["bin_width"]    = BIN_WIDTH
        hf.attrs["corr_chain"]   = chain

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
        print("  Batch mode:  imanm.py --batch")
        print("  Single PDB:  imanm.py <input.pdb> <out_dir> "
              "[chain=A] [ca_per_group=1]")
        sys.exit(1)
    else:
        ch = sys.argv[3] if len(sys.argv) > 3 else "A"
        cg = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        run_pipeline(sys.argv[1], sys.argv[2], chain=ch, ca_per_group=cg)