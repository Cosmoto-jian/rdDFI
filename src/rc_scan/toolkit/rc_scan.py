#!/usr/bin/env python3
"""
rc_scan.py — 批量 RC 参数扫描

遍历 H5 数据库中所有符合长度要求的跨膜蛋白，
对每个蛋白以 RC ∈ {8, 10, 12, 15} Å 逐一计算 imANM，
结果按 RC 值分别写入 data/process/rc_{rc}.h5 文件。

Usage:
    python rc_scan.py
"""

import gzip
import io
import importlib.util
import zipfile
from pathlib import Path

import h5py

# ── 从项目内加载 imanm.py ────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[3]
IMANM_PATH = PROJECT_DIR / "src" / "imANM" / "imanm.py"
imanm_spec = importlib.util.spec_from_file_location("imanm", IMANM_PATH)
if imanm_spec is None or imanm_spec.loader is None:
    raise ImportError(f"Cannot load imANM module from {IMANM_PATH}")
imanm = importlib.util.module_from_spec(imanm_spec)
imanm_spec.loader.exec_module(imanm)

# ── 参数配置 ─────────────────────────────────────────────────────────────
RC_VALUES  = [8, 10, 12, 15]
N_RES_MIN  = 375
N_RES_MAX  = 475
K_KEEP     = [1000]          # 只计算 K=1000
S_KEEP     = [16]            # 只计算 s=16

# monkey-patch imanm 的全局参数，避免无效计算
imanm.S_VALUES = S_KEEP
imanm.K_VALUES = K_KEEP

PROCESS_DIR = PROJECT_DIR / "data" / "process"
H5_PATH     = PROCESS_DIR / "tm_plddt70.h5"
PDB_ZIP     = PROJECT_DIR / "data" / "raw" / "UP000005640_9606.zip"
CATEGORIES = ["GPCR", "large", "medium", "single_pass"]


def main():
    PROCESS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 扫描 ZIP，建立 basename -> member 映射 ──
    with zipfile.ZipFile(PDB_ZIP) as zf:
        pdb_members = {
            name.rsplit("/", 1)[-1]: name
            for name in zf.namelist()
            if not name.startswith("__MACOSX")
            and not name.rsplit("/", 1)[-1].startswith("._")
            and name.endswith(".pdb.gz")
        }

    # ── 2. 扫描 H5，收集符合条件的蛋白 ──
    print(f"Scanning: {H5_PATH}")
    print(f"Filter:  {N_RES_MIN} ≤ N_residues ≤ {N_RES_MAX}")
    proteins = []  # [(cat_key, pid, zip_member, N_res)]
    with h5py.File(H5_PATH, "r") as h5f:
        for cat in CATEGORIES:
            if cat not in h5f:
                continue
            for pid in sorted(h5f[cat].keys()):
                prot = h5f[cat][pid]
                if not isinstance(prot, h5py.Group):
                    continue
                N_res = prot.attrs.get("N_residues", 0)
                if N_RES_MIN <= N_res <= N_RES_MAX:
                    zip_member = pdb_members.get(f"{pid}_tr.pdb.gz")
                    if zip_member:
                        proteins.append((cat, pid, zip_member, N_res))

    n_total = len(proteins)
    print(f"  -> {n_total} proteins matched")
    if n_total == 0:
        print("No proteins to process.")
        return

    # ── 3. 打开 4 个 RC 专属 H5 文件 ──
    h5_files = {}
    for rc in RC_VALUES:
        h5_path = PROCESS_DIR / f"rc_{rc}.h5"
        h5f = h5py.File(h5_path, "a")
        h5f.attrs["rc"] = rc
        h5f.attrs["gamma_z"] = imanm.GAMMA_Z
        h5f.attrs["K_values"] = K_KEEP
        h5f.attrs["s_values"] = S_KEEP
        h5f.attrs["bin_width"] = imanm.BIN_WIDTH
        h5_files[rc] = h5f

    # ── 4. 逐个蛋白处理 ──
    try:
        with zipfile.ZipFile(PDB_ZIP) as zf:
            for idx, (cat, pid, zip_member, N_res) in enumerate(proteins, 1):
                grp_path = f"{cat}/{pid}"

                # 检查是否所有 RC 都已计算
                all_done = True
                for rc in RC_VALUES:
                    if grp_path not in h5_files[rc] or f"s{S_KEEP[0]}" not in h5_files[rc][grp_path]:
                        all_done = False
                        break
                if all_done:
                    print(f"\n[{idx}/{n_total}] SKIP {pid} ({cat}, N={N_res})  "
                          f"— already computed for all RC")
                    continue

                print(f"\n[{idx}/{n_total}] {pid}  ({cat}, N={N_res})")
                print(f"   PDB: {zip_member}")

                try:
                    # 粗粒化（只做一次）
                    print("   Coarse-graining...", end=" ", flush=True)
                    pdb_text = gzip.decompress(zf.read(zip_member)).decode("utf-8")
                    coords, chains_list, _ = imanm._parse_ca(
                        io.StringIO(pdb_text), label=zip_member)
                    N_beads = len(coords)
                    print(f"{N_beads} beads")

                    # 对每个 RC 计算
                    for rc in RC_VALUES:
                        # 跳过已完成的
                        if grp_path in h5_files[rc] and f"s{S_KEEP[0]}" in h5_files[rc][grp_path]:
                            print(f"   RC={rc}: already done, skip")
                            continue

                        print(f"   RC={rc}: computing...", end=" ", flush=True)
                        imanm.RC = rc

                        results, chain_used, N, _, _ = imanm.compute_imANM(
                            coords, chains_list)

                        # 只写入 s=1 和 s=16 的数据
                        results_filtered = {s: results[s] for s in S_KEEP if s in results}

                        imanm._write_imANM_to_h5group(
                            h5_files[rc], grp_path, results_filtered,
                            N, chain_used, ca_per_group=1)
                        # 补充写入 N_residues
                        grp = h5_files[rc][grp_path]
                        grp.attrs["N_residues"] = N_res
                        h5_files[rc].flush()
                        print("done")

                except Exception as e:
                    print(f"\n   ERROR: {e}")
                    continue

    finally:
        for h5f in h5_files.values():
            h5f.close()

    print(f"\n== Batch done. {n_total} proteins processed. ==")
    for rc in RC_VALUES:
        f = PROCESS_DIR / f"rc_{rc}.h5"
        print(f"  {'✓' if f.exists() else '✗'}  {f}")


if __name__ == "__main__":
    main()
