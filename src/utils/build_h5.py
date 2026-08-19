#!/usr/bin/env python3
"""
Build an HDF5 file grouping high-confidence transmembrane proteins.

从 data/raw/UP000005640_9606.zip 一站式重建建库流水线（原 classify_tm.py +
add_plddt_to_csv.py + 旧 build_h5.py 三步合并的入口）：

  1. 解析 zip 内 *_tmdet.xml 的 NUM_TM → 分类（复用 classify_tm.classify_tm）
  2. 解析 zip 内 *_tr.pdb.gz：一趟读出 Cα 坐标与 B-factor 列 →
     N_residues、pLDDT_mean_CA / pLDDT_mean_all
  3. 几何量 L_xy / L_z / f（复用 add_geometry_to_h5.geometric_scales）
  4. 过滤 pLDDT_mean_CA > 70，按 Category 分组
  5. 创建主 HDF5 数据库（首次建库时使用；不会保留旧文件中的 imANM 结果）：
       data/process/tm_classification.csv
    data/process/tm_plddt70.h5

HDF5 结构与现 h5 的 per-protein 布局对齐：
  /{cat}/
      protein_ids          ← 该类别全部入选 ID 列表
      {Protein_ID}/        ← per-protein group
          @N_residues, @NUM_TM, @pLDDT_mean_CA, @pLDDT_mean_all,
          @L_xy, @L_z, @f

Usage:
    python src/utils/build_h5.py
"""

import csv
import gzip
import io
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
from classify_tm import classify_tm, NS                 # noqa: E402
from add_geometry_to_h5 import geometric_scales        # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────
PDB_ZIP  = PROJECT_DIR / "data" / "raw" / "UP000005640_9606.zip"
OUT_DIR  = PROJECT_DIR / "data" / "process"
CSV_PATH = OUT_DIR / "tm_classification.csv"
H5_PATH  = OUT_DIR / "tm_plddt70.h5"

PLDDT_CUTOFF = 70.0

# ── Category → HDF5 group name mapping ─────────────────────────────────────
CATEGORY_GROUPS = {
    "single-pass transmembrane": "single_pass",
    "medium transmembrane":      "medium",
    "GPCR transmembrane":        "GPCR",
    "large transmembrane":       "large",
}


def zip_members(zf):
    """真实条目列表（剔除 __MACOSX/ 与 ._* AppleDouble 垃圾条目）。"""
    return [n for n in zf.namelist()
            if not n.startswith("__MACOSX")
            and not n.rsplit("/", 1)[-1].startswith("._")]


def parse_pdb_bytes(raw):
    """一趟解析解压后的 PDB：Cα 坐标、残基数、pLDDT 均值。

    合并 add_plddt_to_csv.calc_plddt（B-factor 列 61-66）与
    add_geometry_to_h5.extract_ca_coords 的逻辑。

    Returns (N_residues, plddt_ca, plddt_all, coords)；失败返回全 None。
    """
    ca_sum = all_sum = 0.0
    ca_count = all_count = 0
    coords = []

    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            b_factor = float(line[60:66])
        except ValueError:
            continue
        all_sum += b_factor
        all_count += 1

        if line[12:16].strip() == "CA":
            ca_sum += b_factor
            ca_count += 1
            try:
                coords.append([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
            except ValueError:
                pass

    if ca_count == 0 or all_count == 0:
        return None, None, None, None
    return (ca_count,
            round(ca_sum / ca_count, 4),
            round(all_sum / all_count, 4),
            np.asarray(coords, dtype=np.float64))


def main():
    grouped = {grp: [] for grp in CATEGORY_GROUPS.values()}
    csv_rows = []                 # 全部已分类蛋白（含被过滤的）
    total = kept = missing_pdb = 0

    with zipfile.ZipFile(PDB_ZIP) as zf:
        xml_by_pid, pdb_by_pid = {}, {}
        for n in zip_members(zf):
            base = n.rsplit("/", 1)[-1]
            if base.endswith("_tmdet.xml"):
                xml_by_pid[base[:-10]] = n          # 去掉 "_tmdet.xml"
            elif base.endswith("_tr.pdb.gz"):
                pdb_by_pid[base[:-10]] = n          # 去掉 "_tr.pdb.gz"
        print(f"Zip: {PDB_ZIP}")
        print(f"  {len(xml_by_pid)} *_tmdet.xml, {len(pdb_by_pid)} *_tr.pdb.gz")

        # ── 1. 分类 + PDB 解析 ─────────────────────────────────────────
        for idx, (pid, xml_member) in enumerate(sorted(xml_by_pid.items()), 1):
            # 1a. NUM_TM 分类（复用 classify_tm 逻辑）
            try:
                root = ET.parse(io.BytesIO(zf.read(xml_member))).getroot()
                chain = root.find(f"{{{NS}}}CHAIN")
                num_tm = int(chain.get("NUM_TM")) if chain is not None else None
            except (ET.ParseError, TypeError, ValueError):
                num_tm = None
            if num_tm is None:
                continue
            category = classify_tm(num_tm)
            total += 1

            # 1b. PDB 一趟解析（残基数 + pLDDT + Cα 坐标）
            plddt_ca = plddt_all = None
            attrs = {}
            pdb_member = pdb_by_pid.get(pid)
            if pdb_member is None:
                missing_pdb += 1
            else:
                n_res, plddt_ca, plddt_all, coords = parse_pdb_bytes(
                    zf.read(pdb_member))
                if n_res is not None:
                    attrs = {"N_residues": n_res,
                             "NUM_TM": num_tm,
                             "pLDDT_mean_CA": plddt_ca,
                             "pLDDT_mean_all": plddt_all}
                    if len(coords) >= 3:
                        L_xy, L_z, f = geometric_scales(coords)
                        attrs.update(L_xy=round(L_xy, 4),
                                     L_z=round(L_z, 4),
                                     f=round(f, 4))

            csv_rows.append([idx, pid, num_tm, category or "",
                             "" if plddt_ca is None else str(plddt_ca),
                             "" if plddt_all is None else str(plddt_all)])

            # 1c. pLDDT > 70 过滤
            grp_name = CATEGORY_GROUPS.get(category)
            if grp_name is None or plddt_ca is None or plddt_ca <= PLDDT_CUTOFF:
                continue
            grouped[grp_name].append((pid, attrs))
            kept += 1

            if total % 1000 == 0:
                print(f"  [{total}] kept {kept}...")

    print(f"  Total classified: {total}")
    print(f"  Kept (pLDDT > {PLDDT_CUTOFF:g}): {kept}")
    print(f"  PDB missing in zip: {missing_pdb}")

    # ── 2. 写 CSV ─────────────────────────────────────────────────────
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Protein_ID", "NUM_TM", "Category",
                         "pLDDT_mean_CA", "pLDDT_mean_all"])
        writer.writerows(csv_rows)
    print(f"\nCSV: {CSV_PATH}")

    # ── 3. 写 HDF5（per-protein 结构与现 h5 对齐） ────────────────────
    with h5py.File(H5_PATH, "w") as h5f:
        h5f.attrs["description"] = (
            "Transmembrane proteins with pLDDT_mean_CA > 70, grouped by "
            "Category. Rebuilt from UP000005640_9606.zip."
        )
        h5f.attrs["total_entries"] = total
        h5f.attrs["kept_entries"] = kept

        summaries = []
        dt = h5py.string_dtype(encoding="utf-8")
        for grp_name, entries in grouped.items():
            g = h5f.create_group(grp_name)
            ds = g.create_dataset(
                "protein_ids",
                data=np.array([pid for pid, _ in entries], dtype=dt))
            ds.attrs["count"] = len(entries)

            for pid, attrs in entries:
                pg = g.create_group(pid)
                for k, v in attrs.items():
                    pg.attrs[k] = v

            summaries.append(f"  {grp_name:<16} {len(entries):>5}")
        h5f.attrs["summary"] = "\n".join(summaries)

    print(f"H5:  {H5_PATH}")
    print("\nGroups written:")
    for grp_name in CATEGORY_GROUPS.values():
        print(f"  {grp_name:<16} {len(grouped[grp_name]):>5}")


if __name__ == "__main__":
    main()
