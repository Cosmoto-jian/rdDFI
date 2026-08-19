#!/usr/bin/env python3
"""
Calculate mean pLDDT scores from AlphaFold PDB files and append to the
existing tm_classification.csv.

For each protein:
  - pLDDT_mean_CA:  per-residue mean pLDDT (CA atoms only)
  - pLDDT_mean_all: overall mean pLDDT (all ATOM records)

pLDDT values are read from the B-factor column (cols 61-66) of ATOM lines.
"""

import csv
import gzip
import os
import sys
import zipfile
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
PDB_ZIP = PROJECT_DIR / "data" / "raw" / "UP000005640_9606.zip"
CSV_IN  = PROJECT_DIR / "data" / "process" / "tm_classification.csv"
CSV_OUT = PROJECT_DIR / "data" / "process" / "tm_classification.csv"  # overwrite


def calc_plddt(text: str):
    """从解压后的 PDB 文本计算 pLDDT 均值（CA 与全体原子）。

    Returns (plddt_mean_ca, plddt_mean_all) or (None, None) on failure.
    """
    ca_sum = 0.0
    ca_count = 0
    all_sum = 0.0
    all_count = 0

    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        # B-factor is in columns 61-66 (1-indexed in PDB format)
        # Column indices in Python string: 60:66
        try:
            b_factor = float(line[60:66])
        except ValueError:
            continue

        all_sum += b_factor
        all_count += 1

        # CA atom: atom name at cols 13-16, stripped
        atom_name = line[12:16].strip()
        if atom_name == "CA":
            ca_sum += b_factor
            ca_count += 1

    if ca_count == 0 or all_count == 0:
        return None, None

    return round(ca_sum / ca_count, 4), round(all_sum / all_count, 4)


def main():
    # Read existing CSV
    rows = []
    with open(CSV_IN, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    total = len(rows)
    print(f"Loaded {total} entries from {CSV_IN}")

    # Add new column headers
    header.append("pLDDT_mean_CA")
    header.append("pLDDT_mean_all")

    # Process each protein (PDB 从 zip 内读取)
    zf = zipfile.ZipFile(PDB_ZIP)
    members = {n.rsplit("/", 1)[-1]: n for n in zf.namelist()}

    for i, row in enumerate(rows):
        protein_id = row[1]  # Protein_ID is column 2 (0-indexed: 1)
        member = members.get(f"{protein_id}_tr.pdb.gz")

        if member is None:
            print(f"  ⚠  [{i+1}/{total}] {protein_id}: PDB not found in zip, skipping.")
            row.append("")
            row.append("")
            continue

        text = gzip.decompress(zf.read(member)).decode("utf-8")
        ca_mean, all_mean = calc_plddt(text)

        if ca_mean is not None:
            row.append(str(ca_mean))
            row.append(str(all_mean))
        else:
            row.append("")
            row.append("")

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] processed...")

    zf.close()

    # Write updated CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nDone! Written {len(rows)} entries to {CSV_OUT}")
    print(f"New columns: pLDDT_mean_CA, pLDDT_mean_all")


if __name__ == "__main__":
    main()
