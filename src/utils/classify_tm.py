#!/usr/bin/env python3
"""
Parse TMDet XML files and classify proteins by their NUM_TM value.

Reads all *_tmdet.xml entries from the UP000005640_9606 zip, extracts the
NUM_TM attribute from the <CHAIN> element, classifies each protein into a
category, and outputs a CSV summary.
"""

import csv
import io
import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
XML_ZIP  = PROJECT_DIR / "data" / "raw" / "UP000005640_9606.zip"
OUT_FILE = PROJECT_DIR / "data" / "process" / "tm_classification.csv"

# XML namespace used in PDBTM files
NS = "http://pdbtm.enzim.hu"


def classify_tm(num_tm: int) -> str:
    """Classify a protein based on its number of transmembrane helices."""
    if num_tm == 1:
        return "single-pass transmembrane"
    elif 2 <= num_tm <= 6:
        return "medium transmembrane"
    elif num_tm == 7:
        return "GPCR transmembrane"
    elif 8 <= num_tm:
        return "large transmembrane"


def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Collect all *_tmdet.xml entries in the zip, excluding macOS junk
    zf = zipfile.ZipFile(XML_ZIP)
    xml_members = sorted(
        n for n in zf.namelist()
        if n.endswith("_tmdet.xml")
        and not os.path.basename(n).startswith("._")
        and not n.startswith("__MACOSX")
    )

    print(f"Found {len(xml_members)} XML files.")

    results = []

    for idx, member in enumerate(xml_members, start=1):
        filename = os.path.basename(member)
        protein_id = filename.replace("_tmdet.xml", "")

        try:
            tree = ET.parse(io.BytesIO(zf.read(member)))
            root = tree.getroot()

            # Find the <CHAIN> element with namespace
            chain = root.find(f"{{{NS}}}CHAIN")
            if chain is None:
                print(f"  ⚠  No CHAIN element in {filename}, skipping.")
                continue

            num_tm_str = chain.get("NUM_TM")
            if num_tm_str is None:
                print(f"  ⚠  No NUM_TM attribute in {filename}, skipping.")
                continue

            num_tm = int(num_tm_str)
            category = classify_tm(num_tm)

            results.append([idx, protein_id, num_tm, category])

        except ET.ParseError:
            print(f"  ✗ XML parse error in {filename}, skipping.")
        except Exception as e:
            print(f"  ✗ Unexpected error in {filename}: {e}, skipping.")

    zf.close()

    # Write CSV
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Protein_ID", "NUM_TM", "Category"])
        writer.writerows(results)

    print(f"\nDone! {len(results)} proteins classified.")
    print(f"Output: {OUT_FILE}")

    # Print summary statistics
    from collections import Counter
    category_counts = Counter(r[3] for r in results)
    print("\nClassification summary:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
