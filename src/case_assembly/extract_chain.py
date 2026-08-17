#!/usr/bin/env python3
"""extract_chain.py - Extract one (or more) chain(s) from a PDB file into a new PDB file.

Usage:
    python extract_chain.py [INPUT.pdb] [CHAIN] [-o OUTPUT.pdb] [options]

Defaults:
    input = /Volumes/x23/临时/rdDFI/data/raw/case_assembly/3ODU.pdb, chains = A,
    output = <input dir>/<stem>_chain<IDs>.pdb

Examples:
    python extract_chain.py                  # 3ODU chain A -> data/raw/case_assembly/3ODU_chainA.pdb
    python extract_chain.py 3ODU.pdb A -o 3ODU_chainA.pdb
    python extract_chain.py 3ODU.pdb A,B,C -o 3ODU_ABC.pdb   # multiple chains
    python extract_chain.py 3ODU.pdb A --no-water --no-hetatm  # protein only

Notes:
- Chain ID is read from column 22 (0-indexed 21) of ATOM/HETATM/ANISOU records.
- If the file has multiple MODELs, only the first model is written out.
- Atom serial numbers are preserved from the source file (not renumbered).
- A valid TER record is generated for each contiguous chain block; END is appended.
- CONECT records are optionally kept only if every referenced serial belongs to a
  kept atom (so ligand connectivity stays correct for the extracted chain).
"""
import argparse
import os
import sys


# Records that are NOT chain-specific and can be copied verbatim as context.
HEADER_RECORDS = {
    "HEADER", "TITLE ", "KEYWDS", "EXPDTA", "AUTHOR",
    "CRYST1", "SCALE1", "SCALE2", "SCALE3",
    "ORIGX1", "ORIGX2", "ORIGX3", "MTRIX1", "MTRIX2", "MTRIX3",
}
ATOM_RECORDS = {"ATOM  ", "HETATM", "ANISOU"}


def _is_atom_like(rec):
    return rec in ATOM_RECORDS


def _chain_of(line):
    # chain ID lives at column 22 (index 21)
    return line[21] if len(line) > 21 else " "


def _build_ter(last_atom_line):
    """Build a properly formatted TER record from the last ATOM/HETATM line."""
    serial = int(last_atom_line[6:11]) + 1
    resname = last_atom_line[17:20]
    chain = last_atom_line[21]
    resseq = last_atom_line[22:26]
    icode = last_atom_line[26] if len(last_atom_line) > 26 else " "
    return "TER   " + f"{serial:>5}" + "      " + f"{resname:>3}" + " " + chain + f"{resseq:>4}" + icode + "\n"


def extract_chains(input_pdb, chains, output_pdb,
                   keep_header=True, keep_hetatm=True, keep_water=True,
                   keep_conect=False):
    """Extract the requested chain(s) from input_pdb into output_pdb.

    Parameters
    ----------
    chains : iterable of single-character chain IDs (spaces allowed).
    """
    wanted = set(chains)
    kept_atom_lines = []
    kept_serials = set()
    header_lines = []
    conect_lines = []
    in_first_model = True
    saw_model = False

    with open(input_pdb, "r") as fh:
        for line in fh:
            rec = line[:6]
            # Stop at the end of the first model (handles multi-MODEL NMR files).
            if rec == "MODEL ":
                saw_model = True
                continue
            if rec == "ENDMDL":
                if in_first_model:
                    in_first_model = False
                    break
                continue
            if rec == "END   " or line.startswith("END"):
                break

            if _is_atom_like(rec):
                if not in_first_model:
                    continue
                ch = _chain_of(line)
                if ch not in wanted:
                    continue
                # Filtering by record type / water.
                if rec == "HETATM":
                    if not keep_hetatm:
                        continue
                    resname = line[17:20].strip()
                    if (not keep_water) and resname in ("HOH", "WAT", "DOD"):
                        continue
                kept_atom_lines.append(line)
                if rec in ("ATOM  ", "HETATM"):
                    try:
                        kept_serials.add(int(line[6:11]))
                    except ValueError:
                        pass
            elif keep_header and rec in HEADER_RECORDS:
                header_lines.append(line)
            elif rec == "CONECT" and keep_conect:
                conect_lines.append(line)

    if not kept_atom_lines:
        raise ValueError(f"No atoms found for chain(s) {sorted(wanted)} in {input_pdb}")

    # Build TER records: insert a TER after each chain block (when chain changes)
    # and at the very end.
    out_lines = list(header_lines)
    prev_chain = None
    last_atom = None
    for line in kept_atom_lines:
        ch = _chain_of(line)
        if prev_chain is not None and ch != prev_chain:
            out_lines.append(_build_ter(last_atom))
        out_lines.append(line)
        prev_chain = ch
        if line[:6] in ("ATOM  ", "HETATM"):
            last_atom = line
    if last_atom is not None:
        out_lines.append(_build_ter(last_atom))

    # Optionally keep CONECT records whose every referenced serial is kept.
    if keep_conect and conect_lines:
        for line in conect_lines:
            try:
                serials = [int(line[i:i + 5]) for i in range(6, len(line.rstrip("\n")), 5)
                           if line[i:i + 5].strip()]
            except ValueError:
                continue
            if serials and all(s in kept_serials for s in serials):
                out_lines.append(line)

    out_lines.append("END\n")

    with open(output_pdb, "w") as out:
        out.writelines(out_lines)

    # Summary
    n_atoms = sum(1 for l in kept_atom_lines if l[:6] in ("ATOM  ", "HETATM"))
    n_anisou = sum(1 for l in kept_atom_lines if l[:6] == "ANISOU")
    chains_found = sorted({c for c in (_chain_of(l) for l in kept_atom_lines) if c.strip()})
    return {
        "output": output_pdb,
        "chains_written": chains_found,
        "atom_records": n_atoms,
        "anisou_records": n_anisou,
        "header_records": len(header_lines),
        "conect_records": sum(1 for l in out_lines if l.startswith("CONECT")),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Extract chain(s) from a PDB file.")
    p.add_argument("input", nargs="?", default="/Volumes/x23/临时/rdDFI/data/raw/case_assembly/4dkl.pdb",
                   help="input PDB file (default: /Volumes/x23/临时/rdDFI/data/raw/case_assembly/4dkl.pdb)")
    p.add_argument("chains", nargs="?", default="A",
                   help="chain ID(s), comma-separated (e.g. A or A,B,C; default: A)")
    p.add_argument("-o", "--output", help="output PDB file (default: <input dir>/<stem>_chain<IDs>.pdb)")
    p.add_argument("--no-header", action="store_true", help="drop header/crystallographic records")
    p.add_argument("--no-hetatm", action="store_true", help="drop HETATM (ligands, glycans, waters)")
    p.add_argument("--no-water", action="store_true", help="drop water (HOH/WAT/DOD) HETATM only")
    p.add_argument("--keep-conect", action="store_true", help="keep CONECT records for kept atoms")
    args = p.parse_args(argv)

    chain_list = [c if c != "" else " " for c in args.chains.split(",")]
    if args.output:
        output = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output = os.path.join(os.path.dirname(args.input),
                              f"{stem}_chain{''.join(c.strip() for c in chain_list)}.pdb")

    info = extract_chains(
        args.input, chain_list, output,
        keep_header=not args.no_header,
        keep_hetatm=not args.no_hetatm,
        keep_water=not args.no_water,
        keep_conect=args.keep_conect,
    )
    print(f"Wrote {info['output']}")
    print(f"  chains written : {info['chains_written']}")
    print(f"  ATOM/HETATM    : {info['atom_records']}")
    print(f"  ANISOU         : {info['anisou_records']}")
    print(f"  header records : {info['header_records']}")
    print(f"  CONECT records : {info['conect_records']}")


if __name__ == "__main__":
    main()
