#!/usr/bin/env python3
"""
build_suc_pqr.py
================
Build a PQR for the sucrose molecules of a CHARMM36 MD frame, using:
  * charges  -- read directly from SUC_kb.itp [ atoms ] (col 7)
  * radii    -- CHARMM36 Rmin/2 (Angstrom) looked up by CGenFF atom type
                (col 2 of the itp), from par_all36_cgenff.prm NONBONDED

Also writes a protein-only PDB to feed into pdb2pqr.

Charge + type come from ONE file (the itp), so nothing here needs md.tpr and
everything is on the CHARMM scale -- consistent with `pdb2pqr --ff=CHARMM` on
the protein.

Usage
-----
  # all sucrose in the box (4421 molecules):
  python build_suc_pqr.py --gro md_ph73salt30suc12.gro --itp SUC_kb.itp \
      --protein-out capsid_protein.pdb --suc-out sucrose.pqr

  # only sucrose within 10 A of the protein (whole molecules kept):
  python build_suc_pqr.py ... --suc-within 10.0
"""
import argparse
import sys

import MDAnalysis as mda

# CHARMM36 Rmin/2 (Angstrom) by CGenFF atom type -- these ARE the PQR radii.
# Source: par_all36_cgenff.prm NONBONDED section (4th numeric column).
RMIN2 = {
    "CG321": 2.0100,
    "CG311": 2.0000,
    "CG3C50": 2.0100,
    "CG3C51": 2.0100,
    "OG301": 1.6500,
    "OG3C51": 1.6500,
    "OG3C61": 1.6500,
    "OG311": 1.7650,
    "HGA1": 1.3400,
    "HGA2": 1.3400,
    "HGP1": 0.2245,
}


def parse_itp_atoms(itp_path):
    """Return {atom_name: (type, charge)} from the [ atoms ] section.

    Keyed by atom NAME (not position) so the mapping is invariant to how the
    itp -- or the frame -- orders its atoms. Names are unique within a sucrose.
    """
    atoms = {}
    in_atoms = False
    with open(itp_path) as fh:
        for raw in fh:
            line = raw.split(";", 1)[0].strip()   # drop comments
            if not line:
                continue
            if line.startswith("["):
                in_atoms = line.replace(" ", "").lower() == "[atoms]"
                continue
            if in_atoms:
                tok = line.split()
                # nr type resnr residu atom cgnr charge mass
                atype, name, charge = tok[1], tok[4], float(tok[6])
                if name in atoms:
                    sys.exit(f"ERROR: duplicate atom name {name!r} in itp -- "
                             f"name-based matching requires unique names.")
                atoms[name] = (atype, charge)
    return atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gro", required=True)
    ap.add_argument("--itp", required=True)
    ap.add_argument("--protein-out", default="capsid_protein.pdb")
    ap.add_argument("--suc-out", default="sucrose.pqr")
    ap.add_argument("--suc-resname", default="SUC")
    ap.add_argument("--suc-within", type=float, default=None,
                    help="keep only sucrose molecules with any atom within "
                         "this many A of the protein (default: keep all)")
    args = ap.parse_args()

    itp_atoms = parse_itp_atoms(args.itp)          # {name: (type, charge)}
    n_template = len(itp_atoms)
    net_template = sum(q for _, q in itp_atoms.values())
    print(f"itp template: {n_template} atoms, net charge "
          f"{net_template:+.4f} e", file=sys.stderr)

    missing = sorted({t for t, _ in itp_atoms.values() if t not in RMIN2})
    if missing:
        sys.exit(f"ERROR: no Rmin/2 for atom type(s): {missing} "
                 f"-- add them to RMIN2.")
    # name -> (charge, radius)
    param = {name: (q, RMIN2[t]) for name, (t, q) in itp_atoms.items()}

    u = mda.Universe(args.gro)   # MDAnalysis loads coords in Angstrom

    # protein -> PDB for pdb2pqr
    prot = u.select_atoms("protein")
    prot.write(args.protein_out)
    print(f"protein: {prot.n_atoms} atoms, {len(prot.residues)} residues, "
          f"{len(prot.segments)} segments -> {args.protein_out}",
          file=sys.stderr)

    # sucrose selection
    if args.suc_within is not None:
        sel = (f"byres (resname {args.suc_resname} and around "
               f"{args.suc_within} protein)")
    else:
        sel = f"resname {args.suc_resname}"
    suc = u.select_atoms(sel)
    n_mol = len(suc.residues)
    print(f"sucrose: {suc.n_atoms} atoms, {n_mol} molecules "
          f"(selection: {sel})", file=sys.stderr)

    template_names = set(param)
    net = 0.0
    with open(args.suc_out, "w") as fh:
        serial = 0
        for resid_out, res in enumerate(suc.residues, start=1):
            atoms = res.atoms
            frame_names = [a.name for a in atoms]
            if set(frame_names) != template_names:
                extra = set(frame_names) - template_names
                absent = template_names - set(frame_names)
                sys.exit(f"ERROR: SUC residue {res.resid} atom names do not "
                         f"match the itp. unexpected={sorted(extra)} "
                         f"missing={sorted(absent)}.")
            for a in atoms:
                serial += 1
                x, y, z = a.position
                q, r = param[a.name]        # matched BY NAME, order-independent
                net += q
                # free (whitespace-delimited) PQR: no chainID token, so APBS
                # reads the numeric resid as residue number. Handles >99999
                # serials fine.
                fh.write(f"ATOM {serial} {a.name} {args.suc_resname} "
                         f"{resid_out} {x:.3f} {y:.3f} {z:.3f} "
                         f"{q:.4f} {r:.4f}\n")

    print(f"sucrose PQR net charge: {net:+.4f} e (expect ~0) -> {args.suc_out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
