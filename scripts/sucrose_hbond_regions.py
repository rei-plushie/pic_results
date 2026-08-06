#!/usr/bin/env python3
"""
sucrose_hbond_regions.py
========================

Hydrogen-bonding of the excipient (default sucrose, resname SUC) to the AAV
capsid, resolved by icosahedral symmetry region (3-fold vs 5-fold). For each
region it reports the three quantities that matter here:

  1. COUNT   -- how many sucrose<->protein H-bonds that region carries, as a
                mean instantaneous count per frame, and normalized per residue
                so 3F (many residues) and 5F (few) are compared fairly.
  2. LENGTH  -- the H-bond length distribution (donor--acceptor heavy-atom
                distance, the standard "H-bond length"): mean/median/std and a
                binned histogram per region.
  3. RESIDUE -- which protein residue types (and which of their atoms) donate
     TYPES     to / accept from sucrose most often, per region, and whether the
                protein or the sucrose is the donor.

TOPOLOGY CHOICE -- IMPORTANT
----------------------------
Use the water-stripped **.gro** (md_prot_suc.gro), NOT the .tpr, as --top.

The region labels come from `pic_symmetry_regions/per_residue.csv`, which is
keyed on (chain, resid) with per-chain residue numbering (chain 0..59, resid
219..). Chains here are found the same way the rest of the pipeline finds them
-- the resid-reset signal (`resid_reset_chain_key`): a new chain begins wherever
resid drops. A bare .gro preserves that per-chain numbering, so chains AND the
(chain,resid) join line up exactly.

A GROMACS **.tpr renumbers residues continuously** across all 60 chains, which
(a) makes the resid-reset signal fail -- "one giant chain" -- and (b) makes
resids no longer match the region CSV. So the .tpr does not work with the
region join. The .gro's only cost is that bonds aren't stored; this script
guesses them on load (distance-based, reliable for standard-residue H-bonds).

PREPROCESSING (on the cluster) -- see prep_hbond_inputs.sh, which writes
md_prot_suc.gro and md_prot_suc.xtc (water stripped, already PBC-whole).

OUTPUTS (--output-dir)
----------------------
  hbonds_annotated.parquet   one row per (frame, H-bond): frame, suc_molid,
                             suc_atom, prot_chain, prot_resid, prot_resname,
                             prot_atom, region_type, facing_outer, length,
                             angle, direction (suc_donor / prot_donor)
  region_summary.csv         per region_type: n_hbond_rows, mean per frame,
                             n_region_residues, hbonds_per_frame_per_100res,
                             length mean/median/std, % under 3.0 A
  length_histogram.csv       H-bond length counts in fixed bins, per region
  residue_type_summary.csv   per (region_type, prot_resname): count, share of
                             region, mean length, #suc_donor vs #prot_donor
  atom_type_summary.csv      per (region_type, prot_resname, prot_atom):
                             count, mean length  (functional-group detail)
  summary.txt                plain-text digest

Smoke-test:  --max-frames 5   (confirm selections non-empty, region join matches).
Array jobs:  --start/--stop over frame chunks, then concatenate parquet shards.

Usage
-----
  python3 sucrose_hbond_regions.py \\
      --top md_prot_suc.gro --xtc md_prot_suc.xtc \\
      --region-csv pic_symmetry_regions/per_residue.csv \\
      --excipient SUC --output-dir hbond_out
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as HBA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aav.scripts.pic_calculation_exterior import resid_reset_chain_key  # noqa: E402

# H-bond length histogram edges (Angstrom, donor-acceptor distance)
LEN_BINS = np.arange(2.4, 3.61, 0.05)


def setup_logger():
    logger = logging.getLogger("sucrose_hbond_regions")
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logger.handlers.clear()
    logger.addHandler(h)
    return logger


def ensure_bonds(u, logger):
    """H-bond donor/acceptor guessing needs a bond graph. A .tpr already has
    one; a bare .gro does not, so guess it (distance-based). guess_bonds uses
    vdW-radius sums * fudge factor, so it will NOT create spurious covalent
    bonds at H-bond distances (~2.8 A O-O is well beyond the ~1.7 A threshold)."""
    if hasattr(u, "bonds") and len(u.bonds) > 0:
        logger.info("topology already has %d bonds", len(u.bonds))
        return
    logger.info("no bonds in topology (bare .gro) -- guessing bonds by distance ...")
    u.atoms.guess_bonds()
    logger.info("guessed %d bonds", len(u.bonds))


def build_chain_lookup(u, protein, logger):
    """Per-(global atom index) chain id for protein atoms, -1 elsewhere,
    from the same resid-reset signal the rest of the pipeline uses."""
    chain_key = resid_reset_chain_key(protein)
    if chain_key is None:
        raise SystemExit(
            "resid-reset chain signal unusable (resid never decreases).\n"
            "  This almost always means you passed a .tpr as --top: GROMACS "
            "renumbers residues continuously across all chains, which breaks "
            "both chain detection AND the (chain,resid) join to the region CSV.\n"
            "  FIX: pass the water-stripped .gro (md_prot_suc.gro) as --top "
            "instead -- it keeps per-chain numbering that matches the region CSV.")
    n_chains = int(chain_key.max()) + 1
    logger.info("resid-reset signal found %d protein chains", n_chains)
    if n_chains != 60:
        logger.warning("expected 60 VP chains for a T=1 capsid, found %d", n_chains)
    lut = np.full(len(u.atoms), -1, dtype=np.int32)
    lut[protein.atoms.ix] = chain_key
    return lut, n_chains


def default_selections(exc, protein_sel):
    """Geometric (name-based) donor-hydrogen and acceptor selections.

    MDAnalysis's built-in guess_hydrogens/guess_acceptors pick atoms BY PARTIAL
    CHARGE, which a bare .gro does not carry (-> NoDataError). Instead we define
    them purely by name, the standard charge-free H-bond convention:
      - donor hydrogens: any H covalently bonded to an O or N (polar H); needs
        the guessed bond graph (see ensure_bonds). Aliphatic C-H is excluded.
      - acceptors: O and N heavy atoms. (Ions are excluded by restricting to
        protein+excipient; the `between` filter would drop them anyway.)
    Amide backbone N is a weak acceptor and is included here for simplicity;
    tighten with --acceptors-sel '... and name O*' for oxygen-only acceptors."""
    grp = f"({protein_sel} or resname {exc})"
    hydrogens = f"{grp} and (name H*) and (bonded (name O* or name N*))"
    acceptors = f"{grp} and (name O* or name N*)"
    return hydrogens, acceptors


def run_hba(u, sel_a, sel_b, hydrogens_sel, acceptors_sel,
            d_a_cutoff, angle_cutoff, start, stop, step, logger):
    n_h = len(u.select_atoms(hydrogens_sel))
    n_a = len(u.select_atoms(acceptors_sel))
    logger.info("donor-hydrogens: %d atoms | acceptors: %d atoms", n_h, n_a)
    if n_h == 0 or n_a == 0:
        raise SystemExit("hydrogens or acceptors selection is empty -- check that "
                         "bonds were guessed (ensure_bonds) and the atom-name "
                         "patterns match your force field (see --hydrogens-sel/"
                         "--acceptors-sel).")
    hba = HBA(universe=u, between=[[sel_a, sel_b]],
              donors_sel=None, hydrogens_sel=hydrogens_sel, acceptors_sel=acceptors_sel,
              d_a_cutoff=d_a_cutoff, d_h_a_angle_cutoff=angle_cutoff,
              update_selections=False)
    hba.run(start=start, stop=stop, step=step, verbose=True)
    return hba.results.hbonds


def annotate(hbonds, u, chain_lut, exc_resname):
    """HBA rows [frame, d_ix, h_ix, a_ix, dist(D-A), angle] -> annotated frame.
    `between=` guarantees one heavy end is excipient, the other protein."""
    names = u.atoms.names
    resnames = u.atoms.resnames
    resids = u.atoms.resids
    resindices = u.atoms.resindices

    frame = hbonds[:, 0].astype(int)
    d_ix = hbonds[:, 1].astype(int)
    a_ix = hbonds[:, 3].astype(int)
    donor_is_exc = resnames[d_ix] == exc_resname
    exc_ix = np.where(donor_is_exc, d_ix, a_ix)
    prot_ix = np.where(donor_is_exc, a_ix, d_ix)

    return pd.DataFrame({
        "frame": frame,
        "suc_molid": resindices[exc_ix],
        "suc_atom": names[exc_ix],
        "prot_chain": chain_lut[prot_ix],
        "prot_resid": resids[prot_ix],
        "prot_resname": resnames[prot_ix],
        "prot_atom": names[prot_ix],
        "length": hbonds[:, 4],          # donor-acceptor distance, Angstrom
        "angle": hbonds[:, 5],
        "direction": np.where(donor_is_exc, "suc_donor", "prot_donor"),
    })


def attach_region(df, region_csv, logger):
    """Join region_type/facing_outer onto each H-bond via (chain, resid)."""
    reg = pd.read_csv(region_csv)
    reg = reg[["chain", "resid", "region_type", "facing_outer"]].copy()
    reg["region_type"] = reg["region_type"].fillna("").replace("", "neither")
    merged = df.merge(reg, how="left",
                      left_on=["prot_chain", "prot_resid"],
                      right_on=["chain", "resid"]).drop(columns=["chain", "resid"])
    unmatched = merged["region_type"].isna().mean() if len(merged) else 0.0
    merged["region_type"] = merged["region_type"].fillna("unmatched")
    if unmatched > 0.02:
        logger.warning("%.1f%% of H-bonded residues did not match the region map "
                       "-- chain-id/resid mismatch. Are you using the .gro (not "
                       ".tpr) and the region CSV from THIS capsid?", 100 * unmatched)
    else:
        logger.info("region join: %.2f%% unmatched", 100 * unmatched)
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", required=True, help="Topology -- use the .gro (see docstring; .tpr breaks the region join)")
    p.add_argument("--xtc", required=True)
    p.add_argument("--region-csv", required=True,
                   help="pic_symmetry_regions/per_residue.csv for THIS run")
    p.add_argument("--excipient", default="SUC")
    p.add_argument("--protein-selection", default="protein")
    p.add_argument("--d-a-cutoff", type=float, default=3.5, help="Donor-acceptor cutoff, Angstrom (default 3.5)")
    p.add_argument("--angle-cutoff", type=float, default=150.0, help="D-H-A angle cutoff, degrees (default 150)")
    p.add_argument("--hydrogens-sel", default=None,
                   help="Override donor-hydrogen selection (default: polar H on protein+excipient)")
    p.add_argument("--acceptors-sel", default=None,
                   help="Override acceptor selection (default: O/N on protein+excipient)")
    p.add_argument("--outer-only", action="store_true",
                   help="Restrict to facing_outer protein residues (exterior-accessible only)")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None, help="Smoke-test cap")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    logger = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    u = mda.Universe(args.top, args.xtc)
    ensure_bonds(u, logger)
    protein = u.select_atoms(args.protein_selection)
    exc = u.select_atoms(f"resname {args.excipient}")
    logger.info("protein %d atoms | %s %d atoms | %d frames",
                len(protein), args.excipient, len(exc), len(u.trajectory))
    if len(protein) == 0 or len(exc) == 0:
        raise SystemExit("empty protein or excipient selection -- check names / topology.")

    start, stop, step = args.start, args.stop, args.stride
    if args.max_frames is not None:
        stop = (start or 0) + args.max_frames * step
    n_frames = len(range(start or 0,
                         stop if stop is not None else len(u.trajectory), step))

    chain_lut, _ = build_chain_lookup(u, protein, logger)
    hyd_def, acc_def = default_selections(args.excipient, args.protein_selection)
    hydrogens_sel = args.hydrogens_sel or hyd_def
    acceptors_sel = args.acceptors_sel or acc_def
    raw = run_hba(u, f"resname {args.excipient}", args.protein_selection,
                  hydrogens_sel, acceptors_sel,
                  args.d_a_cutoff, args.angle_cutoff, start, stop, step, logger)
    df = annotate(raw, u, chain_lut, args.excipient)
    df = attach_region(df, args.region_csv, logger)
    if args.outer_only:
        df = df[df["facing_outer"] == True]  # noqa: E712
        logger.info("outer-only: %d H-bonds retained", len(df))
    df.to_parquet(os.path.join(args.output_dir, "hbonds_annotated.parquet"), index=False)

    # residues per region (denominator for the normalized count), same filter
    reg = pd.read_csv(args.region_csv)
    reg["region_type"] = reg["region_type"].fillna("").replace("", "neither")
    if args.outer_only:
        reg = reg[reg["facing_outer"] == True]  # noqa: E712
    res_per_region = reg.groupby("region_type").size()

    # ---- 1+2. per-region count and length ----
    rows = []
    for rt, g in df.groupby("region_type"):
        nres = int(res_per_region.get(rt, 0))
        per_frame = len(g) / max(n_frames, 1)
        rows.append({
            "region_type": rt,
            "n_hbond_rows": len(g),
            "hbonds_per_frame": per_frame,
            "n_region_residues": nres,
            "hbonds_per_frame_per_100res": 100 * per_frame / nres if nres else np.nan,
            "length_mean": g["length"].mean(),
            "length_median": g["length"].median(),
            "length_std": g["length"].std(),
            "frac_under_3.0A": (g["length"] < 3.0).mean(),
        })
    pd.DataFrame(rows).sort_values("region_type").to_csv(
        os.path.join(args.output_dir, "region_summary.csv"), index=False)

    # length histogram per region
    hist_rows = []
    centers = (LEN_BINS[:-1] + LEN_BINS[1:]) / 2
    for rt, g in df.groupby("region_type"):
        counts, _ = np.histogram(g["length"].values, bins=LEN_BINS)
        for c, n in zip(centers, counts):
            hist_rows.append({"region_type": rt, "length_center_A": round(float(c), 3), "count": int(n)})
    pd.DataFrame(hist_rows).to_csv(
        os.path.join(args.output_dir, "length_histogram.csv"), index=False)

    # ---- 3. residue-type involvement, per region ----
    rt_rows = []
    for (rt, rn), g in df.groupby(["region_type", "prot_resname"]):
        rt_rows.append({
            "region_type": rt, "prot_resname": rn, "count": len(g),
            "share_of_region": len(g) / len(df[df["region_type"] == rt]),
            "length_mean": g["length"].mean(),
            "n_suc_donor": int((g["direction"] == "suc_donor").sum()),
            "n_prot_donor": int((g["direction"] == "prot_donor").sum()),
        })
    (pd.DataFrame(rt_rows).sort_values(["region_type", "count"], ascending=[True, False])
       .to_csv(os.path.join(args.output_dir, "residue_type_summary.csv"), index=False))

    # functional-group detail: which protein atom
    (df.groupby(["region_type", "prot_resname", "prot_atom"])
       .agg(count=("length", "size"), length_mean=("length", "mean"))
       .reset_index().sort_values(["region_type", "count"], ascending=[True, False])
       .to_csv(os.path.join(args.output_dir, "atom_type_summary.csv"), index=False))

    # ---- digest ----
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(f"topology   : {args.top}\ntrajectory : {args.xtc}\n")
        fh.write(f"excipient  : {args.excipient} | outer_only={args.outer_only}\n")
        fh.write(f"frames analyzed (start={start} stop={stop} stride={step}): {n_frames}\n")
        fh.write(f"cutoffs    : {args.d_a_cutoff} A (D-A), {args.angle_cutoff} deg (D-H-A)\n\n")
        for r in sorted(rows, key=lambda x: x["region_type"]):
            fh.write(f"[{r['region_type']}] {r['hbonds_per_frame']:.2f} H-bonds/frame "
                     f"({r['hbonds_per_frame_per_100res']:.3f} per 100 residues), "
                     f"mean length {r['length_mean']:.2f} A\n")
            top = (df[df["region_type"] == r["region_type"]]["prot_resname"]
                   .value_counts().head(5))
            fh.write("        top residue types: " +
                     ", ".join(f"{k}({v})" for k, v in top.items()) + "\n")
    logger.info("done -> %s", args.output_dir)


if __name__ == "__main__":
    main()