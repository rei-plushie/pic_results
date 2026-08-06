#!/usr/bin/env python3
"""
pic_symmetry_regions.py
========================

Averages the per-residue preferential interaction coefficient (Gamma_23,
"PIC") from a pic_calculation_exterior.py output over the icosahedral
3-fold and 5-fold symmetry regions of an AAV capsid, restricted to
surface-exposed residues (facing_outer == True in the .pkl).

WHY GEOMETRY HAS TO BE (RE)COMPUTED
------------------------------------
pic_calculation_exterior.py's output has no notion of icosahedral symmetry
-- it only knows per-residue Gamma_23 and whether a residue faces the
exterior or the lumen. AAV (and every other T=1 icosahedral capsid) is
built from 60 identical protein chains, related to each other by the
icosahedral rotation group: 12 pentamers around the 5-fold axes, 20 trimers
around the 3-fold axes, 30 dimers around the 2-fold axes. This script
rediscovers that structure directly from the capsid's own 3D coordinates
(same .gro used for the PIC run) rather than assuming any literature
residue numbering, so it works unchanged for any T=1 capsid/serotype:

  1. Group the protein into its 60 chains (same resid-reset signal
     pic_calculation_exterior.py itself uses for chain-level PBC
     realignment -- see resid_reset_chain_key there).
  2. Find which chains are DIRECT pentamer/trimer neighbors of each other
     via pairwise structural alignment (Kabsch) of each pair of chains'
     CA atoms (all 60 chains share an identical residue sequence, so the
     atom correspondence is exact). The best-fit rotation between chain i
     and chain j reveals the icosahedral group element relating them --
     its ANGLE alone is not enough to identify neighbors (e.g. many
     non-neighboring chain pairs are coincidentally related by some
     unrelated 72 degree rotation about a distant axis -- confirmed
     empirically: ~360 pairs fall near 72 degree magnitude, 6x more than
     the 60 true pentamer-adjacent pairs), so a pair only counts as a
     genuine pentamer/trimer neighbor if the rotation's AXIS ALSO passes
     close (<30 degrees) to both chains' own centroid directions --
     i.e. the axis is actually near where both chains physically sit.
     Confirmed on this system: this reduces ~360 angle-only 72/144-degree
     candidates to exactly 120 axis-local pairs (=12 pentamers x
     C(5,2)=10) and ~600 120-degree candidates to exactly 60 (=20 trimers
     x C(3,2)=3) -- connected components of that filtered adjacency graph
     then give exactly 12 components of 5 chains (pentamers) and 20
     components of 3 chains (trimers), with no manual tuning of cluster
     COUNT needed (unlike clustering directly on the 60 chain-centroid
     directions, which does NOT separate cleanly: a subunit's own center
     of mass spans much of its asymmetric unit rather than sitting near
     just its own 5-fold axis, so raw centroid-distance clustering was
     tried first and confirmed to produce uneven, incorrect groups).
  3. Each axis direction = the mean (renormalized) direction of its
     cluster's members' centroids from the capsid centroid.
  4. Every surface-exposed (facing_outer) residue is assigned to whichever
     axis -- among all 12 five-fold and 20 three-fold axes -- it is
     angularly closest to, PROVIDED that angle is within
     --max-axis-angle-deg (default 20 deg, roughly half the ~37-41 deg
     gap between neighboring axis families on a regular icosahedron).
     Residues that aren't close to either axis family (e.g. sitting near
     an unrepresented 2-fold axis, or on a flat region between axes) are
     left unclassified rather than force-assigned -- this keeps "the
     5-fold region" and "the 3-fold region" meaning what they should:
     genuinely axis-proximal patches, not an arbitrary 100% partition of
     the whole surface.
  5. Optionally, ALSO require the residue's radial distance from the
     capsid centroid to fall within a --five-fold-r-min/max or
     --three-fold-r-min/max band (each an offset in A from this run's own
     shell_outer_radius_angstrom, so it transfers across serotypes of
     slightly different size) -- angle alone doesn't distinguish "near the
     3-fold axis at the base of the spike" from "near the 3-fold axis at
     the spike's tip," since both sit at the same angular position but
     very different radii. Unrestricted by default (matches the pre-
     radial-band behavior). Every residue's radial distance is written to
     per_residue.csv (radial_angstrom) regardless of whether a band is
     set, specifically so a past run's output can be used to pick sensible
     band values for the next one -- see EXAMPLE below.

A diagnostic plot/CSV of the discovered axes and chain-cluster membership
is always written -- check it before trusting the region assignment, the
same way pic_calculation_exterior.py's own shell-density plot should be
checked before trusting interior/exterior classification.

EXAMPLE: narrowing both families and shifting 3-fold outward
--------------------------------------------------------------
On the AAV1 system this module was developed against, an unrestricted run
gave 5-fold residues spanning 108-127 A from the centroid (median 113) and
3-fold spanning 108-148 A (median 121) -- i.e. 3-fold reached all the way
down to the SAME inner boundary as 5-fold (generic shell floor, not
distinctively 3-fold) as well as out to the spike tips. Tightening both
and pushing 3-fold's band outward, relative to the run's own
shell_outer_radius_angstrom (103 A here, so e.g. +5 means "108 A"):

    python3 pic_symmetry_regions.py --run SUC:pic_SUC_exterior.pkl:md.gro \\
        --five-fold-r-min 5 --five-fold-r-max 16 \\
        --three-fold-r-min 20 --three-fold-r-max 45

Inspect regions_bfactor.pdb and the radial_angstrom column of a prior
per_residue.csv (or re-run once unrestricted first) before picking your
own numbers -- these are this system's actual percentiles, not universal
constants, and will differ for a different serotype or a different
--radius used upstream in pic_calculation_exterior.py.

INPUT
-----
One or more PIC runs, each given as:
    --run LABEL:PKL:GRO[:PROTEIN_SELECTION]
where PKL is a pic_<EXCIPIENT>_exterior.pkl (must contain the
"facing_outer" per-residue field -- i.e. produced by
pic_calculation_exterior.py, not the plain pic_calculation.py) and GRO is
any single-frame structure with the SAME atom/residue ordering as was used
for that PIC run (the .gro passed to pic_calculation_exterior.py itself is
the natural choice). PROTEIN_SELECTION defaults to "protein".

OUTPUT LOCATION
---------------
Mirrors characterize_high_pic_residues.py: with no --output-dir given, each
--run writes into <directory containing that run's .pkl>/pic_symmetry_regions/
(no per-label subfolder needed there, since it's already scoped to that
run's own pic_calculation_exterior output folder) -- so this drops
straight into an existing pic_results_<EXCIPIENT>_<JOBID>/ directory with
no extra flags, the same way characterize_high_pic_residues.py drops
pic_residue_characterization/ next to the .pdb it reads. Pass --output-dir
explicitly to instead collect every --run's output under one shared
directory, nested by LABEL (needed when combining multiple serotypes into
one place); combined_overall.csv is only written in that case.

Usage:
    # single serotype, default location (writes next to the .pkl):
    python3 pic_symmetry_regions.py \\
        --run AAV1:pic_results_SUC_12345/pic_SUC_exterior.pkl:md_aav1_ph6.gro

    # multiple serotypes, collected under one shared directory:
    python3 pic_symmetry_regions.py \\
        --run AAV1:pic_SUC_exterior.pkl:md_aav1_ph6.gro \\
        --run AAV8:pic_SUC_exterior_aav8.pkl:md_aav8_ph6.gro \\
        --output-dir pic_symmetry_regions_combined

OUTPUTS (see OUTPUT LOCATION above for where)
----------------------------------------------
  per_residue.csv        chain, resid, resname, gamma23, facing_outer,
                          region_type (5F/3F/""), region_index, angle_deg,
                          radial_angstrom (distance from capsid centroid --
                          written regardless of whether a radial band was
                          used, so it can inform the next run's bands)
  per_region.csv          region_type, region_index, n_residues, mean/sem/
                          std/min/max gamma23, axis unit vector
  overall.csv             region_type in {5F, 3F}, n_residues, n_regions,
                          mean/sem gamma23
  axes_diagnostic.csv     every chain's cluster assignment + axis vectors
  regions_histogram.png   bar chart, mean PIC per individual 3F/5F region
  regions_overall.png     bar chart, overall 3F vs 5F average
  axes_diagnostic.png     3D sanity-check scatter of chain centroids
                          colored by pentamer/trimer cluster
  regions_bfactor.pdb     full-atom structure, hybrid-36 numbered, B-factor
                          = -1 (5F), +1 (3F), 0 (neither) -- exactly the
                          selection that feeds the region/overall averages
                          above, for visual validation in PyMOL/ChimeraX

Also written under --output-dir (only when --output-dir is explicitly
given AND 2+ --run are given):
  combined_overall.csv    overall 3F/5F averages, one row per serotype
"""
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers 3d projection)

import MDAnalysis as mda  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aav.scripts.pic_calculation_exterior import (  # noqa: E402
    build_group_meta, mass_weighted_com, resid_reset_chain_key, write_pdb_hybrid36,
)

N_CHAINS_T1 = 60      # AAV (and every T=1 icosahedral capsid): 60 identical subunits
N_FIVEFOLD = 12        # icosahedron vertices
N_THREEFOLD = 20       # icosahedron faces
FIVEFOLD_SIZE = N_CHAINS_T1 // N_FIVEFOLD   # 5 chains/pentamer
THREEFOLD_SIZE = N_CHAINS_T1 // N_THREEFOLD  # 3 chains/trimer


def setup_logger():
    logger = logging.getLogger("pic_symmetry_regions")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def parse_run_spec(spec):
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"--run must be LABEL:PKL:GRO or LABEL:PKL:GRO:PROTEIN_SELECTION, got: {spec!r}"
        )
    label, pkl_path, gro_path = parts[0], parts[1], parts[2]
    selection = parts[3] if len(parts) == 4 else "protein"
    return label, pkl_path, gro_path, selection


def load_pkl_residues(pkl_path, logger):
    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)
    res = payload["residue"]
    if "facing_outer" not in res:
        raise ValueError(
            f"{pkl_path} has no 'facing_outer' field -- this script needs output from "
            f"pic_calculation_exterior.py (surface-exposed classification), not pic_calculation.py."
        )
    df = pd.DataFrame({
        "resid": np.asarray(res["resids"]).astype(np.int64),
        "resname": np.asarray(res["resnames"]),
        "gamma23": np.asarray(res["gamma_23_final"], dtype=float),
        "gamma23_sem": np.asarray(res["gamma_23_final_sem"], dtype=float),
        "facing_outer": np.asarray(res["facing_outer"], dtype=bool),
    })
    is_boundary = np.concatenate([[True], np.diff(df["resid"].to_numpy()) < 0])
    df["chain"] = np.cumsum(is_boundary) - 1
    logger.info("[%s] %d residues, %d chains, %d surface-exposed (facing_outer)",
                pkl_path, len(df), df["chain"].nunique(), int(df["facing_outer"].sum()))
    return df, payload["metadata"]


def compute_residue_geometry(gro_path, protein_selection, n_res_expected, res_resids_expected,
                              res_resnames_expected, logger):
    """Reference-frame residue centers of mass, in the exact same residue
    order as pic_calculation_exterior.py's own per-residue arrays (ascending
    resindex -- i.e. protein.residues order), plus per-residue chain index.

    Returns (res_com, capsid_centroid, res_chain_idx, ca_com, n_chains,
    pdb_context), where pdb_context is a dict (protein AtomGroup, per-atom
    residue-group index, per-atom chain index, box dimensions) with
    everything write_region_pdb needs to write a full-atom structure later,
    without redoing the universe/selection work.
    """
    u = mda.Universe(gro_path)
    protein = u.select_atoms(protein_selection)
    if protein.n_atoms == 0:
        raise ValueError(f"selection '{protein_selection}' matched 0 atoms in {gro_path}")

    n_res = protein.residues.n_residues
    if n_res != n_res_expected:
        raise ValueError(
            f"{gro_path}: selection '{protein_selection}' has {n_res} residues, "
            f"but the .pkl has {n_res_expected} -- these don't correspond to the same "
            f"system/selection. Pass the matching --protein-selection."
        )
    res_resids = protein.residues.resids
    res_resnames = protein.residues.resnames
    if not np.array_equal(res_resids, res_resids_expected) or \
            not np.array_equal(res_resnames, res_resnames_expected):
        raise ValueError(
            f"{gro_path}: residue resid/resname sequence doesn't match the .pkl's "
            f"residue order -- this structure isn't the one used for the PIC run "
            f"(or --protein-selection differs from what was used originally)."
        )

    res_meta = build_group_meta(protein.resindices, n_res, "protein-residue", logger)
    chain_key = resid_reset_chain_key(protein)
    if chain_key is None:
        raise ValueError(
            f"{gro_path}: could not detect chain boundaries from resid resets -- "
            f"is this really a multi-chain capsid selection?"
        )
    n_chains = int(np.unique(chain_key).size)
    if n_chains != N_CHAINS_T1:
        raise ValueError(
            f"{gro_path}: detected {n_chains} chains, expected {N_CHAINS_T1} (AAV and other "
            f"T=1 icosahedral capsids are 60 identical subunits) -- this script's 12-pentamer/"
            f"20-trimer clustering assumes exactly that. Check --protein-selection / topology."
        )

    # Deliberately NOT using unwrap_protein_positions/rigid_align_chains
    # here: those correct per-FRAME drift across periodic boundaries over
    # an MD trajectory, which is not what a single reference structure like
    # this needs. Tried on this exact capsid .gro (whole outer diameter
    # ~490 A, larger than the 345 A box edge -- expected for a dodecahedral
    # box, which has more room along its diagonals than its edge length
    # suggests) and confirmed harmful: rigid_align_chains' per-chain
    # minimum-image realignment shifted ~17% of atoms by up to a full box
    # length, corrupting an already-whole structure and breaking the
    # pentamer/trimer clustering below. Raw positions, by contrast, gave a
    # clean, verified-correct 12x5/20x3 split (see module docstring). Guard
    # with a physical-plausibility check instead: a real chain shouldn't
    # span anywhere near the capsid's own diameter.
    pos = protein.positions
    res_com = mass_weighted_com(pos, protein.masses, res_meta)
    capsid_centroid = res_com.mean(axis=0)

    # per-residue chain index: chain_key is per-ATOM; every atom of a given
    # residue shares one chain, so the first atom of each residue group
    # (res_meta's own anchor atom) carries the residue's chain label.
    res_chain_idx = chain_key[res_meta["first_atom_idx"]]

    # One CA position per residue, in the SAME row order as res_com (both
    # come from res_meta's ascending-resindex grouping) -- this is what
    # lets chain_ca[c] below be built by a plain boolean mask on
    # res_chain_idx, with residues already in ascending-resid order within
    # each chain, giving an exact atom-for-atom correspondence across
    # chains for the pairwise structural alignment in find_symmetry_axes.
    ca_mask = protein.names == "CA"
    if int(ca_mask.sum()) != n_res:
        raise ValueError(
            f"{gro_path}: expected exactly one CA atom per residue ({n_res}), found "
            f"{int(ca_mask.sum())} -- can't build the per-chain structural correspondence "
            f"needed for symmetry-axis detection."
        )
    ca_local_idx = res_meta["local_idx"][ca_mask]
    order = np.argsort(ca_local_idx)
    ca_com = pos[ca_mask][order]

    # Physical-plausibility guard in place of a per-frame PBC unwrap (see
    # above): a real chain shouldn't span anywhere near the capsid's own
    # diameter. If it does, this structure has a chain split across a
    # periodic boundary and needs PBC-whole preprocessing (e.g. `gmx
    # trjconv -pbc mol -ur compact`) before being used with this script.
    capsid_diameter = 2.0 * np.linalg.norm(res_com - capsid_centroid, axis=1).max()
    max_chain_span = 0.0
    for c in range(n_chains):
        chain_pts = res_com[res_chain_idx == c]
        max_chain_span = max(max_chain_span, np.linalg.norm(
            chain_pts[:, None, :] - chain_pts[None, :, :], axis=-1).max())
    if max_chain_span > 0.5 * capsid_diameter:
        raise ValueError(
            f"{gro_path}: at least one chain spans {max_chain_span:.0f} A, more than half the "
            f"capsid's own diameter ({capsid_diameter:.0f} A) -- this indicates a chain split "
            f"across a periodic boundary. Pre-process to a PBC-whole structure (e.g. `gmx "
            f"trjconv -pbc mol -ur compact`) before using it with this script."
        )

    pdb_context = {
        "protein": protein,
        "res_local_idx": res_meta["local_idx"],  # per-atom, values 0..n_res-1
        "chain_key": chain_key,                  # per-atom, values 0..n_chains-1
        "box_dimensions": u.dimensions,
    }
    return res_com, capsid_centroid, res_chain_idx, ca_com, n_chains, pdb_context


def _kabsch_rotation(P, Q):
    """Best-fit rotation matrix R minimizing |R(P-mean(P)) - (Q-mean(Q))|."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def find_symmetry_axes(chain_com, chain_unit, chain_ca, n_chains, logger,
                        angle_tol_deg=8.0, axis_local_tol_deg=30.0):
    """Discover the 12 five-fold and 20 three-fold symmetry axes from the
    60 chains' own CA coordinates, with no assumed literature geometry.

    For every pair of chains, the best-fit (Kabsch) rotation mapping one
    chain's CA atoms onto the other's reveals the icosahedral group element
    relating them. Its rotation ANGLE alone is not enough to identify true
    pentamer/trimer neighbors -- many non-neighboring pairs are also
    related by a 72/144/120 degree rotation about some other, distant axis.
    A pair only counts as a genuine neighbor if the rotation's AXIS is also
    close (< axis_local_tol_deg) to both chains' own centroid directions
    from the capsid centroid, i.e. the axis is actually near where both
    chains physically sit. See the module docstring for the empirical
    validation of this filter (120/60 axis-local pairs recovered out of
    ~360/~600 angle-only candidates, cleanly separating into exactly 12
    components of 5 chains and 20 components of 3).

    Returns (five_axes (12,3), five_members, three_axes (20,3), three_members).
    """
    adj5 = np.zeros((n_chains, n_chains), dtype=bool)
    adj3 = np.zeros((n_chains, n_chains), dtype=bool)
    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            R = _kabsch_rotation(chain_ca[i], chain_ca[j])
            rot = Rotation.from_matrix(R)
            theta = np.degrees(rot.magnitude())
            rotvec = rot.as_rotvec()
            norm = np.linalg.norm(rotvec)
            if norm < 1e-8:
                continue  # ~identity rotation between distinct chains -- shouldn't happen
            axis = rotvec / norm
            ang_i = np.degrees(np.arccos(np.clip(abs(axis @ chain_unit[i]), -1.0, 1.0)))
            ang_j = np.degrees(np.arccos(np.clip(abs(axis @ chain_unit[j]), -1.0, 1.0)))
            axis_local = ang_i < axis_local_tol_deg and ang_j < axis_local_tol_deg
            if axis_local and (abs(theta - 72) < angle_tol_deg or abs(theta - 144) < angle_tol_deg):
                adj5[i, j] = adj5[j, i] = True
            elif axis_local and abs(theta - 120) < angle_tol_deg:
                adj3[i, j] = adj3[j, i] = True

    def clusters_from_adjacency(adj, n_expected, expected_size, label):
        n_comp, labels = connected_components(csr_matrix(adj), directed=False)
        uniq, counts = np.unique(labels, return_counts=True)
        if uniq.size != n_expected or not np.all(counts == expected_size):
            raise ValueError(
                f"[{label}] axis-local adjacency graph gave {uniq.size} components with sizes "
                f"{sorted(counts.tolist())}, expected {n_expected} components of {expected_size} "
                f"-- the capsid geometry may be too distorted for this pairwise-rotation method, "
                f"or this isn't a standard T=1 icosahedral capsid. Check axes_diagnostic.png/csv "
                f"and consider adjusting angle_tol_deg/axis_local_tol_deg."
            )
        axes = np.empty((n_expected, 3), dtype=float)
        members = []
        for i, k in enumerate(uniq):
            idx = np.where(labels == k)[0]
            v = chain_unit[idx].mean(axis=0)
            v /= np.linalg.norm(v)
            axes[i] = v
            members.append(idx)
        logger.info("[%s] resolved %d chains into %d groups of %d (OK)",
                    label, n_chains, n_expected, expected_size)
        return axes, members

    five_axes, five_members = clusters_from_adjacency(adj5, N_FIVEFOLD, FIVEFOLD_SIZE, "5-fold")
    three_axes, three_members = clusters_from_adjacency(adj3, N_THREEFOLD, THREEFOLD_SIZE, "3-fold")
    return five_axes, five_members, three_axes, three_members


def angles_to_axes_deg(unit_vectors, axes):
    cos = np.clip(unit_vectors @ axes.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def assign_regions(res_com, capsid_centroid, five_axes, three_axes,
                    five_max_angle_deg, three_max_angle_deg,
                    five_r_range=None, three_r_range=None):
    """Angular assignment against the nearest axis of EITHER family, then
    two independent per-family cutoffs applied afterward: an angular one
    (tangential -- how far around the axis, i.e. patch width/radius on the
    capsid surface) and an optional radial one (radial -- how far out from
    the centroid, i.e. depth/height). Both cutoffs are applied AFTER the
    angular nearest-axis choice, not searched jointly: a residue that's
    angularly closest to a 5-fold axis but fails the 5-fold family's own
    angle or radius cutoff becomes unclassified rather than falling through
    to 3-fold, even if it would satisfy 3-fold's cutoffs -- each residue
    still has exactly one "natural" axis family, these cutoffs only narrow
    how much of that family's patch counts, they don't let a residue
    switch families.

    five_max_angle_deg / three_max_angle_deg: max angular distance (deg)
    from a residue to its nearest axis of that family, i.e. the cone
    half-angle / patch width, analogous to a great-circle radius on the
    capsid's surface. This is the ONLY thing --max-axis-angle-deg used to
    control, shared between both families; it's now two independent knobs.

    five_r_range / three_r_range: (min, max) radial distance from the
    capsid centroid, in the SAME units as res_com/capsid_centroid (A).
    None (default) imposes no radial restriction on that family, i.e. the
    original unbounded-with-angle-only behavior.

    On AAV1 (this module's reference system), the 3-fold axes sit on the
    capsid's protrusions/spikes, which have real radial extent from base
    to tip -- an unbounded radial window there mixes "generic shell floor
    near the spike's angular position" (radially indistinguishable from
    the rest of the surface) in with the spike tip itself. The radial band
    is how you keep "the 3-fold region" meaning the tip, not the whole
    protrusion's footprint.
    """
    res_radial = np.linalg.norm(res_com - capsid_centroid, axis=1)
    res_unit = (res_com - capsid_centroid) / res_radial[:, None]

    ang5 = angles_to_axes_deg(res_unit, five_axes)
    ang3 = angles_to_axes_deg(res_unit, three_axes)
    min5_idx, min5 = ang5.argmin(axis=1), ang5.min(axis=1)
    min3_idx, min3 = ang3.argmin(axis=1), ang3.min(axis=1)

    is5_closer = min5 <= min3
    region_type = np.where(is5_closer, "5F", "3F")
    region_index = np.where(is5_closer, min5_idx, min3_idx)
    region_angle = np.where(is5_closer, min5, min3)

    within_cutoff = np.where(is5_closer, region_angle <= five_max_angle_deg,
                              region_angle <= three_max_angle_deg)

    def in_range(r_range):
        if r_range is None:
            return np.ones_like(res_radial, dtype=bool)
        r_min, r_max = r_range
        return (res_radial >= r_min) & (res_radial <= r_max)

    within_radius = np.where(is5_closer, in_range(five_r_range), in_range(three_r_range))

    keep = within_cutoff & within_radius
    region_type = np.where(keep, region_type, "")
    region_index = np.where(keep, region_index, -1)
    return region_type, region_index, region_angle, res_radial


def region_summary(df, region_col_type, region_col_index, value_col="gamma23"):
    sub = df[df[region_col_type] != ""].copy()
    g = sub.groupby([region_col_type, region_col_index])[value_col]
    out = g.agg(n="count", mean="mean", std="std", min="min", max="max").reset_index()
    out["sem"] = out["std"] / np.sqrt(out["n"])
    out = out.rename(columns={region_col_type: "region_type", region_col_index: "region_index"})
    out["region_index"] = out["region_index"].astype(int)
    return out.sort_values(["region_type", "region_index"]).reset_index(drop=True)


def overall_summary(df, region_col_type, value_col="gamma23"):
    sub = df[df[region_col_type] != ""].copy()
    rows = []
    for rtype, grp in sub.groupby(region_col_type):
        n_regions = grp["region_index"].nunique() if "region_index" in grp else grp[region_col_type + "_idx"].nunique()
        mean = grp[value_col].mean()
        sem = grp[value_col].std(ddof=1) / np.sqrt(len(grp)) if len(grp) > 1 else np.nan
        rows.append((rtype, len(grp), n_regions, mean, sem))
    return pd.DataFrame(rows, columns=["region_type", "n_residues", "n_regions", "mean_gamma23", "sem_gamma23"])


REGION_COLORS = {"5F": "#2b6cb0", "3F": "#c53030"}


def plot_region_histogram(per_region_df, overall_df, out_path, label):
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = 0
    xticks, xlabels, bar_x, bar_h, bar_err, bar_c = [], [], [], [], [], []
    for rtype in ["5F", "3F"]:
        sub = per_region_df[per_region_df["region_type"] == rtype].sort_values("region_index")
        for _, row in sub.iterrows():
            bar_x.append(x_pos)
            bar_h.append(row["mean"])
            bar_err.append(row["sem"] if np.isfinite(row["sem"]) else 0.0)
            bar_c.append(REGION_COLORS[rtype])
            xticks.append(x_pos)
            xlabels.append(f"{rtype}-{int(row['region_index'])}")
            x_pos += 1
        x_pos += 1.5  # gap between region-type groups

    ax.bar(bar_x, bar_h, yerr=bar_err, color=bar_c, capsize=2, width=0.8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.axhline(0.0, color="black", lw=0.8)

    for rtype in ["5F", "3F"]:
        row = overall_df[overall_df["region_type"] == rtype]
        if not row.empty:
            ax.axhline(row["mean_gamma23"].iloc[0], color=REGION_COLORS[rtype], ls="--", lw=1.2,
                       label=f"{rtype} overall mean = {row['mean_gamma23'].iloc[0]:.3f}")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel(r"mean $\Gamma_{23}$ (PIC), surface-exposed residues")
    ax.set_title(f"{label}: PIC by icosahedral 5-fold / 3-fold region")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overall(overall_df, out_path, label):
    fig, ax = plt.subplots(figsize=(4, 5))
    order = ["5F", "3F"]
    means = [overall_df.loc[overall_df["region_type"] == t, "mean_gamma23"].squeeze()
             if t in overall_df["region_type"].values else np.nan for t in order]
    sems = [overall_df.loc[overall_df["region_type"] == t, "sem_gamma23"].squeeze()
            if t in overall_df["region_type"].values else 0.0 for t in order]
    colors = [REGION_COLORS[t] for t in order]
    ax.bar(order, means, yerr=sems, color=colors, capsize=4, width=0.6)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel(r"mean $\Gamma_{23}$ (PIC), surface-exposed residues")
    ax.set_title(f"{label}: overall 5-fold vs 3-fold PIC")
    for i, t in enumerate(order):
        row = overall_df[overall_df["region_type"] == t]
        if not row.empty:
            ax.text(i, means[i], f"n={int(row['n_residues'].iloc[0])}", ha="center",
                    va="bottom" if means[i] >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_axes_diagnostic(chain_unit, five_members, three_members, five_axes, three_axes, out_path, label):
    fig = plt.figure(figsize=(11, 5.5))
    for panel, (members, axes, title) in enumerate([
        (five_members, five_axes, "5-fold clustering (12 pentamers)"),
        (three_members, three_axes, "3-fold clustering (20 trimers)"),
    ]):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        cmap = plt.get_cmap("tab20" if len(members) > 12 else "tab20b")
        for i, idx in enumerate(members):
            pts = chain_unit[idx]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=cmap(i % 20), s=25)
            a = axes[i]
            ax.plot([0, a[0]], [0, a[1]], [0, a[2]], color=cmap(i % 20), lw=1.0, alpha=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle(f"{label}: chain-centroid directions clustered into symmetry axes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_region_pdb(df, pdb_context, out_path, logger):
    """Full-atom, hybrid-36-numbered PDB for visual validation of the region
    assignment: B-factor = -1 for a surface-exposed residue assigned to a
    5-fold region, +1 for 3-fold, 0 for everything else (not surface-exposed,
    or surface-exposed but not close enough to either axis family) -- i.e.
    exactly the residue selection that feeds region_summary/overall_summary,
    not just the raw geometric nearest-axis assignment. All atoms of one
    residue share its residue's value (matches pic_calculation_exterior.py's
    own *_bfactor.pdb convention).
    """
    selected = np.zeros(len(df), dtype=float)
    selected[(df["facing_outer"]) & (df["region_type"] == "5F")] = -1.0
    selected[(df["facing_outer"]) & (df["region_type"] == "3F")] = 1.0

    per_atom_bfactor = selected[pdb_context["res_local_idx"]]
    write_pdb_hybrid36(
        pdb_context["protein"], per_atom_bfactor, out_path, logger,
        box_dimensions=pdb_context["box_dimensions"],
        chain_local_idx=pdb_context["chain_key"],
        remark_lines=[
            "generated by pic_symmetry_regions.py",
            "B-factor column = icosahedral region selection, NOT Gamma_23:",
            "-1.00 = surface-exposed residue in a 5-fold region",
            "+1.00 = surface-exposed residue in a 3-fold region",
            " 0.00 = neither (not surface-exposed, or outside that family's "
            "angle/radius cutoff -- see this run's log for the values used)",
        ],
    )
    logger.info("wrote region-selection B-factor PDB (-1=5F, 0=neither, +1=3F) -> %s", out_path)


def run_one(label, pkl_path, gro_path, selection, five_max_angle_deg, three_max_angle_deg,
            out_dir, logger, five_r_offsets=None, three_r_offsets=None):
    """five_max_angle_deg / three_max_angle_deg: per-family angular (i.e.
    tangential -- patch width on the capsid surface) cutoff in degrees.
    Independent per family; main() resolves --max-axis-angle-deg as the
    shared fallback when either --five-fold-max-angle-deg /
    --three-fold-max-angle-deg isn't given, so run_one always receives two
    concrete values, never a "use the shared one" sentinel.

    five_r_offsets / three_r_offsets: (min, max) OFFSET in A from this
    run's own shell_outer_radius_angstrom (from the .pkl metadata), not an
    absolute radial distance -- offsets transfer across serotypes even
    though the raw capsid radius itself differs slightly (AAV1/AAV8/AAV2
    in this project are 103-105 A, close but not identical). None (default)
    imposes no radial restriction, matching the original behavior.
    """
    logger.info("=== %s ===", label)
    os.makedirs(out_dir, exist_ok=True)

    df, metadata = load_pkl_residues(pkl_path, logger)
    n_res = len(df)

    shell_r = metadata["shell_outer_radius_angstrom"]
    five_r_range = (shell_r + five_r_offsets[0], shell_r + five_r_offsets[1]) if five_r_offsets else None
    three_r_range = (shell_r + three_r_offsets[0], shell_r + three_r_offsets[1]) if three_r_offsets else None
    if five_r_range or three_r_range:
        logger.info("[%s] radial bands (absolute, from shell_outer_radius=%.1f A): "
                    "5F=%s  3F=%s", label, shell_r, five_r_range, three_r_range)

    res_com, capsid_centroid, res_chain_idx, ca_com, n_chains, pdb_context = compute_residue_geometry(
        gro_path, selection, n_res, df["resid"].to_numpy(), df["resname"].to_numpy(), logger,
    )
    if not np.array_equal(res_chain_idx, df["chain"].to_numpy()):
        raise ValueError(
            f"{gro_path}: chain assignment derived from the structure doesn't match the "
            f".pkl's own resid-reset chain boundaries -- residue ordering mismatch."
        )

    chain_com = np.zeros((n_chains, 3), dtype=float)
    chain_ca = []
    chain_resid = []
    resids_all = df["resid"].to_numpy()
    for c in range(n_chains):
        mask = res_chain_idx == c
        chain_com[c] = res_com[mask].mean(axis=0)
        chain_ca.append(ca_com[mask])
        chain_resid.append(resids_all[mask])
    chain_unit = chain_com - capsid_centroid
    chain_unit /= np.linalg.norm(chain_unit, axis=1, keepdims=True)

    ca_lengths = {len(c) for c in chain_ca}
    if len(ca_lengths) != 1:
        # Not necessarily a chain-boundary bug: real capsid structures can
        # have a handful of surface residues resolved on some of the 60
        # icosahedral copies and not others (e.g. a flexible loop only
        # ordered on a subset of copies) -- a legitimate per-chain modeling
        # difference. Pairwise Kabsch alignment below only needs the SAME
        # physical residues, in the SAME order, across chains -- not every
        # residue -- so fall back to the resid intersection common to all
        # chains instead of failing outright.
        common_resids = set(chain_resid[0].tolist())
        for r in chain_resid[1:]:
            common_resids &= set(r.tolist())
        missing = sorted(set(resids_all.tolist()) - common_resids)
        common_resids = np.array(sorted(common_resids))
        logger.warning(
            "chains have unequal CA counts %s -- %d resid(s) not common to all %d chains "
            "(%s); using the %d-residue common subset for pairwise axis-finding alignment "
            "only. Per-residue Gamma_23/region assignment is computed on the full residue "
            "set and is unaffected.",
            sorted(ca_lengths), len(missing), n_chains, missing, len(common_resids),
        )
        for c in range(n_chains):
            keep = np.isin(chain_resid[c], common_resids)
            chain_ca[c] = chain_ca[c][keep]
        ca_lengths = {len(c) for c in chain_ca}
        if len(ca_lengths) != 1:
            raise ValueError(
                f"chains still have unequal CA counts {sorted(ca_lengths)} after restricting "
                f"to the common resid subset -- residue ordering/duplication differs between "
                f"chains beyond simple presence/absence. Needs manual inspection."
            )
    chain_ca = np.array(chain_ca)

    five_axes, five_members, three_axes, three_members = find_symmetry_axes(
        chain_com, chain_unit, chain_ca, n_chains, logger,
    )

    region_type, region_index, region_angle, res_radial = assign_regions(
        res_com, capsid_centroid, five_axes, three_axes,
        five_max_angle_deg, three_max_angle_deg,
        five_r_range=five_r_range, three_r_range=three_r_range,
    )
    df["region_type"] = region_type
    df["region_index"] = region_index
    df["angle_deg"] = region_angle
    # kept in the output specifically so radial bands can be tuned from a
    # PAST run's per_residue.csv -- no need to re-derive this distribution
    # with a side script the way it was worked out during development.
    df["radial_angstrom"] = res_radial

    surface = df[df["facing_outer"]].copy()
    n_5f = int((surface["region_type"] == "5F").sum())
    n_3f = int((surface["region_type"] == "3F").sum())
    logger.info("[%s] surface-exposed residues assigned: 5F=%d, 3F=%d (of %d surface-exposed, "
                "max-axis-angle: 5F=%.1f deg, 3F=%.1f deg)",
                label, n_5f, n_3f, len(surface), five_max_angle_deg, three_max_angle_deg)

    per_region = region_summary(surface, "region_type", "region_index")
    overall = overall_summary(surface, "region_type")
    overall.insert(0, "label", label)

    df.to_csv(os.path.join(out_dir, "per_residue.csv"), index=False)
    per_region.to_csv(os.path.join(out_dir, "per_region.csv"), index=False)
    overall.drop(columns="label").to_csv(os.path.join(out_dir, "overall.csv"), index=False)

    diag_rows = []
    for rtype, members, axes in [("5F", five_members, five_axes), ("3F", three_members, three_axes)]:
        for i, idx in enumerate(members):
            for c in idx:
                diag_rows.append((rtype, i, int(c), *axes[i]))
    diag_df = pd.DataFrame(diag_rows, columns=["region_type", "region_index", "chain",
                                                "axis_x", "axis_y", "axis_z"])
    diag_df.to_csv(os.path.join(out_dir, "axes_diagnostic.csv"), index=False)

    plot_region_histogram(per_region, overall, os.path.join(out_dir, "regions_histogram.png"), label)
    plot_overall(overall, os.path.join(out_dir, "regions_overall.png"), label)
    plot_axes_diagnostic(chain_unit, five_members, three_members, five_axes, three_axes,
                          os.path.join(out_dir, "axes_diagnostic.png"), label)
    write_region_pdb(df, pdb_context, os.path.join(out_dir, "regions_bfactor.pdb"), logger)

    logger.info("[%s] wrote outputs to %s", label, out_dir)
    return overall


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, dest="runs",
                    help="LABEL:PKL:GRO[:PROTEIN_SELECTION] -- repeat for multiple serotypes "
                         "(e.g. --run AAV1:... --run AAV8:...)")
    p.add_argument("--max-axis-angle-deg", type=float, default=20.0,
                    help="Shared DEFAULT max angular distance (degrees) from a residue to its "
                         "nearest 5-fold/3-fold axis for it to count as 'in that region' -- i.e. "
                         "the TANGENTIAL width of the selected patch on the capsid surface, as "
                         "opposed to --five/three-fold-r-min/max which control radial depth. "
                         "Default 20.0, roughly half the ~37-42 deg gap between neighboring axis "
                         "families on a regular icosahedron. Used for whichever family doesn't get "
                         "its own --five-fold-max-angle-deg / --three-fold-max-angle-deg override.")
    p.add_argument("--five-fold-max-angle-deg", type=float, default=None,
                    help="Override --max-axis-angle-deg for the 5-fold family only. Narrow this to "
                         "shrink the 5-fold patch's tangential width without affecting 3-fold.")
    p.add_argument("--three-fold-max-angle-deg", type=float, default=None,
                    help="Override --max-axis-angle-deg for the 3-fold family only.")
    p.add_argument("--output-dir", default=None,
                    help="Collect every --run's output under this directory (nested by LABEL). "
                         "Default: none -- each --run instead writes into <directory containing "
                         "that run's .pkl>/pic_symmetry_regions/, the same convention "
                         "characterize_high_pic_residues.py uses, so this can be dropped into an "
                         "existing pic_calculation_exterior.py output folder with no extra flags.")
    p.add_argument("--five-fold-r-min", type=float, default=None,
                    help="Radial band for 5-fold membership: min distance in A from this run's own "
                         "shell_outer_radius_angstrom (.pkl metadata), NOT an absolute radius -- an "
                         "offset, so it transfers across serotypes with slightly different capsid "
                         "sizes. Default: unrestricted (matches the pre-radial-band behavior).")
    p.add_argument("--five-fold-r-max", type=float, default=None,
                    help="Radial band for 5-fold membership: max offset in A, see --five-fold-r-min.")
    p.add_argument("--three-fold-r-min", type=float, default=None,
                    help="Radial band for 3-fold membership: min offset in A, see --five-fold-r-min. "
                         "On AAV1 the 3-fold axes sit on the capsid's protrusions/spikes, which span "
                         "a much wider and deeper radial range (base to tip) than the 5-fold pores do "
                         "-- raise this to exclude the spike's base (radially indistinguishable from "
                         "generic shell) and keep just the tip.")
    p.add_argument("--three-fold-r-max", type=float, default=None,
                    help="Radial band for 3-fold membership: max offset in A, see --three-fold-r-min.")
    args = p.parse_args()

    logger = setup_logger()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    def offsets_or_none(r_min, r_max):
        if r_min is None and r_max is None:
            return None
        return (r_min if r_min is not None else -np.inf, r_max if r_max is not None else np.inf)

    five_r_offsets = offsets_or_none(args.five_fold_r_min, args.five_fold_r_max)
    three_r_offsets = offsets_or_none(args.three_fold_r_min, args.three_fold_r_max)

    five_max_angle = (args.five_fold_max_angle_deg if args.five_fold_max_angle_deg is not None
                       else args.max_axis_angle_deg)
    three_max_angle = (args.three_fold_max_angle_deg if args.three_fold_max_angle_deg is not None
                        else args.max_axis_angle_deg)

    overall_all = []
    for spec in args.runs:
        label, pkl_path, gro_path, selection = parse_run_spec(spec)
        for f in (pkl_path, gro_path):
            if not os.path.isfile(f):
                logger.error("Input file not found: %s", f)
                sys.exit(1)
        if args.output_dir is not None:
            out_dir = os.path.join(args.output_dir, label)
        else:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(pkl_path)), "pic_symmetry_regions")
        overall_all.append(
            run_one(label, pkl_path, gro_path, selection, five_max_angle, three_max_angle, out_dir, logger,
                    five_r_offsets=five_r_offsets, three_r_offsets=three_r_offsets)
        )

    if args.output_dir is not None and len(overall_all) > 1:
        combined = pd.concat(overall_all, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "combined_overall.csv")
        combined.to_csv(combined_path, index=False)
        logger.info("wrote combined cross-serotype summary -> %s", combined_path)

    logger.info("done.")


if __name__ == "__main__":
    main()