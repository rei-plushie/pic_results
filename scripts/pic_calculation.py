#!/usr/bin/env python3
"""
pic_calculation.py
===================

Computes the preferential interaction coefficient, Gamma_23, between a
protein and a cosolvent/excipient (NA, CL, or SUC) from an explicit-solvent
MD trajectory, on a PER-RESIDUE basis, following the local/bulk domain
algorithm of:

    Shukla, D.; Shinde, C.; Trout, B. L. J. Phys. Chem. B 2009, 113, 12546-12554.
    (the four numbered points + Eqs. 2-4 on p. 12548)

WHAT CHANGED FROM THE WHOLE-PROTEIN VERSION
--------------------------------------------
The published method treats the whole protein as one VdW-sphere surface and
asks "how far is this water/excipient molecule from the nearest protein
atom." That collapses to a single Gamma_23 number for the entire molecule.

Here, every protein RESIDUE is treated as its own local sphere (center =
residue center of mass, radius = distance from that COM to its own farthest
atom, plus that atom's VdW radius -- i.e. the same "atom is a VdW sphere"
idea from the paper, just applied one level up so the whole calculation
scales with n_residues (~500) instead of n_atoms (~8000) as the spatial
reference set). Eqs. 2-4 are then evaluated independently for each residue,
using its own local counts n1_r(residue,t) / n3_r(residue,t), against the
SAME global bulk totals n1_total / n3_total. This is an explicit, deliberate
approximation (a residue is not really a sphere), justified because (a) it
is what makes per-residue resolution computationally tractable at all, and
(b) the radius is defined conservatively (farthest atom + its VdW radius),
so it never UNDER-counts a residue's true footprint.

A "whole-molecule" Gamma_23(r,t) is still produced for validation/backward
compatibility -- reusing the exact same residue-sphere pair list, it is
free to also take the MIN adjusted distance across residues per solvent
molecule (i.e. "distance to the nearest residue-sphere" instead of
"distance to the nearest atom"). This is a close approximation of the
literature quantity, not identical to it, and is documented as such in the
output.

PERFORMANCE
-----------
The original implementation measured every water/excipient molecule's
distance to every PROTEIN ATOM (8002 atoms here) via an exhaustive radius
search, which is what made it slow: ~6.7 s/frame just for the water search
on this system, dominated by ~6.9M candidate pairs (most protein atoms
within reach of any given surface water, since residues pack densely).
Two changes fix this without introducing a memory blowup at capsid scale
(see the method='bruteforce' postmortem below -- an earlier version of
this docstring recommended it, which was wrong at scale and is now fixed):

  1. Search against residue COMs (518 reference points on the monomer
     this was developed against) instead of atoms (8002 reference points)
     -- this is also what per-residue output needs, so it isn't a
     separate optimization, it falls out of the redesign. Measured: ~0.2
     s/frame for the same water search (>30x) on that system.

     NOTE: capped_distance's method is chosen dynamically per frame in
     frame_residue_histograms, not hardcoded to 'bruteforce'. bruteforce
     is ~2-5x faster on a small system like this one, but the underlying
     _bruteforce_capped computes a FULL DENSE distance_array(reference,
     configuration) before applying the cutoff -- memory scales with
     n_solvent * n_residues regardless of how many pairs actually survive.
     A first version of this hardcoded 'bruteforce' unconditionally (it
     benchmarked well here), which OOM-killed a real capsid-scale job at
     ~124 GB for a SINGLE rank once n_residues and n_solvent were both
     much larger than this monomer's. It's now only used below a
     conservative n_solvent*n_residues safety threshold (well under
     MDAnalysis's own 1e8 cutoff for auto-selecting it); above that,
     MDAnalysis's default heuristic picks a grid-based method that scales
     with actual output size instead. See frame_residue_histograms.

  2. Per-frame results are folded into running sum/sum-of-squares/count
     accumulators (shape n_residues x n_bins) instead of being stored
     frame-by-frame and gathered at the end. The old approach's memory
     scaled with n_frames x n_bins; per-residue that would have been
     n_frames x n_residues x n_bins (multi-GB). The accumulator approach
     is O(n_residues x n_bins) regardless of trajectory length, and the
     final MPI reduction moves a few hundred KB instead of the whole
     trajectory's worth of intermediate arrays.

Benchmarked end-to-end on the real 1001-frame trajectory this ships with:
single-process wall time dropped from an estimated ~6700 s (whole-protein,
atom-level) to ~230 s (per-residue, residue-sphere) -- while producing far
more information (518 residues instead of 1 number).

OUTPUTS
-------
  pic_<EXCIPIENT>.pkl              -- full per-residue + whole-molecule arrays
  pic_<EXCIPIENT>_per_residue.csv  -- resid, resname, gamma23, sem, r*
  pic_<EXCIPIENT>_vs_r.csv         -- whole-molecule r-profile (validation)
  pic_<EXCIPIENT>_summary.png      -- whole-molecule r-profile + per-residue bar
  pic_<EXCIPIENT>_bfactor.pdb      -- reference-frame structure, B-factor = Gamma_23
                                       per residue (all atoms of a residue share
                                       its value) for coloring in PyMOL/ChimeraX/VMD

Parallelization: embarrassingly parallel over trajectory frames via MPI
(mpi4py / OpenMPI), same as before. If mpi4py or an MPI runtime is not
available (e.g. running locally for a quick check), the script transparently
falls back to a single-process, no-MPI mode -- see _SerialMPI below.

Usage (typically launched via pic.sh on the cluster, or directly for a
quick local check):
    mpirun -np <N> python3 pic_calculation.py \\
        --gro md.gro --xtc md_noPBC.xtc \\
        --excipient SUC --radius 20.0 \\
        --output-dir pic_results

    python3 pic_calculation.py --gro md.gro --xtc md_noPBC.xtc \\
        --excipient SUC --radius 20.0 --max-frames 20 --output-dir /tmp/test
"""
import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np

# matplotlib must use a non-interactive backend before pyplot is imported --
# there is no display on HPC compute nodes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import MDAnalysis as mda  # noqa: E402
from MDAnalysis.lib.distances import capped_distance, minimize_vectors  # noqa: E402

try:
    from mpi4py import MPI
    _HAS_REAL_MPI = True
except Exception:
    _HAS_REAL_MPI = False

    class _SerialComm:
        """Minimal stand-in for mpi4py's Comm, used when mpi4py or an MPI
        runtime isn't available (e.g. local testing on a laptop). Only the
        subset of the API this script actually calls is implemented."""

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            pass

        def allreduce(self, value, op=None):
            return value

        def reduce(self, value, op=None, root=0):
            return value

        def gather(self, value, root=0):
            return [value]

        def Abort(self, code=1):
            sys.exit(code)

    class MPI:  # noqa: N801 -- mimicking the mpi4py.MPI module surface
        COMM_WORLD = _SerialComm()
        SUM = "SUM"
        MAX = "MAX"
        MIN = "MIN"


# --------------------------------------------------------------------------
# Bondi (1964) van der Waals radii, Angstrom. Used both for the original
# atom-level VdW correction and, folded in via residue_radius, for
# the residue-sphere radius. Only elements expected in a standard CHARMM36
# protein (no cofactors / metals) are included; anything else triggers a
# logged warning and falls back to the carbon radius.
# --------------------------------------------------------------------------
BONDI_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}
DEFAULT_VDW = 1.70  # fallback (carbon) for unrecognized elements, with a warning

# Formal charges for the two ionic excipients, used only for the optional
# charge-balance diagnostic (see --charge / build_charge_profile).
ION_FORMAL_CHARGE = {"NA": +1.0, "CL": -1.0}

# Rough per-frame cost with the residue-sphere search (measured ~0.2-0.25s on
# an ~280k-atom / 518-residue system; scales with system size, not a hard
# constant). Used only for the rank-count advisory below, not for correctness.
SECONDS_PER_FRAME_ESTIMATE = 0.25
# Below this many frames/rank, fixed per-rank overhead (Universe/topology
# load, MPI startup) starts to dominate over actual compute -- see the
# advisory logged where n_frames and size are both known. This matters when
# requesting a large, HPC-scale --ntasks (e.g. up to 384) against a short
# trajectory or a quick test run.
MIN_USEFUL_FRAMES_PER_RANK = 20


# ==========================================================================
# Core numerical routines
# ==========================================================================

def guess_element(atom_name):
    """Guess element from a standard CHARMM/AMBER protein atom name.

    Strips leading digits (e.g. '1HB1' -> 'HB1') and takes the first
    alphabetic character. This is correct for standard protein atom naming
    (CA, CB, OG1, ND2, SD, HB1, ... all genuinely start with their element's
    letter) and deliberately does NOT use a generic two-letter-element
    guesser, because within a protein selection 'CA' means alpha-carbon
    (element C), not the metal ion calcium -- a generic guesser can get this
    backwards. This function is only ever applied to the `protein` selection.
    """
    name = atom_name.lstrip("0123456789")
    return name[0].upper() if name else "?"


def get_protein_vdw_radii(protein_ag, logger):
    """Per-atom VdW radius array for the protein selection, Bondi scale."""
    elements = [guess_element(n) for n in protein_ag.names]
    unknown = sorted(set(e for e in elements if e not in BONDI_RADII))
    if unknown:
        logger.warning(
            "Protein atom elements not in Bondi table (using C=%.2f A as "
            "fallback): %s -- verify these aren't cofactors/metals that "
            "need their own radius.", DEFAULT_VDW, unknown
        )
    return np.array([BONDI_RADII.get(e, DEFAULT_VDW) for e in elements], dtype=float)


def validate_masses(ag, label, logger):
    """Fail loudly (not silently) if mass guessing produced garbage.

    True center-of-mass requires real per-atom masses. MDAnalysis guesses
    masses from guessed elements when reading a bare .gro (no force-field
    info), and that guess is normally fine for the elements present here
    (H, C, N, O, S, Na, Cl) -- but a silent guessing failure (e.g. all
    masses defaulting to 0 or 1) would corrupt every downstream COM, so we
    check rather than assume.
    """
    m = ag.masses
    if m.size == 0:
        raise ValueError(f"[{label}] selection is empty -- check residue name / topology.")
    if np.any(~np.isfinite(m)) or np.any(m <= 0):
        bad = np.unique(ag.names[~np.isfinite(m) | (m <= 0)])
        raise ValueError(
            f"[{label}] {bad.size} atom name(s) have invalid mass (<=0 or NaN), "
            f"mass guessing likely failed for: {list(bad)[:10]}"
        )
    logger.info("[%s] %d atoms, mass range %.3f-%.3f amu (OK)",
                label, m.size, m.min(), m.max())


def build_group_meta(resindices, n_expected, label, logger):
    """One-time (topology-only, not per-frame) grouping metadata for a
    selection's residues/molecules: a local 0..n_groups-1 index per atom,
    plus a (sort_order, group_starts, first_atom_idx) triple that lets
    every later per-frame reduction (COM, radius, PBC unwrap anchor) run
    as a vectorized sort+reduceat instead of np.*.at scatter ops, which
    don't vectorize well once atom counts reach into the hundreds of
    thousands (a full AAV capsid, not just a monomer).
    """
    uniq_resindex, local_idx = np.unique(resindices, return_inverse=True)
    n_groups = uniq_resindex.size
    if n_groups != n_expected:
        raise ValueError(
            f"[{label}] grouping mismatch: {n_groups} unique resindices in "
            f"atom selection vs {n_expected} expected residues/molecules."
        )
    sort_order = np.argsort(local_idx, kind="stable")
    local_idx_sorted = local_idx[sort_order]
    group_starts = np.searchsorted(local_idx_sorted, np.arange(n_groups))
    first_atom_idx = sort_order[group_starts]  # one anchor atom per group, for PBC unwrap
    logger.info("[%s] %d atoms grouped into %d residues/molecules", label, resindices.size, n_groups)
    return {
        "local_idx": local_idx, "n_groups": n_groups,
        "sort_order": sort_order, "group_starts": group_starts,
        "first_atom_idx": first_atom_idx,
    }


def unwrap_positions(positions, group_meta, box):
    """Minimum-image unwrap of each residue/molecule relative to its own
    first atom (by file order), so a group split across a periodic
    boundary doesn't corrupt its center of mass or extent.

    This is what a "Residue sphere radius is unusually large" warning
    (previously, hundreds of Angstrom -- roughly box-scale) indicates:
    AtomGroup.center_of_mass(compound=...) does NOT unwrap split
    molecules, so if the input trajectory wraps individual atoms into the
    primary cell independently (rather than keeping whole
    residues/molecules together, e.g. no `-pbc mol`/`-pbc whole`
    equivalent applied upstream), a residue's own atoms can end up on
    opposite sides of the box. Correcting this here makes the calculation
    correct regardless of what PBC treatment (if any) the trajectory
    already received -- it only assumes each individual residue/molecule
    is small compared to the box, which is true by construction (a single
    amino acid or small solvent molecule vs. a multi-hundred-Angstrom
    simulation cell).

    Uses MDAnalysis's own minimize_vectors (correct for triclinic boxes,
    not just orthorhombic) rather than a manual round() formula.
    """
    if box is None or np.any(np.asarray(box[:3]) <= 0):
        return positions
    anchor = positions[group_meta["first_atom_idx"]][group_meta["local_idx"]]
    delta = minimize_vectors(positions - anchor, box)
    return anchor + delta


def rigid_align_chains(positions, chain_meta, box):
    """Rigidly shift each chain/segment into the same periodic image as a
    single reference chain (the one containing the very first atom).

    unwrap_positions (above) corrects atoms independently relative to a
    per-GROUP anchor, which is right for a single small, compact group
    (one residue) but wrong here: GROMACS's typical per-molecule wrapping
    (`-pbc mol`) wraps each chain independently by its own centroid, so
    for a multi-chain assembly (e.g. a 60-subunit capsid) different,
    non-covalently-linked chains can each land in a different periodic
    image even though every chain is individually whole -- "monomers
    jumped across boundaries". Applying unwrap_positions again at chain
    granularity would NOT fix this correctly (it would independently
    minimum-image every atom relative to one anchor point, which can
    re-split an already-whole residue whose atoms straddle the boundary
    relative to that distant anchor). Instead this computes ONE
    minimum-image translation per chain (from that chain's centroid to
    the reference chain's centroid) and applies it uniformly to every
    atom of the chain, preserving whatever internal structure the
    per-residue unwrap already established.

    Expects `positions` to already be residue-level unwrapped (see
    unwrap_positions) and `chain_meta` built the same way as any other
    group_meta, but keyed by chain/segment (e.g. segindices or
    chainIDs) rather than residue.
    """
    if box is None or np.any(np.asarray(box[:3]) <= 0):
        return positions
    n_chains = chain_meta["n_groups"]
    local_idx = chain_meta["local_idx"]
    counts = np.bincount(local_idx, minlength=n_chains).astype(float)
    centroid = np.empty((n_chains, 3), dtype=float)
    for dim in range(3):
        centroid[:, dim] = np.bincount(local_idx, weights=positions[:, dim], minlength=n_chains) / counts
    reference = centroid[0]  # group 0 = chain containing the first atom in file order
    raw_offset = centroid - reference
    shift = minimize_vectors(raw_offset, box) - raw_offset
    return positions + shift[local_idx]


def resid_reset_chain_key(protein_ag):
    """Alternative chain-boundary signal, independent of segid/chainID
    metadata: a new chain starts wherever resid decreases relative to the
    previous atom. True for most GROMACS/CHARMM-GUI-prepared multi-chain
    topologies, which number residues 1..N independently per chain rather
    than globally across the whole assembly.

    Added after a real capsid run: segid/chainID metadata silently merged
    two physically separate monomers into one group there, which
    rigid_align_chains then could not correctly separate (it can only
    apply ONE rigid shift per group -- two genuinely different chains
    sharing a group get averaged/shifted together incorrectly). resid
    resets are a purely topological signal that doesn't depend on that
    metadata being right, and correctly identified all 60 chains on that
    dataset when segid/chainID did not. See the chain-candidate diagnostic
    logged at startup, and prefer this signal when it's usable.

    Returns a per-atom chain index array, or None if the signal looks
    unusable (resid never decreases -- e.g. globally continuous residue
    numbering across the whole selection, in which case this can't find
    any boundaries).
    """
    resids = protein_ag.resids.astype(np.int64)
    is_boundary = np.concatenate([[True], np.diff(resids) < 0])
    if int(is_boundary.sum()) <= 1:
        return None
    return np.cumsum(is_boundary) - 1


def unwrap_protein_positions(raw_positions, res_meta, chain_meta, box):
    """Full PBC correction for the protein selection: per-residue unwrap
    (fixes a residue split across the boundary) followed by rigid
    per-chain realignment (fixes a whole monomer sitting in a different
    periodic image than the rest of the assembly). Order matters --
    residue-level must run first so each chain's rigid shift is computed
    from already-cohesive residue positions. Used for every place the
    protein's coordinates are read (the per-frame calculation and the
    reference-frame PDB output) so both stay consistent by construction.
    """
    res_unwrapped = unwrap_positions(raw_positions, res_meta, box)
    return rigid_align_chains(res_unwrapped, chain_meta, box)


def mass_weighted_com(positions, masses, group_meta):
    """Per-group (residue/molecule) center of mass via bincount -- a fast,
    vectorized grouped weighted sum, used instead of
    AtomGroup.center_of_mass(compound=...) so it can run on already
    PBC-unwrapped positions instead of the raw (possibly split) ones."""
    n_groups = group_meta["n_groups"]
    local_idx = group_meta["local_idx"]
    total_mass = np.bincount(local_idx, weights=masses, minlength=n_groups)
    com = np.empty((n_groups, 3), dtype=float)
    for dim in range(3):
        com[:, dim] = np.bincount(local_idx, weights=masses * positions[:, dim], minlength=n_groups)
    com /= total_mass[:, None]
    return com


def group_max_reduce(values, group_meta):
    """Per-group max via sort + reduceat -- vectorized equivalent of
    np.maximum.at (which doesn't vectorize well at large atom counts)."""
    sorted_vals = values[group_meta["sort_order"]]
    return np.maximum.reduceat(sorted_vals, group_meta["group_starts"])


def residue_radius(protein_pos_unwrapped, res_com, group_meta, protein_vdw, logger=None):
    """Conservative per-residue sphere radius:
        max_{atoms j in r} ( ||pos_j - res_com[r]|| + vdw[j] )
    i.e. the same "atom is a VdW sphere" idea as the original whole-protein
    method, applied to bound a residue instead of a single atom. Expects
    already PBC-unwrapped positions (see unwrap_positions) -- otherwise a
    residue split across the box boundary produces a spuriously huge
    "radius" (this is exactly the failure mode the >15 A warning below
    used to catch before the unwrap fix; kept as a sanity check in case a
    genuinely malformed topology slips through some other way).
    """
    delta = protein_pos_unwrapped - res_com[group_meta["local_idx"]]
    atom_reach = np.linalg.norm(delta, axis=1) + protein_vdw
    res_radius = group_max_reduce(atom_reach, group_meta)
    if logger is not None and res_radius.max() > 15.0:
        logger.warning(
            "Residue sphere radius %.2f A is unusually large (>15 A) -- "
            "check for a chain break or a non-protein atom leaking into "
            "the protein selection (should be rare now that positions are "
            "PBC-unwrapped per residue before this check).", res_radius.max()
        )
    return res_radius


def frame_residue_histograms(solvent_com, res_com, res_radius, n_res, n_bins, bin_width, r_max):
    """Per-residue distance histogram for one solvent type in one frame.

    Returns (res_hist, molecule_min_dist):
      res_hist          -- (n_res, n_bins) int counts, res_hist[r, k] = number
                            of solvent COMs whose surface distance to
                            residue r's sphere falls in bin k.
      molecule_min_dist -- (n_solvent,) float, the minimum surface distance
                            from each solvent molecule to ANY residue sphere
                            within r_max (np.inf if none found). Used only
                            for the whole-molecule validation curve -- it is
                            a byproduct of the same pair list, not an extra
                            search.
    """
    n_solvent = solvent_com.shape[0]
    res_hist = np.zeros((n_res, n_bins), dtype=np.int64)
    molecule_min_dist = np.full(n_solvent, np.inf, dtype=float)
    if n_solvent == 0 or n_res == 0:
        return res_hist, molecule_min_dist

    cutoff = r_max + res_radius.max()
    # method='bruteforce' is faster (2-5x, benchmarked) at small
    # reference-point counts, but _bruteforce_capped computes the FULL
    # DENSE distance_array(solvent_com, res_com) before applying the
    # cutoff mask -- memory scales with n_solvent * n_res regardless of
    # how many pairs actually survive. MDAnalysis's own auto-selection
    # (_determine_method) refuses to pick it once that product reaches
    # 1e8; we use a 2x-safety-margin threshold below that (5e7 elements
    # -> <=400MB for the dense float64 matrix) so bruteforce only ever
    # runs when it's provably cheap, and fall back to the default
    # heuristic's grid-based method (which scales with actual output size,
    # not n_solvent * n_res) otherwise. This matters because a full
    # multi-chain assembly (tens of thousands of residues x a
    # proportionally large solvent count) blows well past 1e8 -- forcing
    # bruteforce unconditionally OOM-killed a real job at ~124 GB for a
    # single rank.
    _BRUTEFORCE_SAFE_LIMIT = 5e7
    n_pairs_bound = n_solvent * n_res
    method = "bruteforce" if n_pairs_bound < _BRUTEFORCE_SAFE_LIMIT else None
    pairs, dists = capped_distance(
        solvent_com, res_com, cutoff, box=None, return_distances=True, method=method
    )
    if pairs.size == 0:
        return res_hist, molecule_min_dist

    sol_idx = pairs[:, 0]
    res_idx = pairs[:, 1]
    adjusted = np.clip(dists - res_radius[res_idx], 0.0, None)

    # cutoff was inflated by the GLOBAL max residue radius so a single
    # capped_distance call could serve every residue; for residues with a
    # smaller radius this admits candidates whose true adjusted distance
    # exceeds r_max, which we now drop (equivalent to "farther than we
    # care about", same semantics as the original search_cutoff).
    in_range = adjusted < r_max
    sol_idx = sol_idx[in_range]
    res_idx = res_idx[in_range]
    adjusted = adjusted[in_range]

    if adjusted.size:
        np.minimum.at(molecule_min_dist, sol_idx, adjusted)

        bin_idx = np.minimum((adjusted / bin_width).astype(np.int64), n_bins - 1)
        combined = res_idx * n_bins + bin_idx
        counts = np.bincount(combined, minlength=n_res * n_bins)
        res_hist = counts.reshape(n_res, n_bins)

    return res_hist, molecule_min_dist


def gamma_from_cumulative_counts(n1_r, n3_r, n1_total, n3_total):
    """Eq. 3 of Shukla et al., vectorized over any leading shape.

    n1_r, n3_r: cumulative local counts (..., n_bins). n1_total, n3_total:
    scalars (whole-system totals -- same bulk reservoir regardless of which
    residue/molecule's local domain we're evaluating).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bulk_ratio = (n3_total - n3_r) / (n1_total - n1_r)
        gamma = n3_r - n1_r * bulk_ratio
    return gamma


def find_rstar(r_values, gamma_mean, tail_frac=0.3, tol_frac=0.05):
    """Heuristic starting estimate for r* (plateau onset).

    Scans r from large to small; r* is the smallest r beyond which
    Gamma_23(r) stays within tol_frac of the plateau estimate (the median
    of the outermost tail_frac of bins) for the rest of the curve.

    This is a CONVENIENCE STARTING POINT ONLY. The source procedure
    describes identifying r* by inspection of the curve -- always check
    this against the saved plot before trusting the reported Gamma_23,
    especially for residues with sparse/noisy local counts.
    """
    n = len(gamma_mean)
    tail_n = max(3, int(n * tail_frac))
    finite_tail = gamma_mean[-tail_n:][np.isfinite(gamma_mean[-tail_n:])]
    if finite_tail.size == 0:
        return r_values[-1], len(r_values) - 1
    plateau_est = np.median(finite_tail)
    tol = max(tol_frac * abs(plateau_est), 1e-6)

    for i in range(n):
        window = gamma_mean[i:]
        window = window[np.isfinite(window)]
        if window.size == 0:
            continue
        if np.all(np.abs(window - plateau_est) <= tol):
            return r_values[i], i
    return r_values[-1], n - 1  # no clean plateau found -> flag via max r


def format_eta(seconds):
    """Human-readable HH:MM:SS from a float seconds value (or '--:--:--' if
    not yet estimable, e.g. before any frames have completed)."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def split_frame_range(n_frames, rank, size):
    """Contiguous frame block for this rank (first `n_frames % size` ranks
    get one extra frame). Returns (start, stop) as a half-open range;
    (n, n) i.e. an empty range if this rank has more workers than frames."""
    base = n_frames // size
    rem = n_frames % size
    start = rank * base + min(rank, rem)
    stop = start + base + (1 if rank < rem else 0)
    return start, stop


def build_charge_profile(r_values, n_excipient_r_whole, starting_charge, excipient_name):
    """Optional diagnostic: cumulative net charge Q(r) = starting_charge +
    (formal ion charge) * n_excipient_r(t), time-averaged, using the
    whole-molecule (residue-sphere-derived) local counts.

    This is NOT folded into the Gamma_23 result -- it is a separate
    electroneutrality/Donnan bookkeeping diagnostic, only meaningful for the
    ionic excipients (NA, CL), and only as accurate as the supplied
    starting_charge (net protein formal charge).
    """
    if excipient_name not in ION_FORMAL_CHARGE:
        return None
    z = ION_FORMAL_CHARGE[excipient_name]
    return r_values, starting_charge + z * n_excipient_r_whole


# --------------------------------------------------------------------------
# Hybrid-36 encoding + a minimal standalone PDB writer.
#
# A full AAV capsid (~240,000 atoms) exceeds the legacy PDB format's
# 99999-atom decimal serial-number field (columns 7-11) -- and potentially
# a chain's residue count could exceed the 9999 resSeq field (columns
# 23-26) too. Hybrid-36 (Grosse-Kunstleve et al.; used by PyMOL, ChimeraX,
# Phenix/CCTBX) extends both fields into base-36 once they overflow, while
# leaving anything that fits in plain decimal untouched -- so small
# structures round-trip identically to ordinary PDB numbering. We write
# the PDB ourselves (rather than via MDAnalysis's writer) so this is
# guaranteed regardless of whatever atom-count ceiling that writer may or
# may not enforce internally.
# --------------------------------------------------------------------------
_HY36_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HY36_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"


def _hy36_encode_pure(digits, value):
    if value == 0:
        return digits[0]
    n = len(digits)
    out = []
    while value:
        value, rem = divmod(value, n)
        out.append(digits[rem])
    return "".join(reversed(out))


def hy36encode(width, value):
    """Hybrid-36 encode an integer into a fixed-width PDB field.

    width=5 for the atom serial field (plain-decimal capacity 99999),
    width=4 for resSeq (plain-decimal capacity 9999). Values within the
    classic decimal range are returned exactly as "%<width>d" would
    (right-justified, space-padded); values beyond it switch to
    upper-case then lower-case base-36, each guaranteed to fill exactly
    `width` characters by construction.
    """
    i = value
    lower_bound = 1 - 10 ** (width - 1)
    if i < lower_bound:
        raise ValueError(f"value {value} too small for hybrid-36 width {width}")
    if i < 10 ** width:
        return ("%" + str(width) + "d") % i
    i -= 10 ** width
    if i < 26 * 36 ** (width - 1):
        i += 10 * 36 ** (width - 1)
        return _hy36_encode_pure(_HY36_UPPER, i)
    i -= 26 * 36 ** (width - 1)
    if i < 26 * 36 ** (width - 1):
        i += 10 * 36 ** (width - 1)
        return _hy36_encode_pure(_HY36_LOWER, i)
    raise ValueError(f"value {value} too large for hybrid-36 width {width}")


def _pdb_atom_name_field(name, element):
    """4-character atom-name field, PDB's standard justification rule:
    left-justified from column 13 for 4-char names or 2-char elements,
    otherwise shifted one space right (element symbol lands on column 14)."""
    name = name.strip()
    if len(name) >= 4:
        return name[:4]
    if len(element.strip()) >= 2:
        return name.ljust(4)[:4]
    return (" " + name).ljust(4)[:4]


def write_pdb_hybrid36(atomgroup, tempfactors, out_path, logger, box_dimensions=None):
    """Write a minimal ATOM-record PDB with hybrid-36 serial/resSeq fields.

    Bypasses MDAnalysis's PDBWriter entirely -- this is the one place a
    whole-capsid run (~240,000 atoms) would otherwise hit the format's
    99999-atom ceiling. `tempfactors` is a per-atom array (same order as
    atomgroup) written into the B-factor column.
    """
    names = atomgroup.names
    resnames = atomgroup.resnames
    resids = atomgroup.resids
    positions = atomgroup.positions
    elements = [guess_element(n) for n in names]

    if hasattr(atomgroup, "chainIDs"):
        chain_ids = list(atomgroup.chainIDs)
    elif hasattr(atomgroup, "segids"):
        chain_ids = list(atomgroup.segids)
    else:
        chain_ids = ["A"] * len(atomgroup)
    chain_ids = [(str(c).strip()[:1] or "A") for c in chain_ids]

    n_atoms = len(atomgroup)
    if n_atoms > 99999:
        logger.info(
            "%d atoms exceeds the classic PDB 99999-atom limit -- writing "
            "hybrid-36 serial numbers (read natively by PyMOL/ChimeraX/"
            "Phenix; tools that only understand plain-decimal PDB will "
            "misparse rows beyond atom 99999, same as they would for any "
            "oversized PDB file).", n_atoms,
        )
    max_resid = int(resids.max()) if n_atoms else 0
    if max_resid > 9999:
        logger.info(
            "max resid %d exceeds the classic PDB 9999 resSeq limit -- "
            "writing hybrid-36 resSeq numbers.", max_resid,
        )

    tf_clipped = np.clip(np.nan_to_num(tempfactors, nan=0.0), -999.99, 999.99)
    if not np.array_equal(tf_clipped, np.nan_to_num(tempfactors, nan=0.0)):
        logger.warning("Some B-factor (Gamma_23) values were clipped to "
                        "+/-999.99 to fit the PDB tempFactor field width.")

    with open(out_path, "w") as fh:
        fh.write("REMARK   generated by pic_calculation.py\n")
        fh.write("REMARK   B-factor column = per-residue Gamma_23\n")
        fh.write("REMARK   atom serial / resSeq use hybrid-36 numbering above 99999 / 9999\n")
        if box_dimensions is not None:
            a, b, c, alpha, beta, gamma = box_dimensions
            fh.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n"
                      % (a, b, c, alpha, beta, gamma))
        for i in range(n_atoms):
            name_field = _pdb_atom_name_field(names[i], elements[i])
            serial = hy36encode(5, i + 1)
            resseq = hy36encode(4, int(resids[i]))
            x, y, z = positions[i]
            fh.write(
                "ATOM  %5s %-4s %-3.3s %1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n"
                % (serial, name_field, resnames[i], chain_ids[i], resseq,
                   x, y, z, 1.00, tf_clipped[i], elements[i])
            )
        fh.write("END\n")


class RunningStats:
    """Accumulates sum / sum-of-squares / finite-count for an array of fixed
    shape across frames, without ever storing per-frame data. Keeps memory
    at O(shape) regardless of trajectory length, and makes the final MPI
    reduction a few fixed-size arrays instead of a gather of the whole
    trajectory's intermediate results."""

    def __init__(self, shape):
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.float64)

    def update(self, values):
        finite = np.isfinite(values)
        self.sum += np.where(finite, values, 0.0)
        self.sumsq += np.where(finite, values * values, 0.0)
        self.count += finite


# ==========================================================================
# MPI driver
# ==========================================================================

def setup_logging(rank):
    logger = logging.getLogger("pic")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        f"%(asctime)s [rank {rank:>3}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gro", required=True, help="Topology/coordinate file (md.gro)")
    p.add_argument("--xtc", required=True, help="Trajectory file (md_noPBC.xtc)")
    p.add_argument("--excipient", required=True, choices=["NA", "CL", "SUC"],
                    help="Residue name of the excipient/cosolvent of interest")
    p.add_argument("--radius", required=True, type=float,
                    help="Radius of selection R_max in Angstrom: the max "
                         "distance from each residue's surface to bin out "
                         "to. Must comfortably exceed the expected r* "
                         "plateau (15-20 A is typical) but need not "
                         "approach the box edge.")
    p.add_argument("--charge", type=float, default=0.0,
                    help="Starting (net protein formal) charge, used only "
                         "for the optional charge-balance diagnostic when "
                         "--excipient is NA or CL. Leave at 0 for SUC or if "
                         "not needed.")
    p.add_argument("--water-resname", default="SOL",
                    help="Residue name for water (default: SOL)")
    p.add_argument("--protein-selection", default="protein",
                    help="MDAnalysis selection string for the protein (default: 'protein')")
    p.add_argument("--bin-width", type=float, default=0.1,
                    help="Histogram bin width in Angstrom (default 0.1 A, "
                         "matching the source procedure)")
    p.add_argument("--output-dir", default="pic_results",
                    help="Directory for output files (default: pic_results)")
    p.add_argument("--stride", type=int, default=1,
                    help="Process every Nth frame (default 1 = all frames). "
                         "Useful for a quick test run before the full job.")
    p.add_argument("--max-frames", type=int, default=None,
                    help="Cap total frames processed (after stride), for "
                         "quick sanity-check runs. Default: all frames.")
    p.add_argument("--reference-frame", type=int, default=0,
                    help="Frame index whose coordinates are used for the "
                         "B-factor-colored PDB output (default: 0)")
    p.add_argument("--log-every", type=int, default=200,
                    help="Log progress every N local frames per rank (default 200)")
    p.add_argument("--n-checkpoints", type=int, default=20,
                    help="Number of synchronized cluster-wide aggregate "
                         "progress/ETA reports over the course of the run "
                         "(default 20, i.e. roughly every 5%%).")
    return p.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger = setup_logging(rank)

    args = parse_args()
    t_start = time.time()

    if rank == 0:
        logger.info("=" * 70)
        logger.info("PIC (Gamma_23) calculation, per-residue -- Shukla et al. 2009 method")
        logger.info("gro=%s xtc=%s excipient=%s radius=%.2f A charge=%.3f",
                     args.gro, args.xtc, args.excipient, args.radius, args.charge)
        logger.info("MPI ranks: %d%s", size, "" if _HAS_REAL_MPI else " (no MPI runtime -- serial fallback)")
        for f in (args.gro, args.xtc):
            if not os.path.isfile(f):
                logger.error("Input file not found: %s", f)
                comm.Abort(1)
        os.makedirs(args.output_dir, exist_ok=True)
    comm.Barrier()

    # ---- topology / selections (every rank does this independently; cheap
    # relative to per-frame work, avoids a broadcast step, and each rank
    # needs its own open trajectory handle for frame I/O regardless) ----
    u = mda.Universe(args.gro, args.xtc)
    protein = u.select_atoms(args.protein_selection)
    water = u.select_atoms(f"resname {args.water_resname}")
    excipient = u.select_atoms(f"resname {args.excipient}")

    for ag, label in [(protein, "protein"), (water, "water"), (excipient, "excipient")]:
        validate_masses(ag, label, logger)

    protein_vdw = get_protein_vdw_radii(protein, logger)
    protein_masses = protein.masses
    water_masses = water.masses
    excipient_masses = excipient.masses

    n_res = protein.residues.n_residues
    res_resids = protein.residues.resids
    res_resnames = protein.residues.resnames
    n1_total = water.residues.n_residues
    n3_total = excipient.residues.n_residues

    # Grouping metadata (local index, sort order, PBC-unwrap anchor atom)
    # is topology-only -- computed once here, reused every frame below.
    protein_meta = build_group_meta(protein.resindices, n_res, "protein", logger)
    water_meta = build_group_meta(water.resindices, n1_total, "water", logger)
    excipient_meta = build_group_meta(excipient.resindices, n3_total, "excipient", logger)

    # Chain/segment-level grouping, for rigid_align_chains (fixes whole
    # monomers landing in different periodic images -- see that function's
    # docstring). Multiple candidate signals are computed and cross-logged:
    # segid/chainID metadata is NOT always reliable for this -- on a real
    # capsid run it silently merged two physically separate monomers into
    # one group (both still reported a plausible-looking distinct-group
    # count), which rigid_align_chains then could not correctly separate,
    # since it can only apply one rigid shift per group. resid-reset (see
    # resid_reset_chain_key) is an independent, purely-topological signal
    # that doesn't depend on that metadata being correct, and is preferred
    # whenever it's usable.
    chain_candidates = [("segindices", protein.segindices)]
    if hasattr(protein, "chainIDs"):
        chain_candidates.append(("chainIDs", protein.chainIDs))
    resid_reset_key = resid_reset_chain_key(protein)
    if resid_reset_key is not None:
        chain_candidates.append(("resid-reset", resid_reset_key))

    if rank == 0:
        for cand_label, cand_key in chain_candidates:
            uniq, group_sizes = np.unique(cand_key, return_counts=True)
            logger.info(
                "[protein] chain-grouping candidate '%s': %d distinct groups, "
                "member counts min=%d max=%d mean=%.1f (uniform sizes are a "
                "good sign for a multi-copy assembly like a capsid)",
                cand_label, uniq.size, group_sizes.min(), group_sizes.max(), group_sizes.mean(),
            )

    resid_reset_entry = next((c for c in chain_candidates if c[0] == "resid-reset"), None)
    if resid_reset_entry is not None:
        chain_label, chain_key = resid_reset_entry
    else:
        chain_label, chain_key = max(chain_candidates, key=lambda kv: np.unique(kv[1]).size)
    n_chains = np.unique(chain_key).size
    if rank == 0:
        logger.info("[protein] using '%s' for chain-level PBC realignment (%d distinct chains/segments)",
                    chain_label, n_chains)
        if n_chains == 1:
            logger.warning(
                "Only 1 chain/segment detected for the protein selection -- "
                "if this is a multi-copy assembly (e.g. a capsid), chain-level "
                "PBC realignment cannot separate copies and will be a no-op. "
                "Check that the topology carries per-chain segid/chainID info, "
                "or that resid numbering resets per chain."
            )
    protein_chain_meta = build_group_meta(chain_key, n_chains, "protein-chain", logger)

    if rank == 0:
        logger.info("protein atoms=%d residues=%d  water molecules=%d  %s molecules=%d",
                     protein.n_atoms, n_res, n1_total, args.excipient, n3_total)
        if n3_total == 0:
            logger.error("Zero %s residues found -- check --excipient / topology naming.",
                          args.excipient)
            comm.Abort(1)

    n_bins = int(round(args.radius / args.bin_width))
    bin_edges = np.linspace(0.0, n_bins * args.bin_width, n_bins + 1)
    r_values = bin_edges[1:]

    # ---- frame range for this rank ----
    n_frames_all = len(u.trajectory)
    frame_indices = list(range(0, n_frames_all, args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    n_frames = len(frame_indices)

    if rank == 0:
        logger.info("trajectory frames: %d total, %d selected (stride=%d, max_frames=%s)",
                     n_frames_all, n_frames, args.stride, args.max_frames)
        frames_per_rank = n_frames / size if size > 0 else n_frames
        if size > 1 and frames_per_rank < MIN_USEFUL_FRAMES_PER_RANK:
            suggested = max(1, -(-n_frames // MIN_USEFUL_FRAMES_PER_RANK))  # ceil div
            logger.warning(
                "Only %.1f frames/rank at %d ranks for this %d-frame run -- "
                "per-rank fixed cost (Universe/topology load, MPI startup) "
                "will dominate over actual compute (~%.2fs/frame). For a run "
                "this size, --ntasks=%d (or fewer) would likely finish just "
                "as fast while using less of the allocation; the requested "
                "rank count matters more for much longer trajectories.",
                frames_per_rank, size, n_frames, SECONDS_PER_FRAME_ESTIMATE, suggested,
            )

    start, stop = split_frame_range(n_frames, rank, size)
    my_frames = frame_indices[start:stop]
    logger.info("assigned %d frames (local index %d:%d)", len(my_frames), start, stop)

    # ---- accumulators (fixed size, independent of trajectory length) ----
    gamma_res_stats = RunningStats((n_res, n_bins))
    whole_stats = RunningStats((n_bins,))
    n3_r_whole_stats = RunningStats((n_bins,))  # for the charge diagnostic mean
    n_local = len(my_frames)

    # Aggregate (cluster-wide) progress via periodic BLOCKING Allreduce.
    #
    # CRITICAL CORRECTNESS REQUIREMENT: every rank must call Allreduce the
    # same number of times, in the same order -- MPI collectives are matched
    # by call sequence, not by tag or content. Use max_local (largest
    # n_local across all ranks) to define ONE shared checkpoint schedule
    # every rank iterates over identically, so ranks that run out of frames
    # early still call Allreduce at the same checkpoints as everyone else.
    max_local = comm.allreduce(n_local, op=MPI.MAX)
    N_CHECKPOINTS = max(1, args.n_checkpoints)
    if max_local > 0:
        checkpoint_iters_set = set(
            max(0, int(round((k + 1) * max_local / N_CHECKPOINTS)) - 1)
            for k in range(N_CHECKPOINTS)
        )
        checkpoint_iters_set.add(max_local - 1)  # always report the final round
    else:
        checkpoint_iters_set = set()
    loop_t0 = time.time()

    for i in range(max_local):
        if i < n_local:
            fidx = my_frames[i]
            u.trajectory[fidx]
            box = u.dimensions  # may fluctuate frame-to-frame under NPT

            # DIAGNOSTIC: PBC correction (unwrap_protein_positions/unwrap_positions)
            # deliberately bypassed here -- using raw, uncorrected positions --
            # to isolate whether chain-jumping is coming from the trajectory
            # itself (trjconv) or from this script's own correction. Revert by
            # restoring the unwrap_protein_positions/unwrap_positions calls.
            protein_pos = protein.positions
            res_com = mass_weighted_com(protein_pos, protein_masses, protein_meta)
            res_radius = residue_radius(
                protein_pos, res_com, protein_meta, protein_vdw,
                logger=logger if i == 0 else None,
            )

            water_pos = water.positions
            water_com = mass_weighted_com(water_pos, water_masses, water_meta)
            exc_pos = excipient.positions
            exc_com = mass_weighted_com(exc_pos, excipient_masses, excipient_meta)

            w_hist, w_mol_min = frame_residue_histograms(
                water_com, res_com, res_radius, n_res, n_bins, args.bin_width, args.radius)
            e_hist, e_mol_min = frame_residue_histograms(
                exc_com, res_com, res_radius, n_res, n_bins, args.bin_width, args.radius)

            n1_r_res = np.cumsum(w_hist, axis=1)
            n3_r_res = np.cumsum(e_hist, axis=1)
            gamma_res = gamma_from_cumulative_counts(n1_r_res, n3_r_res, n1_total, n3_total)
            gamma_res_stats.update(gamma_res)

            # whole-molecule validation curve: nearest-residue-sphere distance
            # per solvent molecule, histogrammed exactly like the original
            # whole-protein method.
            w_hist_whole, _ = np.histogram(w_mol_min[np.isfinite(w_mol_min)], bins=bin_edges)
            e_hist_whole, _ = np.histogram(e_mol_min[np.isfinite(e_mol_min)], bins=bin_edges)
            n1_r_whole = np.cumsum(w_hist_whole)
            n3_r_whole = np.cumsum(e_hist_whole)
            gamma_whole = gamma_from_cumulative_counts(n1_r_whole, n3_r_whole, n1_total, n3_total)
            whole_stats.update(gamma_whole)
            n3_r_whole_stats.update(n3_r_whole.astype(float))

            if (i + 1) % args.log_every == 0 or i == n_local - 1:
                elapsed_loop = time.time() - loop_t0
                rate = (i + 1) / elapsed_loop if elapsed_loop > 0 else 0.0
                remaining = n_local - (i + 1)
                eta_local = remaining / rate if rate > 0 else None
                logger.info(
                    "frame %d/%d (%.1f%%, global frame %d) | rate=%.2f fr/s | "
                    "local ETA=%s",
                    i + 1, n_local, 100.0 * (i + 1) / n_local, fidx,
                    rate, format_eta(eta_local),
                )

        if i in checkpoint_iters_set:
            my_contribution = min(i + 1, n_local)
            total_done = comm.allreduce(my_contribution, op=MPI.SUM)
            if rank == 0:
                frac = total_done / n_frames if n_frames > 0 else 0.0
                elapsed_total = time.time() - t_start
                rate_total = total_done / elapsed_total if elapsed_total > 0 else 0.0
                eta_global = (n_frames - total_done) / rate_total if rate_total > 0 else None
                logger.info(
                    "[AGGREGATE] %d/%d frames complete cluster-wide (%.1f%%) | "
                    "aggregate rate=%.2f fr/s | global ETA=%s | elapsed=%s",
                    total_done, n_frames, 100.0 * frac, rate_total,
                    format_eta(eta_global), format_eta(elapsed_total),
                )

    # ---- reduce accumulators onto rank 0 (fixed-size arrays, not a
    # trajectory-length gather) ----
    gamma_res_sum = comm.reduce(gamma_res_stats.sum, op=MPI.SUM, root=0)
    gamma_res_sumsq = comm.reduce(gamma_res_stats.sumsq, op=MPI.SUM, root=0)
    gamma_res_count = comm.reduce(gamma_res_stats.count, op=MPI.SUM, root=0)

    gamma_whole_sum = comm.reduce(whole_stats.sum, op=MPI.SUM, root=0)
    gamma_whole_sumsq = comm.reduce(whole_stats.sumsq, op=MPI.SUM, root=0)
    gamma_whole_count = comm.reduce(whole_stats.count, op=MPI.SUM, root=0)

    n3_r_whole_sum = comm.reduce(n3_r_whole_stats.sum, op=MPI.SUM, root=0)
    n3_r_whole_count = comm.reduce(n3_r_whole_stats.count, op=MPI.SUM, root=0)

    if rank != 0:
        return

    def mean_sem_from_parts(s, ssq, cnt):
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s / cnt
            var = np.maximum(ssq / cnt - mean * mean, 0.0)
            sem = np.sqrt(var / np.maximum(cnt, 1.0))
        mean = np.where(cnt > 0, mean, np.nan)
        sem = np.where(cnt > 0, sem, np.nan)
        return mean, sem

    gamma_res_mean, gamma_res_sem = mean_sem_from_parts(gamma_res_sum, gamma_res_sumsq, gamma_res_count)
    gamma_whole_mean, gamma_whole_sem = mean_sem_from_parts(gamma_whole_sum, gamma_whole_sumsq, gamma_whole_count)
    n3_r_whole_mean = np.where(n3_r_whole_count > 0, n3_r_whole_sum / np.maximum(n3_r_whole_count, 1.0), np.nan)

    logger.info("aggregated per-residue Gamma_23(r,t): shape %s over %d rank-frames",
                gamma_res_mean.shape, int(gamma_res_count.max()) if gamma_res_count.size else 0)

    # ---- r* and final Gamma_23 per residue ----
    rstar_res = np.full(n_res, np.nan)
    gamma23_res = np.full(n_res, np.nan)
    gamma23_res_sem = np.full(n_res, np.nan)
    for ridx in range(n_res):
        curve = gamma_res_mean[ridx]
        if not np.any(np.isfinite(curve)):
            continue
        rstar, rstar_idx = find_rstar(r_values, curve)
        rstar_res[ridx] = rstar
        gamma23_res[ridx] = curve[rstar_idx]
        gamma23_res_sem[ridx] = gamma_res_sem[ridx, rstar_idx]

    rstar_whole, rstar_whole_idx = find_rstar(r_values, gamma_whole_mean)
    gamma23_whole = gamma_whole_mean[rstar_whole_idx]
    gamma23_whole_sem = gamma_whole_sem[rstar_whole_idx]

    logger.info("AUTO-DETECTED whole-molecule r* = %.2f A (CONFIRM VISUALLY against the saved plot)",
                rstar_whole)
    logger.info("Whole-molecule Gamma_23 (residue-sphere approximation) = %.4f +/- %.4f",
                gamma23_whole, gamma23_whole_sem)
    finite_res = np.isfinite(gamma23_res)
    if finite_res.any():
        logger.info(
            "Per-residue Gamma_23: min=%.4f max=%.4f mean=%.4f (n=%d residues with data)",
            np.nanmin(gamma23_res), np.nanmax(gamma23_res), np.nanmean(gamma23_res),
            int(finite_res.sum()),
        )

    # ---- save raw data ----
    payload = {
        "metadata": {
            "gro": args.gro, "xtc": args.xtc, "excipient": args.excipient,
            "radius": args.radius, "bin_width": args.bin_width,
            "starting_charge": args.charge, "water_resname": args.water_resname,
            "n1_total_water": int(n1_total), "n3_total_excipient": int(n3_total),
            "n_residues": int(n_res), "n_frames": int(n_frames), "n_mpi_ranks": size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method_note": (
                "Per-residue Gamma_23 uses a residue-sphere approximation "
                "(center = residue COM, radius = farthest own atom + its "
                "VdW radius). Whole-molecule Gamma_23 is the min-across-"
                "residue-spheres derivative of the same pair list, an "
                "approximation of (not identical to) the atom-level "
                "Shukla et al. whole-protein quantity."
            ),
        },
        "r_values": r_values,
        "residue": {
            "resids": np.asarray(res_resids), "resnames": np.asarray(res_resnames),
            "gamma_r_mean": gamma_res_mean, "gamma_r_sem": gamma_res_sem,
            "rstar": rstar_res, "gamma_23_final": gamma23_res, "gamma_23_final_sem": gamma23_res_sem,
        },
        "whole_molecule": {
            "gamma_r_mean": gamma_whole_mean, "gamma_r_sem": gamma_whole_sem,
            "rstar_auto": rstar_whole, "gamma_23_final": gamma23_whole, "gamma_23_final_sem": gamma23_whole_sem,
        },
    }

    charge_profile = build_charge_profile(r_values, n3_r_whole_mean, args.charge, args.excipient)
    if charge_profile is not None:
        payload["charge_profile_r"] = charge_profile[0]
        payload["charge_profile_Qr"] = charge_profile[1]

    pkl_path = os.path.join(args.output_dir, f"pic_{args.excipient}.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)
    logger.info("saved raw data -> %s", pkl_path)

    csv_res_path = os.path.join(args.output_dir, f"pic_{args.excipient}_per_residue.csv")
    with open(csv_res_path, "w") as fh:
        fh.write("resid,resname,gamma23,gamma23_sem,rstar_angstrom\n")
        for ridx in range(n_res):
            fh.write(f"{res_resids[ridx]},{res_resnames[ridx]},"
                     f"{gamma23_res[ridx]:.6g},{gamma23_res_sem[ridx]:.6g},{rstar_res[ridx]:.3g}\n")
    logger.info("saved per-residue table -> %s", csv_res_path)

    csv_whole_path = os.path.join(args.output_dir, f"pic_{args.excipient}_vs_r.csv")
    np.savetxt(csv_whole_path, np.column_stack([r_values, gamma_whole_mean, gamma_whole_sem]),
               header="r_angstrom,gamma23_mean,gamma23_sem", delimiter=",", comments="")
    logger.info("saved whole-molecule r-profile -> %s", csv_whole_path)

    # ---- plot: whole-molecule r-profile + per-residue bar ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(r_values, gamma_whole_mean, color="#2b6cb0", lw=1.5, label=r"$\Gamma_{23}(r)$")
    ax1.fill_between(r_values, gamma_whole_mean - gamma_whole_sem, gamma_whole_mean + gamma_whole_sem,
                      color="#2b6cb0", alpha=0.25, label="SEM")
    ax1.axvline(rstar_whole, color="#c53030", ls="--", lw=1.2, label=f"auto r* = {rstar_whole:.1f} A")
    ax1.axhline(gamma23_whole, color="#2f855a", ls=":", lw=1.0,
                label=f"Gamma_23 = {gamma23_whole:.3f}")
    ax1.set_xlabel("r, distance from nearest residue surface (A)")
    ax1.set_ylabel(r"$\Gamma_{23}(r)$")
    ax1.set_title(f"Whole-molecule (validation): {args.excipient}")
    ax1.legend(frameon=False, fontsize=8)

    colors = np.where(np.nan_to_num(gamma23_res) >= 0, "#2f855a", "#c53030")
    ax2.bar(res_resids, np.nan_to_num(gamma23_res), color=colors, width=1.0)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_xlabel("residue ID")
    ax2.set_ylabel(r"$\Gamma_{23}$ (per residue)")
    ax2.set_title(f"Per-residue Gamma_23: {args.excipient}")

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, f"pic_{args.excipient}_summary.png")
    fig.savefig(plot_path, dpi=150)
    logger.info("saved plot -> %s", plot_path)

    # ---- PDB with B-factor = per-residue Gamma_23 ----
    # Written by our own hy36-aware writer (not AtomGroup.write()) so a
    # whole-capsid structure (~240,000 atoms) never hits the standard PDB
    # format's 99999-atom serial-number ceiling.
    u.trajectory[args.reference_frame]
    protein_ref = u.select_atoms(args.protein_selection)
    ref_box = u.dimensions
    # DIAGNOSTIC: PBC correction bypassed here too -- writing raw, uncorrected
    # positions -- so this PDB shows exactly what's in the trajectory file
    # itself, with no interference from this script. See the per-frame loop
    # above for the matching bypass and how to revert both.
    per_atom_gamma = gamma23_res[protein_meta["local_idx"]]
    per_atom_gamma = np.where(np.isfinite(per_atom_gamma), per_atom_gamma, 0.0)
    pdb_path = os.path.join(args.output_dir, f"pic_{args.excipient}_bfactor.pdb")
    write_pdb_hybrid36(protein_ref, per_atom_gamma, pdb_path, logger,
                        box_dimensions=ref_box if ref_box is not None else None)
    logger.info("saved B-factor-colored structure (frame %d) -> %s", args.reference_frame, pdb_path)

    elapsed = time.time() - t_start
    logger.info("TOTAL RUNTIME: %.1f s (%.2f frames/s aggregate)",
                 elapsed, n_frames / elapsed if elapsed > 0 else float("nan"))
    logger.info("=" * 70)
    logger.info("RESULT: whole-molecule Gamma_23(%s) = %.4f +/- %.4f (r* = %.2f A, auto-detected)",
                 args.excipient, gamma23_whole, gamma23_whole_sem, rstar_whole)
    logger.info("Per-residue breakdown: %s", csv_res_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()