#!/usr/bin/env python3
"""
characterize_high_pic_residues.py
==================================

Characterizes the residues with the highest per-residue preferential
interaction coefficient (Gamma_23, "PIC") in a `pic_<EXCIPIENT>_bfactor.pdb`
file produced by pic_calculation.py (B-factor column = Gamma_23, all atoms
of a residue share one value).

For every residue it looks up:
  - amino acid identity, one-letter code, and side-chain class
    (aliphatic / aromatic / acidic / basic / polar-uncharged / sulfur / special)
  - Kyte-Doolittle hydropathy, side-chain volume, molecular weight
  - charge, using the ACTUAL protonation state recorded in the resname
    (e.g. HISH/HISE/ASPH/GLUH are CHARMM-style protonation variants written
    by the MD forcefield, not typos -- ASPH is a protonated, neutral Asp;
    HISH is doubly-protonated, positively charged His; etc.)

...then ranks all residues by Gamma_23 and asks whether the top-N are drawn
disproportionately from any residue type or property class, via:
  - per-type / per-class Gamma_23 summary statistics (all residues)
  - Spearman correlation of Gamma_23 against each continuous property
  - Fisher's exact enrichment test of each type/property in the top-N
    vs. the rest of the protein

Outputs (written to --output-dir):
  ranked_residues.csv          per-residue table, sorted by Gamma_23 desc.
  top_N_residues.csv           the top --top-n rows of the above
  bottom_N_residues.csv        the bottom --bottom-n rows (only if requested)
  type_summary.csv             Gamma_23 stats grouped by residue type
  class_summary.csv            Gamma_23 stats grouped by side-chain class
  correlations.csv             Spearman r/p vs. each continuous property
  enrichment.csv                Fisher's-exact enrichment of top-N by type/class/flag
  summary.txt                  plain-text digest of the above
  plots/top_residues_bar.png
  plots/gamma23_by_class_box.png
  plots/hydropathy_vs_gamma23.png
  plots/enrichment_bar.png

Usage:
  python3 characterize_high_pic_residues.py --pdb pic_results_SUC/pic_SUC_bfactor.pdb
  python3 characterize_high_pic_residues.py --pdb pic_SUC_bfactor.pdb --top-n 30 --bottom-n 15
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================================
# Amino acid property table
# ============================================================================
# hydropathy: Kyte & Doolittle 1982; volume (A^3) and MW: Zamyatnin 1972.
# charge: formal side-chain charge at "standard" protonation (His neutral).
AA_PROPERTIES = {
    #        1-letter class              hydropathy charge volume  mw
    "ALA": ("A", "Aliphatic",             1.8,        0,    88.6,   89.09),
    "ARG": ("R", "Basic",                -4.5,       +1,   173.4,  174.20),
    "ASN": ("N", "Polar_uncharged",      -3.5,        0,   114.1,  132.12),
    "ASP": ("D", "Acidic",               -3.5,       -1,   111.1,  133.10),
    "CYS": ("C", "Sulfur",                2.5,        0,   108.5,  121.16),
    "GLN": ("Q", "Polar_uncharged",      -3.5,        0,   143.8,  146.15),
    "GLU": ("E", "Acidic",               -3.5,       -1,   138.4,  147.13),
    "GLY": ("G", "Special",              -0.4,        0,    60.1,   75.07),
    "HIS": ("H", "Basic",                -3.2,        0,   153.2,  155.16),
    "ILE": ("I", "Aliphatic",             4.5,        0,   166.7,  131.17),
    "LEU": ("L", "Aliphatic",             3.8,        0,   166.7,  131.17),
    "LYS": ("K", "Basic",                -3.9,       +1,   168.6,  146.19),
    "MET": ("M", "Sulfur",                1.9,        0,   162.9,  149.21),
    "PHE": ("F", "Aromatic",              2.8,        0,   189.9,  165.19),
    "PRO": ("P", "Special",              -1.6,        0,   112.7,  115.13),
    "SER": ("S", "Polar_uncharged",      -0.8,        0,    89.0,  105.09),
    "THR": ("T", "Polar_uncharged",      -0.7,        0,   116.1,  119.12),
    "TRP": ("W", "Aromatic",             -0.9,        0,   227.8,  204.23),
    "TYR": ("Y", "Aromatic",             -1.3,        0,   193.6,  181.19),
    "VAL": ("V", "Aliphatic",             4.2,        0,   140.0,  117.15),
}
POLAR_CLASSES = {"Polar_uncharged", "Acidic", "Basic"}
AROMATIC_RESNAMES = {"PHE", "TYR", "TRP", "HIS"}

# Non-standard / protonation-variant resnames written by CHARMM-style
# forcefields -> (parent 3-letter code, protonation label, charge override).
PROTONATION_VARIANTS = {
    "HISH": ("HIS", "HIP: doubly-protonated (+1)",        +1),
    "HISE": ("HIS", "HIE: neutral, N(eps)-H",               0),
    "HISD": ("HIS", "HID: neutral, N(delta)-H",             0),
    "ASPH": ("ASP", "protonated (neutral -COOH)",           0),
    "GLUH": ("GLU", "protonated (neutral -COOH)",           0),
    "LSN":  ("LYS", "neutral (deprotonated -NH2)",          0),
    "CYM":  ("CYS", "deprotonated thiolate (-1)",          -1),
    "CYX":  ("CYS", "disulfide-bonded (0)",                 0),
}


def resolve_residue(resname_raw):
    """Map a raw PDB resname to (parent_code, protonation_label, charge)."""
    resname_raw = resname_raw.strip()
    if resname_raw in PROTONATION_VARIANTS:
        parent, label, charge = PROTONATION_VARIANTS[resname_raw]
        return parent, label, charge
    if resname_raw in AA_PROPERTIES:
        charge = AA_PROPERTIES[resname_raw][2]
        return resname_raw, "standard", charge
    return None, "unknown", 0


# ============================================================================
# hybrid-36 aware PDB parsing
# ============================================================================
_HY36_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HY36_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"


def _hy36_decode(width, field):
    """Decode a fixed-width PDB integer field, falling back to hybrid-36.

    pic_calculation.py's own writer switches resSeq/serial to hybrid-36
    once a system exceeds the classic PDB 9999/99999 ceilings (documented
    in its write_pdb_hybrid36()), so plain int() alone would silently
    misparse resids on a large enough system (e.g. a capsid run).
    """
    s = field
    first = s.lstrip()[:1] or " "
    if first.isdigit() or first == "-" or first == " ":
        try:
            return int(s)
        except ValueError:
            pass
    digits = _HY36_UPPER if first in _HY36_UPPER else _HY36_LOWER
    value = 0
    for c in s:
        value = value * 36 + digits.index(c)
    if digits is _HY36_UPPER:
        return value - 10 * 36 ** (width - 1) + 10 ** width
    return value - 10 * 36 ** (width - 1) - 26 * 36 ** (width - 1) + 10 ** width


def parse_bfactor_pdb(path):
    """Read ATOM records, average the (shared) B-factor per residue.

    Returns a DataFrame: chain, resid, resname_raw, n_atoms, gamma23.
    """
    rows = {}
    with open(path, "r") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname_raw = line[17:20].strip()
            chain = line[21:22].strip() or "A"
            resid = _hy36_decode(4, line[22:26])
            try:
                bfactor = float(line[60:66])
            except ValueError:
                continue
            key = (chain, resid, resname_raw)
            acc = rows.setdefault(key, [])
            acc.append(bfactor)

    if not rows:
        raise ValueError(f"No ATOM/HETATM records with a parseable B-factor found in {path}")

    records = []
    for (chain, resid, resname_raw), values in rows.items():
        values = np.asarray(values, dtype=float)
        spread = values.max() - values.min()
        if spread > 0.02:  # atoms of one residue should share one value (0.01 A2 PDB precision)
            print(f"WARNING: residue {chain}/{resname_raw}{resid} has non-uniform B-factors "
                  f"(range {spread:.4f}) -- averaging anyway", file=sys.stderr)
        records.append((chain, resid, resname_raw, len(values), float(values.mean())))

    df = pd.DataFrame(records, columns=["chain", "resid", "resname_raw", "n_atoms", "gamma23"])
    return df.sort_values(["chain", "resid"]).reset_index(drop=True)


# ============================================================================
# Annotation
# ============================================================================
def annotate(df):
    parents, protonations, charges = [], [], []
    one_letters, classes, hydropathy, volume, mw, aromatic, polar = [], [], [], [], [], [], []

    for resname_raw in df["resname_raw"]:
        parent, label, charge = resolve_residue(resname_raw)
        parents.append(parent if parent else resname_raw)
        protonations.append(label)
        charges.append(charge)
        if parent and parent in AA_PROPERTIES:
            one_letter, cls, hyd, _base_charge, vol, m = AA_PROPERTIES[parent]
        else:
            one_letter, cls, hyd, vol, m = "X", "Unknown", np.nan, np.nan, np.nan
        one_letters.append(one_letter)
        classes.append(cls)
        hydropathy.append(hyd)
        volume.append(vol)
        mw.append(m)
        aromatic.append(parent in AROMATIC_RESNAMES if parent else False)
        polar.append(cls in POLAR_CLASSES)

    df = df.copy()
    df["resname"] = parents
    df["protonation"] = protonations
    df["one_letter"] = one_letters
    df["residue_class"] = classes
    df["charge"] = charges
    df["is_charged"] = df["charge"] != 0
    df["is_aromatic"] = aromatic
    df["is_polar"] = polar
    df["hydropathy"] = hydropathy
    df["volume_A3"] = volume
    df["mw_Da"] = mw
    df = df.sort_values("gamma23", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


# ============================================================================
# Group summaries
# ============================================================================
def group_summary(df, by):
    g = df.groupby(by)["gamma23"]
    out = g.agg(n="count", mean="mean", std="std", min="min", max="max").reset_index()
    out["sem"] = out["std"] / np.sqrt(out["n"])
    out = out.sort_values("mean", ascending=False).reset_index(drop=True)
    return out[[by, "n", "mean", "sem", "std", "min", "max"]]


def correlations(df):
    props = ["hydropathy", "volume_A3", "mw_Da", "charge"]
    rows = []
    for p in props:
        sub = df[[p, "gamma23"]].dropna()
        if len(sub) < 3:
            continue
        r, pval = stats.spearmanr(sub[p], sub["gamma23"])
        rows.append((p, len(sub), r, pval))
    return pd.DataFrame(rows, columns=["property", "n", "spearman_r", "p_value"])


def enrichment(df, top_n, alpha):
    """Fisher's-exact enrichment of each type/class/flag in the top-N vs. the rest."""
    top_mask = df["rank"] <= top_n
    n_top = int(top_mask.sum())
    n_rest = int((~top_mask).sum())

    tests = []
    for col, label_prefix in [("resname", "type"), ("residue_class", "class")]:
        for val in df[col].dropna().unique():
            in_group = df[col] == val
            tests.append((f"{label_prefix}:{val}", in_group))
    for col in ["is_charged", "is_aromatic", "is_polar"]:
        tests.append((f"flag:{col}", df[col].astype(bool)))

    rows = []
    for label, in_group in tests:
        a = int((top_mask & in_group).sum())          # top & has property
        b = n_top - a                                   # top & not
        c = int((~top_mask & in_group).sum())           # rest & has property
        d = n_rest - c                                   # rest & not
        if a + c == 0:
            continue
        odds_ratio, pval = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        freq_top = a / n_top if n_top else np.nan
        freq_rest = c / n_rest if n_rest else np.nan
        fold = (freq_top / freq_rest) if freq_rest > 0 else np.inf
        rows.append((label, a, n_top, c, n_rest, freq_top, freq_rest, fold, odds_ratio, pval))

    out = pd.DataFrame(rows, columns=[
        "group", "n_in_top", "n_top", "n_in_rest", "n_rest",
        "freq_in_top", "freq_in_rest", "fold_enrichment", "odds_ratio", "p_value",
    ])
    out["significant"] = out["p_value"] < alpha
    return out.sort_values("p_value").reset_index(drop=True)


# ============================================================================
# Plots
# ============================================================================
CLASS_COLORS = {
    "Aliphatic": "#4a5568", "Aromatic": "#805ad5", "Acidic": "#c53030",
    "Basic": "#2b6cb0", "Polar_uncharged": "#2f855a", "Sulfur": "#d69e2e",
    "Special": "#718096", "Unknown": "#a0aec0",
}


def plot_top_residues(df, top_n, out_path):
    top = df[df["rank"] <= top_n].sort_values("gamma23", ascending=True)
    labels = [f"{r.resname}{r.resid}{'*' if r.protonation != 'standard' else ''}"
              for r in top.itertuples()]
    colors = [CLASS_COLORS.get(c, "#000000") for c in top["residue_class"]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(top))))
    ax.barh(labels, top["gamma23"], color=colors)
    ax.set_xlabel(r"$\Gamma_{23}$ (PIC)")
    ax.set_title(f"Top {top_n} residues by Gamma_23")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS.values()]
    ax.legend(handles, CLASS_COLORS.keys(), fontsize=7, frameon=False,
              loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_box(df, out_path):
    classes = df.groupby("residue_class")["gamma23"].mean().sort_values(ascending=False).index
    data = [df.loc[df["residue_class"] == c, "gamma23"].values for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, tick_labels=list(classes), showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(CLASS_COLORS.get(c, "#a0aec0"))
        patch.set_alpha(0.7)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel(r"$\Gamma_{23}$ (PIC)")
    ax.set_title("Gamma_23 distribution by side-chain class (all residues)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hydropathy_scatter(df, out_path):
    sub = df.dropna(subset=["hydropathy"])
    colors = [CLASS_COLORS.get(c, "#000000") for c in sub["residue_class"]]
    r, pval = stats.spearmanr(sub["hydropathy"], sub["gamma23"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["hydropathy"], sub["gamma23"], c=colors, s=22, alpha=0.8, edgecolor="none")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("Kyte-Doolittle hydropathy")
    ax.set_ylabel(r"$\Gamma_{23}$ (PIC)")
    ax.set_title(f"Hydropathy vs. Gamma_23 (Spearman r={r:.2f}, p={pval:.1e})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_enrichment(enr_df, alpha, out_path, max_groups=20):
    sub = enr_df.dropna(subset=["fold_enrichment"])
    sub = sub[np.isfinite(sub["fold_enrichment"])].head(max_groups)
    if sub.empty:
        return
    sub = sub.sort_values("fold_enrichment")
    colors = ["#c53030" if s else "#a0aec0" for s in sub["significant"]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(sub))))
    ax.barh(sub["group"], sub["fold_enrichment"], color=colors)
    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(f"fold enrichment in top-N (red = p < {alpha})")
    ax.set_title("Top/rest enrichment by residue type / property")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================================
# Report
# ============================================================================
def write_summary(df, top_n, bottom_n, type_summary, class_summary, corr_df, enr_df, alpha, path):
    lines = []
    lines.append("=" * 72)
    lines.append("HIGH-PIC RESIDUE CHARACTERIZATION")
    lines.append("=" * 72)
    lines.append(f"Residues analyzed: {len(df)}")
    lines.append(f"Gamma_23 range: {df['gamma23'].min():.3f} to {df['gamma23'].max():.3f}")
    lines.append("")

    lines.append(f"-- Top {top_n} residues by Gamma_23 --")
    top = df[df["rank"] <= top_n]
    for r in top.itertuples():
        tag = f" [{r.protonation}]" if r.protonation != "standard" else ""
        lines.append(f"  #{r.rank:>3} {r.resname}{r.resid:<5} ({r.one_letter}) "
                      f"gamma23={r.gamma23:8.3f}  class={r.residue_class:<16}{tag}")
    lines.append("")

    lines.append(f"Top-{top_n} residue-type composition: "
                  + ", ".join(f"{k}={v}" for k, v in
                               top["resname"].value_counts().items()))
    lines.append(f"Top-{top_n} class composition: "
                  + ", ".join(f"{k}={v}" for k, v in
                               top["residue_class"].value_counts().items()))
    lines.append("")

    lines.append("-- Gamma_23 by residue type (all residues, sorted by mean) --")
    lines.append(type_summary.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    lines.append("")

    lines.append("-- Gamma_23 by side-chain class (all residues, sorted by mean) --")
    lines.append(class_summary.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    lines.append("")

    lines.append("-- Spearman correlation of continuous properties vs. Gamma_23 (all residues) --")
    lines.append(corr_df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    lines.append("")

    lines.append(f"-- Fisher's-exact enrichment in top-{top_n} vs. rest (alpha={alpha}) --")
    sig = enr_df[enr_df["significant"]]
    if sig.empty:
        lines.append("  No group reached significance at this alpha.")
    else:
        lines.append(sig.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    lines.append("")

    if bottom_n:
        lines.append(f"-- Bottom {bottom_n} residues by Gamma_23 (most cosolvent-excluding) --")
        bottom = df.tail(bottom_n).sort_values("gamma23")
        for r in bottom.itertuples():
            lines.append(f"  #{r.rank:>3} {r.resname}{r.resid:<5} ({r.one_letter}) "
                          f"gamma23={r.gamma23:8.3f}  class={r.residue_class:<16}")
        lines.append("")

    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(text)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", required=True, help="pic_<EXCIPIENT>_bfactor.pdb from pic_calculation.py")
    p.add_argument("--top-n", type=int, default=20, help="how many highest-Gamma_23 residues to report")
    p.add_argument("--bottom-n", type=int, default=0, help="also report the N lowest-Gamma_23 residues")
    p.add_argument("--alpha", type=float, default=0.05, help="significance threshold for enrichment tests")
    p.add_argument("--output-dir", default=None,
                    help="default: 'pic_residue_characterization' next to --pdb")
    p.add_argument("--no-plots", action="store_true", help="skip generating PNG plots")
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.pdb)),
                                                "pic_residue_characterization")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Parsing {args.pdb} ...")
    raw = parse_bfactor_pdb(args.pdb)
    df = annotate(raw)
    n_unknown = int((df["residue_class"] == "Unknown").sum())
    if n_unknown:
        print(f"WARNING: {n_unknown} residue(s) had an unrecognized resname and were "
              f"left uncharacterized (class=Unknown): "
              f"{sorted(df.loc[df['residue_class'] == 'Unknown', 'resname_raw'].unique())}",
              file=sys.stderr)

    df.to_csv(os.path.join(out_dir, "ranked_residues.csv"), index=False)

    top_n = min(args.top_n, len(df))
    df[df["rank"] <= top_n].to_csv(os.path.join(out_dir, "top_N_residues.csv"), index=False)

    bottom_n = min(args.bottom_n, len(df))
    if bottom_n:
        df.tail(bottom_n).to_csv(os.path.join(out_dir, "bottom_N_residues.csv"), index=False)

    type_summary = group_summary(df, "resname")
    class_summary = group_summary(df, "residue_class")
    corr_df = correlations(df)
    enr_df = enrichment(df, top_n, args.alpha)

    type_summary.to_csv(os.path.join(out_dir, "type_summary.csv"), index=False)
    class_summary.to_csv(os.path.join(out_dir, "class_summary.csv"), index=False)
    corr_df.to_csv(os.path.join(out_dir, "correlations.csv"), index=False)
    enr_df.to_csv(os.path.join(out_dir, "enrichment.csv"), index=False)

    text = write_summary(df, top_n, bottom_n, type_summary, class_summary, corr_df, enr_df,
                          args.alpha, os.path.join(out_dir, "summary.txt"))
    print(text)

    if not args.no_plots:
        plot_top_residues(df, top_n, os.path.join(plots_dir, "top_residues_bar.png"))
        plot_class_box(df, os.path.join(plots_dir, "gamma23_by_class_box.png"))
        plot_hydropathy_scatter(df, os.path.join(plots_dir, "hydropathy_vs_gamma23.png"))
        plot_enrichment(enr_df, args.alpha, os.path.join(plots_dir, "enrichment_bar.png"))
        print(f"Plots saved to {plots_dir}")

    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
