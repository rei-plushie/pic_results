#!/bin/bash
#SBATCH --job-name=hbond_calc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=mem-queue-multi-az
#SBATCH --nodelist=mem-queue-multi-az-dy-r8i-24xl-1
#SBATCH --exclusive

cd "${SLURM_SUBMIT_DIR}" || { echo "ERROR: cannot cd to SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"; exit 1; }
source /apps/gromacs/2025.3-gpu-mpi/bin/GMXRC.bash
source /apps/miniconda3/bin/activate pic_env

mkdir -p logs

echo "=========================================================="
echo "PIC calculation (per-residue): ${EXCIPIENT}"
echo "Job: ${SLURM_JOB_ID}   Node: $(hostname)   Started: $(date)"
echo "GRO: ${GRO_FILE}"
echo "XTC: ${XTC_FILE}"
echo "Radius: ${RADIUS} A   Bin width: ${BIN_WIDTH} A   Charge: ${STARTING_CHARGE}"
echo "MPI ranks: ${SLURM_NTASKS}   CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "=========================================================="


DIR="10ns_aav1ph6salt30suc1"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph8salt30suc1"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph6salt30suc10"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph8salt30suc10"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph6salt200suc1"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph8salt200suc1"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph6salt200suc10"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph8salt200suc10"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph6salt110suc8"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph73salt170suc5"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph73salt30suc12"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}

DIR="10ns_aav1ph6salt30suc12"
python sucrose_hbond_regions.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv \
    --excipient SUC --outer-only --output-dir hbond/${DIR}


EXIT_CODE=$?

echo "=========================================================="
echo "Finished: $(date)   Exit code: ${EXIT_CODE}"
echo "Results in: ${OUTPUT_DIR}/"
echo "  pic_${EXCIPIENT}.pkl                 -- full per-residue + whole-molecule arrays"
echo "  pic_${EXCIPIENT}_per_residue.csv     -- resid, resname, Gamma_23, SEM, r* per residue"
echo "  pic_${EXCIPIENT}_vs_r.csv            -- whole-molecule r-profile (validation)"
echo "  pic_${EXCIPIENT}_summary.png         -- whole-molecule r-profile + per-residue bar plot"
echo "  pic_${EXCIPIENT}_bfactor.pdb         -- structure with B-factor = per-residue Gamma_23"
echo "=========================================================="
echo "REMINDER: auto-detected r* (whole-molecule and per-residue) is a"
echo "starting estimate only -- confirm the plateau visually in the .png"
echo "before trusting Gamma_23."

exit ${EXIT_CODE}