#!/bin/bash
#SBATCH --job-name=pic_calc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=mem-queue-multi-az
#SBATCH --nodelist=mem-queue-multi-az-dy-r8i-24xl-1
#SBATCH --exclusive


set -eo pipefail
cd "${SLURM_SUBMIT_DIR}" || { echo "ERROR: cannot cd to SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"; exit 1; }
mkdir -p logs

EXCIPIENT="SUC"
RADIUS=10
STARTING_CHARGE=0.0
BIN_WIDTH=0.1
WATER_RESNAME="SOL"
PROTEIN_SELECTION="protein"

echo "=========================================================="
echo "PIC calculation (per-residue): ${EXCIPIENT}"
echo "Job: ${SLURM_JOB_ID}   Node: $(hostname)   Started: $(date)"
echo "GRO: ${GRO_FILE}"
echo "XTC: ${XTC_FILE}"
echo "Radius: ${RADIUS} A   Bin width: ${BIN_WIDTH} A   Charge: ${STARTING_CHARGE}"
echo "MPI ranks: ${SLURM_NTASKS}   CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "=========================================================="

module load openmpi/4.1.8
source /apps/miniconda3/bin/activate pic_env
set -u
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg


# NAME="10ns_aav1ph6salt30suc1"
# GRO_FILE="aav1/${NAME}/md.gro"
# XTC_FILE="aav1/${NAME}/md_whole.xtc"
# OUTPUT_DIR="pic/${NAME}"
# mkdir -p "${OUTPUT_DIR}"
# mpirun -np "${SLURM_NTASKS}" --bind-to core python3 pic_calculation_exterior.py \
#     --gro "${GRO_FILE}" --xtc "${XTC_FILE}" --excipient "${EXCIPIENT}" --radius "${RADIUS}" \
#     --force-rstar 10 --charge "${STARTING_CHARGE}" --bin-width "${BIN_WIDTH}" \
#     --water-resname "${WATER_RESNAME}" --protein-selection "${PROTEIN_SELECTION}" \
#     --output-dir "${OUTPUT_DIR}" --log-every 200
# python3 characterize_high_pic_residues.py --pdb "${OUTPUT_DIR}"/pic_${EXCIPIENT}_exterior_bfactor.pdb
# python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}"

NAME="10ns_aav1ph6salt30suc1"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph6salt30suc1_rerun"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph6salt30suc1_rerun2"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph6salt30suc10"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph6salt30suc12"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph6salt110suc8"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph6salt200suc1"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph6salt200suc10"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph8salt30suc1"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph8salt30suc10"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph8salt200suc1"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph8salt200suc10"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph73salt30suc12"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

NAME="10ns_aav1ph73salt170suc5"
GRO_FILE="aav1/${NAME}/md.gro"
XTC_FILE="aav1/${NAME}/md_whole.xtc"
OUTPUT_DIR="pic/${NAME}"
python3 pic_symmetry_regions.py --run "${EXCIPIENT}:${OUTPUT_DIR}/pic_${EXCIPIENT}_exterior.pkl:${GRO_FILE}" \
    --three-fold-r-min 20 --three-fold-r-max 45 \
    --five-fold-r-min 4.5 --five-fold-r-max 30 \
    --five-fold-max-angle-deg 12

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
exit ${EXIT_CODE}