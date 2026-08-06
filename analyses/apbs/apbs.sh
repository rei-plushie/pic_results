#!/bin/bash
#SBATCH --job-name=apbs_calc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=mem-queue-multi-az
#SBATCH --nodelist=mem-queue-multi-az-dy-r8i-32xl-1
#SBATCH --exclusive

cd "${SLURM_SUBMIT_DIR}" || { echo "ERROR: cannot cd to SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"; exit 1; }
source /apps/gromacs/2025.3-gpu-mpi/bin/GMXRC.bash
source /apps/miniconda3/bin/activate apbs_env

mkdir -p logs

echo "=========================================================="
echo "PIC calculation (per-residue): ${EXCIPIENT}"
echo "Job: ${SLURM_JOB_ID}   Node: $(hostname)   Started: $(date)"
echo "=========================================================="

apbs apbs_8_8_30.in