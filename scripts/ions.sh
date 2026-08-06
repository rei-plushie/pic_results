#!/bin/bash
#SBATCH --job-name=ion_calc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=mem-queue-multi-az
#SBATCH --nodelist=mem-queue-multi-az-dy-r8i-24xl-2
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
echo "MPI ranks: ${SLURM_NTASKS}   CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "=========================================================="


DIR="10ns_aav1ph6salt30suc1"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph8salt30suc1"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph6salt30suc10"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph8salt30suc10"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph6salt200suc1"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph8salt200suc1"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph6salt200suc10"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph8salt200suc10"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph6salt110suc8"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph73salt170suc5"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph73salt30suc12"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="10ns_aav1ph6salt30suc12"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}


DIR="aav2ph73salt30"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="aav2ph73salt300"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="aav8ph6salt30"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="aav8ph6salt300"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="aav8ph8salt30"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}

DIR="aav8ph8salt300"

python ion_atmosphere.py --top ion_input/${DIR}/md_prot_suc_ions.gro --xtc ion_input/${DIR}/md_prot_suc_ions.xtc \
    --region-csv pic/${DIR}/pic_symmetry_regions/per_residue.csv --max-shell 15 --stride 2 \
    --bin-width 0.05 --output-dir ion_out/${DIR}


EXIT_CODE=$?

echo "=========================================================="
echo "Finished: $(date)   Exit code: ${EXIT_CODE}"
echo "=========================================================="

exit ${EXIT_CODE}