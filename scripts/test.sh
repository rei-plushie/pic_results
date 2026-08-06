#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=gen-queue-spot
#SBATCH --exclusive


