#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=752G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/rojulien/maoc/slurm-%j.out

module load python/3.6
source /home/rojulien/maoc/bin/activate

python -m scripts.run_schedule.py --storage_dir "$1" --n_processes "$2"
