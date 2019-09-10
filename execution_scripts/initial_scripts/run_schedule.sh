#!/bin/bash
#SBATCH -o slurm-%j.out
# usage: `sbatch --qos=unkillable run_schedule.sh`
source ~/.bashrc
source activate maoc
python -m scripts.run_schedule --storage_dir "$1" --pbar False
