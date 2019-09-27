#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --time=12:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --job-name=ppo_switch_hyperparam
#SBATCH --output=/scratch/wardnade/maoc/%x-%j.out
#SBATCH --array=1-16
#SBATCH --mail-user=wardnade@mila.quebec
#SBATCH --mail-type=ALL

module load python/3.6
source $HOME/option-critic/bin/activate

python scripts/run_schedule.py --storage_dir="$1"