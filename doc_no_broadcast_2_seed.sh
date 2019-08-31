#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --job-name=doc_no_broad
#SBATCH --output=/scratch/wardnade/maoc/%x-%j.out
#SBATCH --mail-user=wardnade@mila.quebec
#SBATCH --mail-type=ALL

module load python/3.6
source $HOME/option-critic/bin/activate

python scripts/train.py --seed=124 --use_central_critic=True --use_always_broadcast=False --use_teamgrid=True --desc='doc_no_broad' --algo='doc' --env='TEAMGrid-FourRooms-v0' --tb=False --num_options=3 --experiment_dir=1