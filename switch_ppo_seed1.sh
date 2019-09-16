#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --job-name=ppo_switch
#SBATCH --output=/scratch/wardnade/maoc/%x-%j.out
#SBATCH --mail-user=wardnade@mila.quebec
#SBATCH --mail-type=ALL

module load python/3.6
source $HOME/option-critic/bin/activate

python scripts/train.py --use_central_critic=False --use_always_broadcast=True --use_teamgrid=True --desc='ppo_switch' --algo='ppo' --env='TEAMGrid-Switch-v0' --tb=False --num_options=1 --experiment_dir=1