#!/bin/bash
#SBATCH --account=def-bengioy                                                                            
#SBATCH --time=10:00:00                                                                                  
#SBATCH --mem=500G                                                                                       
#SBATCH --cpus-per-task=40                                                                               
#SBATCH --ntasks=1                                                                                       
#SBATCH --job-name=multiagent-doc-train                                                                 
#SBATCH --output=/scratch/wardnade/maoc/%x-%j.out                                                        
#SBATCH --mail-user=wardnade@mila.quebec                                                                 
#SBATCH --mail-type=ALL

module load python/3.6
source $HOME/option-critic/bin/activate

python scripts/train.py --desc='doc_test' --algo='doc' --env='TEAMGrid-FourRooms-v0'
