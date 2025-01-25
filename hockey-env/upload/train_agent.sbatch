#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=TiM_RL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --error=job.%A_%a-%J-%j.err
#SBATCH --output=job.%A_%a-%J-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pablo-santos.blazquez@student.uni-tuebingen.de

singularity exec container.sif python hockey-en-mallorca/first_test/Dueling_DQN_hockey_training.py

echo DONE!

