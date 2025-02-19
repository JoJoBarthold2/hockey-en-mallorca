#!/bin/bash
#SBATCH --job-name=n-step-6-UwU
#SBATCH --partition=week
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3-0:50:00
#SBATCH --error=job.DDDQN_combined_%A_%a-%J-%j.err
#SBATCH --output=job.DDDQN_combined_%A_%a-%J-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=valentin.schmidt@student.uni-tuebingen.de

singularity exec updated_container.sif bash -c "
    source /venv/bin/activate && \
    python3 hockey-en-mallorca/src/hockey-env/prio_agent_training_self_play.py
"

echo "DONE!"
