#!/bin/bash

# ============================
# How to submit:
# sbatch run_array.sh
#
# Check job:
# squeue -u scur0741
#
# Cancel job:
# scancel <jobid>
# 
# Check logs example:
# tail -f /home/scur0741/kun/logs/<jobid>_<taskid>.out
# 
# Copy result example:
# scp -r scur0741@snellius.surf.nl:/home/scur0741/kun/We-3-Women---Group-Project-Semester-4/news_output .
# ============================

#SBATCH --job-name=IWD_2018_Global
#SBATCH --output=/home/scur0741/kun/logs/%A_%a.out
#SBATCH --error=/home/scur0741/kun/logs/%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-4

cd /home/scur0741/kun/We-3-Women---Group-Project-Semester-4

source /home/scur0741/kun/venv-agent/bin/activate

echo "Running array job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Start time: $(date)"

python collect_articles_array.py --task-id $SLURM_ARRAY_TASK_ID

echo "End time: $(date)"