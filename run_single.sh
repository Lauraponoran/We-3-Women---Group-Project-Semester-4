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

#SBATCH --job-name=publisher1
#SBATCH --output=/home/scur0741/kun/logs/%j.out
#SBATCH --error=/home/scur0741/kun/logs/%j.err
#SBATCH --time=200:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /home/scur0741/kun/We-3-Women---Group-Project-Semester-4

source /home/scur0741/kun/venv-agent/bin/activate

echo "Running on $(hostname)"
echo "Start time: $(date)"

python collect_articles.py --workers 4

echo "End time: $(date)"