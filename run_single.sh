#!/bin/bash
#SBATCH --job-name=publisher_single
#SBATCH --output=/home/scur0741/kun/logs/%j.out
#SBATCH --error=/home/scur0741/kun/logs/%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd /home/scur0741/kun/We-3-Women---Group-Project-Semester-4

source /home/scur0741/kun/venv-agent/bin/activate

echo "Running on $(hostname)"
echo "Start time: $(date)"

python collect_articles.py

echo "End time: $(date)"