#!/bin/bash
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=24:00:00             # Time limit (24 hour)
#SBATCH --job-name=python_gpu_job   # Job name
#SBATCH --output=job_output_%j.log
#SBATCH --error=job_error_%j.log

# Activate the conda environment
source ~/miniconda3/bin/activate RL

# Run your Python script
python 8__reinforce_agg_100s.py

