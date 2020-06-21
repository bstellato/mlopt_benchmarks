#!/bin/zsh
#SBATCH -c 1
#SBATCH -n 40
#SBATCH -N 1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:volta:1 -p gpu
#SBATCH -o /home/gridsan/stellato/results/online/obstacle/obstacle_train_%A_N%a.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bartolomeo.stellato@gmail.com


# Activate environment
. "/home/gridsan/stellato/miniconda/etc/profile.d/conda.sh"
conda activate online

# module load gurobi/8.0.1
export GRB_LICENSE_FILE="/home/software/gurobi/gurobi.lic"


# Run actual script
HDF5_USE_FILE_LOCKING=FALSE python online_optimization/obstacle/training.py --n_obstacles $SLURM_ARRAY_TASK_ID

# Process data and put together with other results
# python online_optimization/portfolio/process_data.py
