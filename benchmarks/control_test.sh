#!/bin/zsh
#SBATCH -c 1
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:volta:1 -p gpu
#SBATCH -o /home/gridsan/stellato/results/online/control/control_test_%A_N%a.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Mandatory for slurm stuff
source /etc/profile

# Activate environment
. "/home/gridsan/stellato/miniconda/etc/profile.d/conda.sh"
conda activate online

export GRB_LICENSE_FILE="/home/software/gurobi/gurobi.lic"

# Process data and put together with other results
HDF5_USE_FILE_LOCKING=FALSE python online_optimization/control/testing.py --horizon $SLURM_ARRAY_TASK_ID
