#!/bin/zsh
#SBATCH -c 10
#SBATCH --mem-per-cpu=12G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p sched_mit_sloan_interactive
#SBATCH --time=2-00:00
#SBATCH -o results/output_transportation_%J.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# Mandatory for slurm stuff
source /etc/profile

# Activate environment
# conda activate python37

# module load gurobi/8.0.1
# export GRB_LICENSE_FILE="/home/software/gurobi/gurobi.lic"

python transportation/transportation.py

