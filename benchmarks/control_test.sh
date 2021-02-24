#!/bin/bash
#SBATCH --job-name=controltest
#SBATCH --array=10,20,30,40,50,60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G
#SBATCH --time=4-00:00
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gpfs/bs37/mlopt_research/results/online/control/control_train_%A_N%a.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

export GRB_LICENSE_FILE=/usr/licensed/gurobi/license/gurobi.lic

module purge
module load anaconda3
conda activate mlopt39

python online_optimization/control/testing.py --horizon $SLURM_ARRAY_TASK_ID



# OLD STUFF
#!/bin/zsh
#SBATCH -c 1
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:volta:1 -p gpu
#SBATCH -o /home/gridsan/stellato/results/online/control/control_test_%A_N%a.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bartolomeo.stellato@gmail.com

# # Mandatory for slurm stuff
# source /etc/profile
#
# # Activate environment
# . "/home/gridsan/stellato/miniconda/etc/profile.d/conda.sh"
# conda activate online
#
# export GRB_LICENSE_FILE="/home/software/gurobi/gurobi.lic"
#
# # Process data and put together with other results
# HDF5_USE_FILE_LOCKING=FALSE python online_optimization/control/testing.py --horizon $SLURM_ARRAY_TASK_ID
