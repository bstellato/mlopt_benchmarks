# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())
import online_optimization.obstacle.utils as u
import numpy as np
import mlopt
import pickle
import argparse
import os
import pandas as pd



STORAGE_DIR = "/home/gridsan/stellato/results/online/obstacle"


np.random.seed(1)

desc = 'Obstacle Avoidance Example'

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--horizon', type=int, default=30, metavar='N',
                    help='horizon length (default: 30)')
arguments = parser.parse_args()
T_horizon = arguments.horizon
print(desc, " T = %d\n" % T_horizon)

EXAMPLE_NAME = STORAGE_DIR + '/obstacle_%d_' % T_horizon


# Problem data
n_train = 1000
seed_train = 0

# Get problem
problem = u.create_problem(T=T_horizon)

# Create mlopt problem
m_mlopt = mlopt.Optimizer(problem, parallel=True)

# Check if learning data already there
if not os.path.isfile(EXAMPLE_NAME + 'data.pkl'):
    # TODO: Continue from here. Get data by sampling in box
    # over a corner of the state space
