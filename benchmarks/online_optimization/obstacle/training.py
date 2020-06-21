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



desc = 'Obstacle Avoidance Example'

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--n_obstacles', type=int, default=10, metavar='N', help='number of obstacles between 1 and 10 (default: 10)')

arguments = parser.parse_args()
n_obstacles = arguments.n_obstacles
print(desc, " n_obstacles = %d\n" % n_obstacles)

EXAMPLE_NAME = STORAGE_DIR + '/obstacle_%d_' % n_obstacles

# Problem data
n_train = 100000
seed_train = 0
np.random.seed(seed_train)

# Get first n_obstacles obstacles
obstacles = u.OBSTACLES[:n_obstacles]

# Get problem
problem = u.create_problem(obstacles)

# Create mlopt problem
m_mlopt = mlopt.Optimizer(problem,
                          Threads=1,  # to avoid issues with parallelization
                          #  MIPGap=0.05,
                          parallel=True)

# Check if learning data already there
if not os.path.isfile(EXAMPLE_NAME + 'data.pkl'):
    print("Sampling points")
    df_train = u.sample_points(obstacles, n_train)

    # Get samples
    m_mlopt.get_samples(df_train,
                        parallel=True,
                        filter_strategies=False)  # Filter strategies after saving
    m_mlopt.save_training_data(EXAMPLE_NAME + 'data.pkl',
                               delete_existing=True)
else:
    print("Loading data from file")
    m_mlopt.load_training_data(EXAMPLE_NAME + 'data.pkl')


# Filter strategies and resave
m_mlopt.filter_strategies(parallel=True)
m_mlopt.save_training_data(EXAMPLE_NAME + 'data_filtered.pkl',
                           delete_existing=True)

# Learn optimizer
m_mlopt.train(learner=mlopt.PYTORCH,
              n_best=10,
              filter_strategies=False,  # Do not filter strategies again
              #  n_train_trials=2,
              parallel=True)

# Save model
m_mlopt.save(EXAMPLE_NAME + 'model', delete_existing=True)
#
