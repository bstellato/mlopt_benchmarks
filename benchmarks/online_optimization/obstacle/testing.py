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



import matplotlib.pyplot as plt


STORAGE_DIR = "/scratch/gpfs/bs37/mlopt_research/results/online/obstacle/"


desc = 'Obstacle Avoidance Example'
seed_test = 1
n_test = 10000
np.random.seed(seed_test)

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--n_obstacles', type=int, default=10, metavar='N', help='number of obstacles between 1 and 10 (default: 10)')

arguments = parser.parse_args()
n_obstacles = arguments.n_obstacles
print(desc, " n_obstacles = %d\n" % n_obstacles)

EXAMPLE_NAME = STORAGE_DIR + '/obstacle_%d_' % n_obstacles

# Get first n_obstacles obstacles
obstacles = u.OBSTACLES[:n_obstacles]

# Get problem
problem = u.create_problem(obstacles)

# Load model
m = mlopt.Optimizer.from_file(EXAMPLE_NAME + 'model')
m.load_training_data(EXAMPLE_NAME + 'data.pkl')
m.cache_factors()   # Cache KKT systems for speed

# Get training dataframe
df = u.sample_points(obstacles, n_test, integer=False)

# Solve with miqp solver and get result
results_full = m._problem.solve_parametric(df,
                                           parallel=True,
                                           message="Solve Gurobi")


# Solve with Gurobi Heuristic
#  m._problem.solver_options['MIPGap'] = 0.1  # 10% MIP Gap
# Focus on feasibility
m._problem.solver_options['MIPFocus'] = 1
# Limit time to one second
m._problem.solver_options['TimeLimit'] = 1.
results_heuristic = m._problem.solve_parametric(df,
                                                parallel=True,
                                                message="Compute " +
                                                "tight constraints " +
                                                "with heuristic MIP Gap 10 %% " +
                                                "for test set")
#  m._problem.solver_options.pop('MIPGap')  # Remove MIP Gap option
m._problem.solver_options.pop('MIPFocus')
m._problem.solver_options.pop('TimeLimit')


# Iterate over k
results = []
results_detail = []
k_max = 101
for k in [1, 20, 40, 60, 80, 100]:

    print("n_best = %d" % k)

    # Change n_best in learner
    m._learner.options['n_best'] = k

    # Get results with our solver
    results_k, results_detail_k = m.performance(
        df, results_test=results_full, results_heuristic=results_heuristic,
        parallel=False, use_cache=True)

    # Add horizon
    results_k['n_obstacles'] = n_obstacles
    results_detail_k['n_obstacles'] = [n_obstacles] * len(results_detail_k)

    results.append(results_k)
    results_detail.append(results_detail_k)

    # Store
    df_results = pd.concat(results, axis=1).T.sort_values(by=['n_obstacles', 'n_best'])
    df_results.to_csv(EXAMPLE_NAME + 'performance.csv')
    df_results_detail = pd.concat(results_detail, ignore_index=True).sort_values(by=['n_obstacles', 'n_best']).reset_index(drop=True)
    df_results_detail.to_csv(EXAMPLE_NAME + 'performance_detail.csv')
