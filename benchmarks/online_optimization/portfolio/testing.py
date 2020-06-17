# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())

import online_optimization.portfolio.simulation.settings as stg
from online_optimization.portfolio.learning_data import learning_data, sample_around_points, get_dimensions
from online_optimization.portfolio.utils import create_problem, get_problem_dimensions
import mlopt
import numpy as np
import scipy.sparse as spa
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
import pandas as pd
import datetime as dt
import logging
import argparse


np.random.seed(1)

STORAGE_DIR = "/home/gridsan/stellato/results/online/portfolio/"

desc = 'Online Portfolio Example Testing'

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--sparsity', type=int, default=5, metavar='N',
                    help='sparsity level (default: Full)')
arguments = parser.parse_args()
k_sparsity = arguments.sparsity

EXAMPLE_NAME = STORAGE_DIR + '/portfolio_%d_' % k_sparsity

n_test = 10000

# Define cost weights
lambda_cost = {'risk': stg.RISK_COST,
               'borrow': stg.BORROW_WEIGHT_COST,
               #  'norm0_trade': stg.NORM0_TRADE_COST,
               #  'norm1_trade': stg.NORM1_TRADE_COST,
               'norm1_trade': 0.01}


# Define initial value
t_start = dt.date(2008, 1, 1)
t_end = dt.date(2013, 1, 1)
T_periods = 1

# Get problem dimensions
n, m = get_dimensions()

# Define mlopt problem
problem = create_problem(n, m, T_periods, k=k_sparsity,
                         lambda_cost=lambda_cost)

# Load model
m = mlopt.Optimizer.from_file(EXAMPLE_NAME + 'model')
m.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')
m.cache_factors()   # Cache KKT systems for speed


# Get test data
df_history = learning_data(t_start=t_start,
                           t_end=t_end,
                           T_periods=T_periods,
                           lambda_cost=lambda_cost)
idx_pick = np.arange(len(df_history))
np.random.shuffle(idx_pick)  # Debug, remove shuffle
n_history_train = int(len(df_history) * 0.8)
test_idx = idx_pick[n_history_train:]
df_history_test = df_history.iloc[test_idx].reset_index(drop=True)
df = sample_around_points(df_history_test, n_total=n_test)

# Solve with miqp solver and get result
results_full = m._problem.solve_parametric(df,
                                           parallel=True,
                                           message="Solve Gurobi")


# Solve with Gurobi Heuristic
m._problem.solver_options['MIPGap'] = 0.1  # 10% MIP Gap
results_heuristic = m._problem.solve_parametric(df,
                                                parallel=True,
                                                message="Compute " +
                                                "tight constraints " +
                                                "with heuristic MIP Gap 10 %% " +
                                                "for test set")
m._problem.solver_options.pop('MIPGap')  # Remove MIP Gap option


# Iterate over k (best)
results = []
results_detail = []
k_max = 101
for k in [1] + list(range(10, min(k_max, m._learner.n_classes), 10)):

    print("n_best = %d" % k)

    # Change n_best in learner
    m._learner.options['n_best'] = k

    # Get results with our solver
    results_k, results_detail_k = m.performance(
        df, results_test=results_full, results_heuristic=results_heuristic,
        parallel=False, use_cache=True)

    # Add horizon
    results_k['K'] = k_sparsity
    results_detail_k['K'] = [k_sparsity] * len(results_detail_k)

    results.append(results_k)
    results_detail.append(results_detail_k)

    # Store
    df_results = pd.concat(results, axis=1).T.sort_values(by=['K', 'n_best'])
    df_results.to_csv(EXAMPLE_NAME + 'performance.csv')
    df_results_detail = pd.concat(results_detail, ignore_index=True).sort_values(by=['K', 'n_best']).reset_index(drop=True)
    df_results_detail.to_csv(EXAMPLE_NAME + 'performance_detail.csv')
