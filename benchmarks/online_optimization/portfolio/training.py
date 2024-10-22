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

STORAGE_DIR = "/scratch/gpfs/bs37/mlopt_research/results/online/portfolio/"


if __name__ == '__main__':

    desc = 'Online Portfolio Example'

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--sparsity', type=int, default=5, metavar='N',
                        help='sparsity level (default: Full)')
    arguments = parser.parse_args()
    k = arguments.sparsity

    EXAMPLE_NAME = STORAGE_DIR + '/portfolio_%d_' % k

    n_train = 100000
    #  n_test = 10000

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
    problem = create_problem(n, m, T_periods, k=k,
                             lambda_cost=lambda_cost)

    m_mlopt = mlopt.Optimizer(problem, parallel=True)

    # Check if learning data already there
    if not os.path.isfile(EXAMPLE_NAME + 'data.pkl'):
        # Get data for learning
        print("Get learning data by simulating with no integer variables (faster)")
        df_history = learning_data(t_start=t_start,
                                   t_end=t_end,
                                   T_periods=T_periods,
                                   lambda_cost=lambda_cost)


        # Split dataset properly and shuffle
        idx_pick = np.arange(len(df_history))
        # DEBUG REMOVE SHUFFLE
        #  np.random.shuffle(idx_pick)
        n_history_train = int(len(df_history) * 0.8)

        train_idx = idx_pick[:n_history_train]
        #  test_idx = idx_pick[n_history_train:]
        df_history_train = df_history.iloc[train_idx].reset_index(drop=True)

        #  # DEBUG use test dataset as training
        #  #  df_history_test = df_history.iloc[train_idx].reset_index(drop=True)
        #  df_history_test = df_history.iloc[test_idx].reset_index(drop=True)

        # Sample around points
        df_train = sample_around_points(df_history_train,
                                        n_total=n_train)

        # Get problem dimensions
        n, m = get_problem_dimensions(df_train)

        # Get samples
        print("Get samples in parallel")
        m_mlopt.get_samples(df_train,
                            parallel=True,
                            filter_strategies=False)
        m_mlopt.save_training_data(EXAMPLE_NAME + 'data.pkl',
                                   delete_existing=True)

    else:
        # Load data
        print("Loading data frome file")
        m_mlopt.load_training_data(EXAMPLE_NAME + 'data.pkl')

    # Filter strategies?
    m_mlopt.filter_strategies(parallel=True)
    m_mlopt.save_training_data(EXAMPLE_NAME + 'data_filtered.pkl',
                              delete_existing=True)

    # Learn
    m_mlopt.train(learner=mlopt.PYTORCH,
                  n_best=10,
                  filter_strategies=False,
                  parallel=True)


    # Save model
    m_mlopt.save(EXAMPLE_NAME + 'model', delete_existing=True)



    # Testing (remove)

    #  df_test = sample_around_points(df_history_test,
    #                                 n_total=n_test)
    #  res_general, res_detail = m_mlopt.performance(df_test,
    #                                                parallel=False,
    #                                                use_cache=True)
    #
    #  res_general.to_csv(EXAMPLE_NAME + "test_general.csv",
    #                     header=True)
    #  res_detail.to_csv(EXAMPLE_NAME + "test_detail.csv")

    # Evaluate loop performance
    # TODO: Add!
