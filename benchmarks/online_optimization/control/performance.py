import os
import sys
sys.path.append(os.getcwd())
import online_optimization.control.utils as u
import numpy as np
import mlopt
import pickle
import argparse
import os
import pandas as pd


import matplotlib.pyplot as plt


STORAGE_DIR = "/home/gridsan/stellato/results/online/control"

desc = 'Online Control Example'
seed_test = 1
n_test = 10000
tau = 1.0


for T_horizon in [10, 20, 30, 40]:
    print("Horizon %d" % T_horizon)

    EXAMPLE_NAME = STORAGE_DIR + '/control_%d_' % T_horizon

    problem, cost_function_data = u.control_problem(T_horizon, tau=tau)

    # Load model
    m = mlopt.Optimizer.from_file(EXAMPLE_NAME + 'model')
    m.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')
    m.cache_factors()   # Cache KKT systems for speed


    # Get test
    P_load_test = u.P_load_profile(n_test, seed=seed_test)

    init_data = {'E': [7.7],
                 'z': [0.],
                 's': [0.],
                 'P': [],
                 'past_d': [np.zeros(T_horizon)],
                 'P_load': [P_load_test[:T_horizon]],
                 'sol': []}

    sim_data_test = u.simulate_loop(problem, init_data,
                                    u.basic_loop_solve,
                                    P_load_test,
                                    n_test,
                                    T_horizon)
    df = u.sim_data_to_params(sim_data_test)

    # Solve with miqp solver and get result
    results_full = m._problem.solve_parametric(df,
                                               parallel=True,
                                               message="Solve Gurobi")


    # Iterate over k
    results = []
    results_detail = []
    k_max = 100
    for k in [1] + list(range(5, min(k_max, m._learner.n_classes), 5)):

        print("n_best = %d" % k)

        # Change n_best in learner
        m._learner.options['n_best'] = k

        # Get results with our solver
        results_k, results_detail_k = m.performance(
            df, results_test=results_full, parallel=False, use_cache=True)

        # Add horizon
        results_k['T'] = T_horizon
        results_detail_k['T'] = [T_horizon] * len(results_detail_k)

        results.append(results_k)
        results_detail.append(results_detail_k)

        # Store
        df_results = pd.concat(results, axis=1).T.sort_values(by=['T', 'n_best'])
        df_results.to_csv(EXAMPLE_NAME + 'performance.csv')
        df_results_detail = pd.concat(results_detail, ignore_index=True).sort_values(by=['T', 'n_best']).reset_index(drop=True)
        df_results_detail.to_csv(EXAMPLE_NAME + 'performance_detail.csv')
