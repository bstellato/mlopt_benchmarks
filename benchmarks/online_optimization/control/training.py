# Needed for slurm
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


STORAGE_DIR = "/scratch/gpfs/bs37/mlopt_research/results/online/control/"


np.random.seed(1)


if __name__ == '__main__':

    desc = 'Online Control Example'

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--horizon', type=int, default=10, metavar='N',
                        help='horizon length (default: 10)')
    arguments = parser.parse_args()
    T_horizon = arguments.horizon

    EXAMPLE_NAME = STORAGE_DIR + '/control_%d_' % T_horizon

    # Problem data
    n_traj = 10000  # Trajectory sampling to get points
    tau = 1.0
    n_train = 100000
    n_test = n_traj  # Number of samples in test set (new trajectory)
    seed_train = 0
    seed_test = 1

    print(desc, " N = %d\n" % T_horizon)

    # Get trajectory
    P_load = u.P_load_profile(n_traj, seed=seed_train)

    # Create simulation data
    init_data = {'E': [7.7],
                 'z': [0.],
                 's': [0.],
                 'P': [],
                 'past_d': [np.zeros(T_horizon)],
                 'P_load': [P_load[:T_horizon]],
                 'sol': []}

    # Define problem
    problem, cost_function_data = u.control_problem(T_horizon, tau=tau)


    # Create mlopt problem
    m_mlopt = mlopt.Optimizer(problem,
                              parallel=True)


    # Check if learning data already there
    if not os.path.isfile(EXAMPLE_NAME + 'data.pkl'):

        print("Get learning data by simulating closed loop")
        sim_data = u.simulate_loop(problem, init_data,
                                   u.basic_loop_solve,
                                   P_load,
                                   n_traj,
                                   T_horizon)

        # Store simulation data as parameter values (avoid sol parameter)
        df = u.sim_data_to_params(sim_data)


        # Sample over balls around all the parameters
        df_train = u.sample_around_points(df,
                                          radius={'z_init': .4,  # .2,
                                                  #  's_init': .6,  # .2,
                                                  'P_load': 0.001,  # 0.01
                                                  },
                                          n_total=n_train)


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


    # Evaluate performance
    # # Generate test trajectory and collect points
    #  print("Simulate loop again to get trajectory points")
    #  P_load_test = u.P_load_profile(n_test, seed=seed_test)
    #
    #  sim_data_test = u.simulate_loop(problem, init_data,
    #                                  u.basic_loop_solve,
    #                                  P_load_test,
    #                                  n_test,
    #                                  T_horizon)
    #
    #  # Evaluate open-loop performance on those parameters
    #  print("Evaluate open loop performance")
    #  df_test = u.sim_data_to_params(sim_data_test)
    #  res_general, res_detail = m_mlopt.performance(df_test,
    #                                                parallel=False,
    #                                                use_cache=True)
    #
    #  res_general.to_csv(EXAMPLE_NAME + "test_general.csv",
    #                     header=True)
    #  res_detail.to_csv(EXAMPLE_NAME + "test_detail.csv")




    # Old performance evaluation
    #  print("Evaluate closed loop performance")
    #  # Loop with MPC basic function
    #  sim_data_test = u.simulate_loop(problem, init_data,
    #                                  u.basic_loop_solve,
    #                                  P_load_test,
    #                                  n_test,
    #                                  T_horizon)
    #  # Loop with predictor
    #  sim_data_mlopt = u.simulate_loop(m_mlopt, init_data,
    #                                   u.predict_loop_solve,
    #                                   P_load_test,
    #                                   n_test,
    #                                   T_horizon)
    #
    #  # Evaluate loop performance
    #  perf_solver = u.performance(cost_function_data, sim_data_test)
    #  perf_mlopt = u.performance(cost_function_data, sim_data_mlopt)
    #  res_general['perf_solver'] = perf_solver
    #  res_general['perf_mlopt'] = perf_mlopt
    #  res_general['perf_degradation_perc'] = 100 * (1. - perf_mlopt/perf_solver)
    #
    #  # Export files
    #  with open(EXAMPLE_NAME + 'sim_data_mlopt.pkl', 'wb') as handle:
    #      pickle.dump(sim_data_mlopt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #  with open(EXAMPLE_NAME + 'sim_data_test.pkl', 'wb') as handle:
    #      pickle.dump(sim_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #  res_general.to_csv(EXAMPLE_NAME + "test_general.csv",
    #                     header=True)
    #  res_detail.to_csv(EXAMPLE_NAME + "test_detail.csv")

    #  print("Plot data")
    #  u.plot_sim_data(sim_data_mlopt, T_horizon, P_load_test,
    #                  title='sim_data_mlopt',
    #                  name=EXAMPLE_NAME)
    #  u.plot_sim_data(sim_data_test, T_horizon, P_load_test,
    #                  title='sim_data_test',
    #                  name=EXAMPLE_NAME)
