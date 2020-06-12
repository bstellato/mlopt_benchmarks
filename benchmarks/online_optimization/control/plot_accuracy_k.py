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



# Plot
desc = 'Online Control Example'

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--horizon', type=int, default=10, metavar='N',
                    help='horizon length (default: 10)')
arguments = parser.parse_args()
T_horizon = arguments.horizon


# Load model
m = mlopt.Optimizer.from_file(EXAMPLE_NAME + 'model', delete_existing=True)
# Load training data
m.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')

# Solve with miqp solver and get result

# Iterate over k
for k in range(1, 30):
    # Change n_best in learner
    m._learner.options['n_best'] = k

    # get results with our solver

    # Compare suboptimality

    # Compare infeasibility






