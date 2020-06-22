# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())
import cvxpy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import online_optimization.obstacle.utils as u
import mlopt
from copy import deepcopy


obstacles = u.OBSTACLES
p_0 = [9, 1]

problem, variables = u.create_problem(obstacles, T=60, return_variables=True)

# Load model
m = mlopt.Optimizer.from_file('/home/gridsan/stellato/results/online/obstacle/obstacle_10_model')
m.load_training_data('/home/gridsan/stellato/results/online/obstacle/obstacle_10_data.pkl')
m.cache_factors()   # Cache KKT systems for speed
m._learner.options['n_best'] = 100


# Extract variables
p = variables['p']
p_goal = variables['p_goal']
p_min = variables['p_min']
p_max = variables['p_max']

# Assign parameter values
p_init = problem.parameters()[0]
p_init.value = p_0

# Solve
problem.solve(solver=cp.GUROBI, verbose=True)
p_gurobi = p.value
print("Gurobi solution time: %.4f" % problem.solver_stats.solve_time)

# Solve with MLOPT
res_mlopt = m.solve(pd.DataFrame({'p_init': [p_0]}))
p_mlopt = [v for v in m.variables() if v.shape == (60, 2)][0].value
print("MLOPT solution time: %.4f" % res_mlopt['time'])


# Plot with obstacles
fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(patches.Rectangle(
        xy=p_min,  # point of origin.
        width=(p_max - p_min)[0],
        height=(p_max - p_min)[1],
        linewidth=1,
        linestyle='dotted',
        color='k',
        fill=False
    ))

for o in obstacles:
    ax.add_patch(patches.Rectangle(
            xy=o['min'],  # point of origin.
            width=o['max'][0] - o['min'][0],
            height=o['max'][1] - o['min'][1],
            linewidth=0.5,
            color='k',
            fill=False
        ))

ax.scatter(p_gurobi[:, 0], p_gurobi[:, 1], marker='o',
           s=8, facecolors='none', color='k')
ax.scatter(p_mlopt[:, 0], p_mlopt[:, 1], marker='s',
           s=8, facecolors='none', color='k')
ax.plot([p_0[0]], [p_0[1]], marker='o', markersize=8, color="k")
ax.plot([p_goal[0]], [p_goal[1]], marker='*', markersize=10, color="k")

#  plt.show(block=False)
plt.axis('off')
plt.tight_layout()
plt.savefig("/home/gridsan/stellato/results/online/obstacle/obstacle.pdf")

