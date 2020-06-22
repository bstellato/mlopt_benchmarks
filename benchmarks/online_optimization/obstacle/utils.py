import cvxpy as cp
import numpy as np
from control import ss, c2d
import pandas as pd
from tqdm import tqdm


OBSTACLES = [
        {'min': [-19, -16],
         'max': [-11, -4]},
        {'min': [-10, -14],
         'max': [-3.5, -2.5]},
        {'min': [-14, -2],
         'max': [-3, 7]},
        {'min': [-18, 8],
         'max': [-2.5, 16]},
        {'min': [-2, 2],
         'max': [7, 17]},
        {'min': [-2.5, -4.5],
         'max': [3, 1]},
        {'min': [-2.8, -12],
         'max': [4, -6]},
        {'min': [4, -4],
         'max': [12, 0.5]},
        {'min': [5, -15.5],
         'max': [15, -7]},
        {'min': [8, 3],
         'max': [12, 12]}
        ]



def sample_points(obstacles, n=10, p_min=[-20, -20], p_max=[20, 20], integer=True):

    list_p_0 = []

    if integer:
        samp = np.random.randint
    else:
        samp = np.random.uniform

    for i in tqdm(range(n)):
        # Varying problem parameter
        p_0 = samp(p_min, p_max)
        while not is_outside_obstacles(p_0, obstacles):
            p_0 = samp(p_min, p_max)
        list_p_0.append(p_0)

    return pd.DataFrame({"p_init": list_p_0})


def is_outside_obstacles(p, obstacles):
    shift = 0.6  # Make marging around obstacles

    for o in obstacles:

        if (p[0] >= o['min'][0] - shift) and \
                (p[0] <= o['max'][0] + shift) and \
                (p[1] >= o['min'][1] - shift) and \
                (p[1] <= o['max'][1] + shift):
            return False

    return True

#  def is_outside_obstacles(p, obstacles):
#      for o in obstacles:
#
#          if (p[0] >= o['min'][0]) and (p[0] <= o['max'][0]) and \
#                  (p[1] >= o['min'][1]) and (p[1] <= o['max'][1]):
#              return False
#
#      return True


def create_problem(obstacles, T=60, return_variables=False):
    # Define problem data
    d = 2        # Dimension 2D
    n = 2 * d    # Number of states
    M = 50       # Big-M
    deltaT = 0.1  # Time discretization
    p_max = 20 * np.ones(d)   # Maximum position
    p_min = -20 * np.ones(d)   # Minimum position
    v_max = 8 * np.ones(d)   # Maximum position
    v_min = - 8 * np.ones(d)   # Minimum position
    v_0 = np.zeros(d)
    u_max = 10
    p_goal = np.array([-10.5, -10])
    v_goal = np.zeros(d)
    sqrtQp = np.sqrt(1. * np.ones(d))
    sqrtRu = np.sqrt(0.01 * np.ones(d))


    n_obstacles = len(obstacles)

    # Dynamics (2D double integrator)
    Ac = np.block([[np.zeros((d, d)), np.eye(d)],
                   [np.zeros((d, 2*d))]])
    Bc = np.block([[np.zeros((d, d))],
                   [np.eye(d)]])
    sys = ss(Ac, Bc, np.eye(2 * d), np.zeros((2 * d, d)))
    sysd = c2d(sys, deltaT)
    A, B = np.asarray(sysd.A), np.asarray(sysd.B)

    # Parameters
    # ---------
    # Initial and final points
    p_init = cp.Parameter(d, name="p_init")  # Initial state


    # Variables
    # ---------
    p = cp.Variable((T, d))  # Position
    v = cp.Variable((T, d))  # Velocity
    u = cp.Variable((T - 1, d))  # Input in each dimension
    delta_max = [cp.Variable((n_obstacles, d), integer=True) for _ in range(T)]
    delta_min = [cp.Variable((n_obstacles, d), integer=True) for _ in range(T)]

    # Define constraints
    # ------------------
    constraints = []

    # Initial point
    constraints += [p[0] == p_init,
                    v[0] == v_0]

    # Cost
    cost = 0

    # Dynamics
    for t in range(T-1):
        constraints += \
            [cp.hstack([p[t+1], v[t+1]]) == A @ cp.hstack([p[t], v[t]]) + B @ u[t]]

    for t in range(T):
        # Position bounds
        constraints += [p_min <= p[t], p[t] <= p_max]

        # Obstacle constraints
        for i in range(n_obstacles):
            obs_loc = obstacles[i]
            constraints += [p[t] >= obs_loc['max'] - M * delta_max[t][i],
                            p[t] <= obs_loc['min'] + M * delta_min[t][i]]
            constraints += [cp.sum(delta_min[t][i] + delta_max[t][i]) <= 2 * d - 1]
            constraints += [0 <= delta_max[t], delta_max[t] <= 1,
                            0 <= delta_min[t], delta_min[t] <= 1]

        # Velocity bounds
        constraints += [v_min <= v[t], v[t] <= v_max]

        # Cost
        cost += 1/T * cp.sum_squares(cp.multiply(sqrtQp, p[t] - p_goal))

    for t in range(T-1):

        # Control constraints
        constraints += [cp.norm(u[t], np.inf) <= u_max]

        # Cost
        cost += 1/T * cp.sum_squares(cp.multiply(sqrtRu, u[t]))


    # Define problem
    problem = cp.Problem(cp.Minimize(cost), constraints)

    if return_variables:
        variables = {'p': p, 'p_goal': p_goal, 'p_min': p_min,
                     'p_max': p_max, 'v': v, 'u': u}
        return problem, variables
    else:
        return problem
