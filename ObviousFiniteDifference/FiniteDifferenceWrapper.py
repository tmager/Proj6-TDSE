"""
ObviousFiniteDifference/FiniteDifferenceWrapper.py

Command-line wrapper for the FiniteDifferenceSolver file.

usage: python FiniteDifferenceWrapper.py TIMESTEPS STEP_INTERVAL GRID_POINTS
                                                            BOUNDARY_CONDITIONS
where boundary conditions can be either "zero" or "periodic".
"""

import numpy as np
import FiniteDifferenceSolver as Solver
import matplotlib as mpl

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("timesteps",
                    type=int)
parser.add_argument("stepInterval",
                    type=float)
parser.add_argument("gridPoints",
                    type=int)
parser.add_argument("boundaryConditions",
                     choices=["zero", "periodic"])

args = parser.parse_args()

def V(x):
    return 0


dx = .1
J = 1000
dt = .1
N = 1000
boundaryConditions = "periodic"

potentialFunction = V

fds = Solver.FiniteDifferenceSolver(1./args.gridPoints,args.gridPoints,
                                    args.stepInterval,args.timesteps,
                                    args.boundaryConditions,
                                    potentialFunction)
fds.solve()
fds.plot3d()
