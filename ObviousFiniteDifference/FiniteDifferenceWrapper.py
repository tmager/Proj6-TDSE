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
parser.add_argument("gridSpacing",
                    type=float)
parser.add_argument("boundaryConditions",
                     choices=["zero", "periodic"])

args = parser.parse_args()

def freeParticle(x):
    return 0

def infiniteSquareWell(x):
    if (x > 5) or (x < -5):
        return 9
    else:
        return 0


potentialFunction = infiniteSquareWell

fds = Solver.FiniteDifferenceSolver(args.gridSpacing,args.gridPoints,
                                    args.stepInterval,args.timesteps,
                                    args.boundaryConditions,
                                    potentialFunction)
fds.solve()
fds.animPlot()
