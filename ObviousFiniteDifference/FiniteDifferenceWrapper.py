import numpy as np
import FiniteDifferenceSolver2 as Solver
import matplotlib as mpl

def V(x):
    return 0


dx = .1
J = 1000
dt = .1
N = 1000
boundaryConditions = "periodic"
potentialFunction = V
fds = Solver.FiniteDifferenceSolver(dx,J,dt,N,
                                    boundaryConditions,
                                    potentialFunction)
#fds.solve()
#fds.plot()

