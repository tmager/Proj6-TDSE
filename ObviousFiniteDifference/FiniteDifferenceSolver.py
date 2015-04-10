"""
ObviousFiniteDifference/FiniteDifferenceSolver.py

Solver class for the simplistic (and not very good) finite-difference scheme for
the time-dependent Schrodinger equation.
"""


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import scipy.sparse.linalg as linalg
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

class FiniteDifferenceSolver:

    def __init__(self, dx, J, dt, N, boundaryConditions, potentialFunction):

        # Formatting stuff for the plots so that they can have nice labels
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=28)
        mpl.rcParams['xtick.major.pad'] = 8
        mpl.rcParams['ytick.major.pad'] = 8
        mpl.rcParams['text.latex.preamble'] = r"\usepackage{xfrac}"

        self.dx = dx
        self.J = J
        self.dt = dt
        self.N = N
        self.V = potentialFunction

        # Find the dimensions of the matrix, which must be of the form (2^k) - 1
        # for some integer k
        self.dim = self.J #2**math.ceil(math.log(self.J+1,2)) - 1

        # Create an empty sparse matrix of complex numbers of the size necessary
        # for our solution; we use a DoK sparse matrix because it is more
        # efficient to build in the manner we use, and we can easily convert it
        # into a CSR matrix later.
        self.mtrx = dok_matrix((self.dim,self.dim),dtype=complex)

        a = -(1j)*self.dt/(self.dx**2)
        # b is defined as a function since it varies with j
        c = -(1j)*self.dt/(self.dx**2)

        # First and last rows of matrix are special cases because they depend on
        # what type of boundary conditions are being used. Thus, we add them
        # separately afterwards.
        for j in range(1,self.J-1):
            self.mtrx[j,j-1] = a
            self.mtrx[j,j]   = self.__b(j)
            self.mtrx[j,j+1] = c

        self.mtrx[0,0] = self.__b(0)
        self.mtrx[0,1] = c
        self.mtrx[self.J - 1,self.J - 2] = a
        self.mtrx[self.J - 1,self.J - 1] = self.__b(J-1)
        if boundaryConditions == "periodic":
            self.mtrx[0,self.J - 1] = a
            self.mtrx[self.J - 1,0] = c

        # Convert our sparse matrix to CSC format to invert it
        self.mtrx = self.mtrx.tocsc()
        self.inverse = linalg.inv(self.mtrx)

        self.initialWavepacket = np.linspace(0,1,self.dim)


    def solve(self):
        """
        Iterate forward from the initial configuration of the system using the
        matrix that was set up in __init__ to step forward in time.
        """
        self.solutionSteps = np.empty([self.N,self.J],dtype=complex)
        self.solutionSteps[0] = self.initialWavepacket
        for n in range(1,self.N):
            self.solutionSteps[n] = self.inverse*self.solutionSteps[n-1]


    def plot(self, step):
        """
        Plot the wavefunction at the given timestep.
        """
        positions = np.arange(0,1,self.dx)

        plt.plot(positions, np.real(self.solutionSteps[step]),    label="Real")
        plt.plot(positions, np.imag(self.solutionSteps[step]),    label="Imag")
        plt.plot(positions, np.absolute(self.solutionSteps[step]), label="Magn")
        plt.show()

    def animPlot(self):
        """
        Create an animated plot of the wavefunction as it progresses forward in
        time.
        """
        pass

    def plot3d(self):
        """
        Plot the wavefunction over all timesteps using a 3D plot.
        """
        fig = plt.figure(figsize=(12,6))
        magn = fig.add_subplot(1,2,1,projection='3d')
        positions = np.arange(0,1,self.dx)
        timesteps = np.arange(0,self.N*self.dt,self.dt)
        X, Y = np.meshgrid(positions, timesteps)
        #Z = self.solutionSteps[X,Y]
        magn.plot_surface(X,Y,np.absolute(self.solutionSteps),color='g',
                          label="|$\psi$|")
        magn.plot_surface(X,Y,np.real(self.solutionSteps),color='b',
                          label="real($\psi$)")
        magn.plot_surface(X,Y,np.imag(self.solutionSteps),color='r',
                          label="imag($\psi$)")
        plt.show()


    # the coefficient on the (n+1),j wavefunction value [unlike the other
    # coefficients, this one varies with position]
    def __b(self, j):
        return ((1j)*self.dt)*((2/(self.dx**2)) + self.V(j*self.dx)
                               - (1j)/self.dt)
