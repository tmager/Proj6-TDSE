import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import scipy.sparse.linalg as linalg
import math
import matplotlib.pyplot as plt

class FiniteDifferenceSolver:

    def __init__(self, dx, J, dt, N, boundaryConditions, potentialFunction):
        self.dx = dx
        # The problem is posed such that J is the last position, but from a
        # programming standpoint that is a mess; our internal representation
        # will say that J-1 is the last position, and just says that J our J
        # should be 1 larger to take that into account.
        self.J = J + 1
        self.dt = dt
        # See note about J
        self.N = N + 1
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

        #self.initialWavepacket


    def solve(self):
        self.solutionSteps = [self.initialWavepacket]
        for n in range(1,N):
            solutionSteps[n] = self.inverse*solutionSteps[n-1]

    def plot(self, step):
        positions = arange(0,self.J,self.dx)

        plt.plot(positions, np.real(self.solutionSteps[step]),    label="Real")
        plt.plot(positions, np.imag(self.solutionSteps[step]),    label="Imag")
        plt.plot(positions, np.absolute(self.solutionStep[step]), label="Magn")


    # the coefficient on the (n+1),j wavefunction value [unlike the other
    # coefficients, this one varies with position]
    def __b(self, j):
        return ((1j)*self.dt)*((2/(self.dx**2)) + self.V(j*self.dx)
                               - (1j)/self.dt)
