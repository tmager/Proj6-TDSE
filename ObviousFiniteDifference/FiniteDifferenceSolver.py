"""
ObviousFiniteDifference/FiniteDifferenceSolver.py

Solver class for the simplistic (and not very good) finite-difference scheme for
the time-dependent Schrodinger equation.
"""


import numpy as np
import scipy as sp
import scipy.integrate as integrate
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import scipy.sparse.linalg as linalg
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as anim

class FiniteDifferenceSolver:

    def __init__(self, dx, J, dt, N, boundaryConditions, potentialFunction):

        # Formatting stuff for the plots so that they can have nice labels
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=20)
        mpl.rcParams['xtick.major.pad'] = 6
        mpl.rcParams['ytick.major.pad'] = 6
        mpl.rcParams['text.latex.preamble'] = r"\usepackage{xfrac}\usepackage{amsbsy}"

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

        self.initialWavepacket = self.__discretizedWavefunction()


    def solve(self):
        """
        Iterate forward from the initial configuration of the system using the
        matrix that was set up in __init__ to step forward in time.
        """
        self.solutionSteps = np.zeros([self.N,self.J],dtype=complex)
        self.solutionSteps[0] = self.initialWavepacket
        for n in range(1,self.N):
            self.solutionSteps[n] = self.inverse*self.solutionSteps[n-1]


    def plot(self, step):
        """
        Plot the wavefunction at the given timestep.
        """
        positions = np.arange(-self.dx*self.J/2,self.dx*self.J/2,self.dx)
        print(len(positions))
        print(len(self.solutionSteps[step]))
        fig = plt.figure()
        graph = fig.add_subplot(1,1,1)
        graph.plot(positions, np.absolute(self.solutionSteps[step]),
                   label="$\\boldsymbol{|\\Psi|}$")
        graph.plot(positions, np.real(self.solutionSteps[step]),
                   label="Real($\\boldsymbol\\Psi$)")
        graph.plot(positions, np.imag(self.solutionSteps[step]),
                   label="Imag($\\boldsymbol\\Psi$)")
        graph.set_xlabel("Position")
        graph.set_ylabel("Wavefunction Value")
        graph.legend()
        graph.grid(b=True, which='both', color='0.65',linestyle='-')
        plt.show()

    def animPlot(self):
        """
        Create an animated plot of the wavefunction as it progresses forward in
        time.
        """
        positions = np.arange(-self.dx*self.J/2,self.dx*self.J/2,self.dx)
        
        fig = plt.figure()
        graph = fig.add_subplot(1,1,1)
        graph.set_xlim([-self.dx*self.J/2 - 2*self.dx,
                        self.dx*self.J/2 + 2*self.dx])
        graph.set_ylim([-1,1])
        
        magnGraph, = graph.plot(positions,np.absolute(self.solutionSteps[0]))
        realGraph, = graph.plot(positions,np.real(self.solutionSteps[0]))
        imagGraph, = graph.plot(positions,np.imag(self.solutionSteps[0]))

        def init():
            magnGraph.set_data([],[])
            realGraph.set_data([],[])
            imagGraph.set_data([],[])
            return magnGraph,realGraph,imagGraph,

        def animate(i):
            magnGraph.set_data(positions,np.absolute(self.solutionSteps[i]))
            realGraph.set_data(positions,np.real(self.solutionSteps[i]))
            imagGraph.set_data(positions,np.imag(self.solutionSteps[i]))
            return magnGraph,realGraph,imagGraph,

        ani = anim.FuncAnimation(fig, animate, np.arange(0, self.N),
                                 init_func=init,interval=25, blit=True)
        #ani.save('animExportTest.mp4', fps=30)
        plt.show()


    def plot3d(self):
        """
        Plot the wavefunction over all timesteps using a 3D plot.
        """
        fig = plt.figure()
        graph = fig.add_subplot(1,1,1,projection='3d')
        positions = np.arange(-self.dx*self.J/2,self.dx*self.J/2,self.dx)
        timesteps = np.arange(0,self.N*self.dt,self.dt)
        X, Y = np.meshgrid(positions, timesteps)
        print(len(X))
        print(len(Y))
        print(len(self.solutionSteps),' ',len(self.solutionSteps[1]))
        #Z = self.solutionSteps[X,Y]
        graph.plot_surface(X,Y,np.absolute(self.solutionSteps),color='g',
                          label="$\\boldsymbol{|\\Psi|}$")
        graph.plot_surface(X,Y,np.real(self.solutionSteps),color='b',
                          label="Imag($\\boldsymbol\\Psi$)")
        graph.plot_surface(X,Y,np.imag(self.solutionSteps),color='r',
                          label="Magn($\\boldsymbol\\Psi$)")
        graph.set_xlabel("Position")
        graph.set_ylabel("Time")
        graph.set_zlabel("Wavefunction Value")
        # graph.legend()
        ## Legends/labels don't work right for plot_surface; needs a workaround
        plt.show()


    # the coefficient on the (n+1),j wavefunction value [unlike the other
    # coefficients, this one varies with position]
    def __b(self, j):
        return ((1j)*self.dt)*((2/(self.dx**2))
                               + self.V(j*self.dx - self.dx*self.J/2)
                               - (1j)/self.dt)

    def __wavefunction(self, x, exponent):
        exponentFxn = lambda x: math.exp(-exponent*x**2)**2
        normCoef = integrate.quad(exponentFxn,-50,50)[0]
        return exponentFxn(x)/normCoef

    def __discretizedWavefunction(self):
        vector = np.empty(self.J)
        for i in range(self.J):
            vector[i] = self.__wavefunction(i*self.dx - self.dx*self.J/2, 5)
        return vector
