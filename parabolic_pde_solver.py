
#This file has been written with the intention to successfully solve
#parabolic partial differential equations (PPDEs).

# TODO Inputs to the function should include

    # TODO - Diffusion coefficient | kappa > 0
    # TODO - Size of the Domain | L
    # TODO - Desired computation time | T
    # TODO - Initial temperature distribution | func?
    # TODO - Number of discretisation points in space and time
    # pl.legend(loc='upper right')
    # pl.legend(loc='upper right')

    # TODO - fullOutput=True returns additional info default false
# The function should be tested against known solutions wherever possible

import numpy as np
from math import pi
from numpy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags

def diags_m(m, n, dif_ji=[], val=[]):
    """
    This function should return a diagonal matrix of size m x n
    with values specified along the digonals

    USAGE: parabolic_pde_solver.diags_m(m, n, dif_ji, val)

    INPUT:

        m : (Int) specifies that the matrix should have m rows
        n : (Int) specifies that the matrix should have n cols
        dif_ji : (array<Int>) specifies the diagonal index
            away from principal 0
        val : (array<Float>) val[i] specifies the value at diagonal dif_ji[i]

    OUTPUT: tridiagonal matrix of size m x n

    EXAMPLE: tri_diag(10, 10, [-0.5, 2, -0.5], [-1, 0, 1])

        [[ 2.  -0.5  0.   0.   0.   0.   0.   0.   0.   0.   0. ]
         [-0.5  2.  -0.5  0.   0.   0.   0.   0.   0.   0.   0. ]
         [ 0.  -0.5  2.  -0.5  0.   0.   0.   0.   0.   0.   0. ]
         [ 0.   0.  -0.5  2.  -0.5  0.   0.   0.   0.   0.   0. ]
         [ 0.   0.   0.  -0.5  2.  -0.5  0.   0.   0.   0.   0. ]
         [ 0.   0.   0.   0.  -0.5  2.  -0.5  0.   0.   0.   0. ]
         [ 0.   0.   0.   0.   0.  -0.5  2.  -0.5  0.   0.   0. ]
         [ 0.   0.   0.   0.   0.   0.  -0.5  2.  -0.5  0.   0. ]
         [ 0.   0.   0.   0.   0.   0.   0.  -0.5  2.  -0.5  0. ]
         [ 0.   0.   0.   0.   0.   0.   0.   0.  -0.5  2.  -0.5]
         [ 0.   0.   0.   0.   0.   0.   0.   0.   0.  -0.5  2. ]]

    NOTE: could add an iterations counter number
    """
    # create grid in zeros m x n
    grid = np.zeros((m, n))
    # loop over grid for
    for i in range(0, m):
        for j in range(0, n):
            for k in range(len(dif_ji)):
                d = dif_ji[k]
                if j - i == d:
                    v = val[k] # specified here to reduce iterations
                    grid[i][j] = v
    return grid


# internal function so no description string
def plot_x_t_u(x, t, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # create meshgrid
    [X, Y] = np.meshgrid(x, t)
    #plot surface XYZ
    Z = np.asmatrix(Z)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    # set axis labels
    ax.set_title('Evolution of Equation\n')
    ax.set_xlabel('Position ($X$)')
    ax.set_ylabel('Time ($T$)')
    ax.set_zlabel('Temp $u$')
    # create colourbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# internal function so no description added
def create_A_CN(x, f_kappa, deltat, deltax):
    diagonals = [[],[],[]]
    for i in x:
        kappa_v = f_kappa(i)
        # calculate lmbda
        lmbda = kappa_v*deltat/(deltax**2) # mesh fourier number
        # calc -1 0 1 and add to diagonals
        diagonals[0].append(-lmbda/2)
        diagonals[1].append(1+lmbda)
        diagonals[2].append(-lmbda/2)
    # remove 1 value from 1st and third array
    diagonals[0].pop()
    diagonals[2].pop()
    # create matrix 1 and matrix 2
    ACN = diags(diagonals, [-1, 0, 1]).toarray()
    return ACN

#internal function so no description added
def create_B_CN(x, f_kappa, deltat, deltax):
    diagonals = [[],[],[]]
    for i in x:
        kappa_v = f_kappa(i)
        # calculate lmbda
        lmbda = kappa_v*deltat/(deltax**2) # mesh fourier number
        # calc -1 0 1 and add to diagonals
        diagonals[0].append(lmbda/2)
        diagonals[1].append(1-lmbda)
        diagonals[2].append(lmbda/2)
    # remove 1 value from 1st and third array
    diagonals[0].pop()
    diagonals[2].pop()
    # create matrix 1 and matrix 2
    BCN = diags(diagonals, [-1, 0, 1]).toarray()
    return BCN

#set up the function with args
def pde_solve(L, T, u_I, mx, mt, f_kappa= lambda x: 1,
        logger=True, bcf=lambda t: [0,0], plot=None):
    """
    This function should return the solution to a parabolic
    partial differential equation

    USAGE: parabolic_pde_solver.pde_solve(kappa, L, T, u_I, mx, mt)

    INPUT:

        L : (float) size of the domain
        T : (float) desired computation time
        u_I : (func) function that describes Initial temperature distribution
        mx: (array) discretisation points in space
        mt: (array) discretiation points in time

        **optional** (default)
        f_kappa: (lambda x: 1) specifies a variable diffusion coefficient over x
        logger: (True) specifies whether to log outputs to console
        bcf: (() => [0,0]) function that specifies the boundary conditions
        given input t i.e bcs(t) the function must return an array
        of length 2
        plot: (False)


    OUTPUT: (array) [
       U_j: (array) solution to the parabolic PDE
       x : mesh points in space
       t : mesh points in time
    ]

    NOTE:
    """
    # TODO where can one put forcing conditions?
    # set up the numerical environment variables
    x = np.linspace(0, L, mx+1)      # mesh points in space # TODO 2d bar
    t = np.linspace(0, T, mt+1)      # mesh points in time
    deltax = x[1] - x[0]             # gridspacing in x
    deltat = t[1] - t[0]             # gridspacing in t
    # create matrix 1 and matrix 2
    A_CN=create_A_CN(x, f_kappa, deltat, deltax)
    # Prepare ACN & BCN matrix
    B_CN=create_B_CN(x, f_kappa, deltat, deltax)
    #print diagonal matrices
    if logger:
         print("ACN to solve for each step: \n{}".format(A_CN))
         print("BCN to solve for each step: \n{}".format(B_CN))
    # print values we are using
    if logger:
        print("deltax=",deltax); print("deltat=",deltat);
    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step
    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])
    Z = []
    # Solve the PDE: loop over all time points
    for n in range(0, mt+1):
        # dependent var to solve
        b = np.dot(B_CN, u_j) # TODO tridiagonal matrix algoritm or sparse matrix operations?
        # Backward Euler timestep solve matrix equation
        u_jp1 = solve(A_CN, b)
        # Boundary conditions
        [bc1, bc2] = bcf(n)
        u_jp1[0] = bc1; u_jp1[mx] = bc2 #TODO dirchilet and neumann or mixed b cond will affect this part
        # Update u_j
        u_j = u_jp1
        # save u_j values for each time T
        if plot:
            Z.append(u_j)
    # show plot
    if plot:
        plot_x_t_u(x, t, Z)
    return [ u_j, x, t ]


if __name__ == "__main__":
    # solve the heat equation with homogeneous diricelet boundary conditions
    # set problem parameters/functions
    L=1        # length of spatial domain
    T=0.1        # total time to solve for

    # set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 100   # number of gridpoints in time

    # define initial params
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi*x/L)
        return y

    # solve the heat equation
    [u_j, x, t] = pde_solve(L, T, u_I, mx, mt)
    print("Final Solution:\n{}".format(u_j))

    # define this to compare witht the exact solution
    u_exact = lambda x, t: np.exp(-1*(pi**2/L**2)*t)*np.sin(pi*x/L)

    # plot the final result and exact solution
    plt.plot(x, u_j,'ro',label='num')
    xx = np.linspace(0,L,250)
    plt.plot(xx,u_exact(xx,T),'b-',label='exact')
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.legend(loc='upper right')
    plt.show()

    # do the same with varying diricelet boundary conditions
    def bcf(t):
        return [t, t]

    [u_j, x, t] = pde_solve(L, T, u_I, mx, mt, bcf=bcf, plot=True)

    print('Final Solution with varying Dirichlet boundary conditions:\n{}'.format(u_j))

    [u_j, x, t] = pde_solve(L, T, u_I, mx, mt, bcf=bcf, plot=True, f_kappa=lambda x: x)

    print('Final Solution with varying Dirichlet boundary conditions:\n{}'.           format(u_j))
