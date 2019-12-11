
#This file has been written with the intention to successfully solve
#parabolic partial differential equations (PPDEs).

# TODO Inputs to the function should include

    # TODO - Diffusion coefficient | kappa > 0
    # TODO - Size of the Domain | L
    # TODO - Desired computation time | T
    # TODO - Initial temperature distribution | func?
    # TODO - Number of discretisation points in space and time

    # TODO - fullOutput=True returns additional info default false
# The function should be tested against known solutions wherever possible

import numpy as np
from math import pi
from numpy.linalg import solve

#set up the function with args
def pde_solve(kappa, L, T, u_I, mx, mt):
    """
    This function should return the solution to a parabolic
    partial differential equation

    USAGE: parabolic_pde_solver.pde_solve(kappa, L, T, u_I, mx, mt)

    INPUT:

        kappa : (float) diffusion coefficient > 0
        L : (float) size of the domain
        T : (float) desired computation time
        u_I : (func) function that describes Initial temperature distribution
        mx: (array) discretisation points in space
        mt: (array) discretiation points in time

    OUTPUT: (array) [
       U_j: (array) solution to the parabolic PDE
       x : mesh points in space
       t : mesh points in time
    ]

    NOTE:
    """

    #TODO possible function
    # set up the numerical environment variables
    x = np.linspace(0, L, mx+1)      # mesh points in space
    t = np.linspace(0, T, mt+1)      # mesh points in time
    deltax = x[1] - x[0]             # gridspacing in x
    deltat = t[1] - t[0]             # gridspacing in t
    lmbda = kappa*deltat/(deltax**2) # mesh fourier number
    # print values we are using TODO log optionality
    print("deltax=",deltax); print("deltat=",deltat); print("lambda=",lmbda)


    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    #set up A_BE matrix
    A_CN = np.zeros((mx+1, mx+1))
    B_CN = np.zeros((mx+1, mx+1))

    #TODO create diagonals function
    # Prepare ACN matrix
    for i in range(0, mx+1):
        for j in range(0, mx+1):
            if i == j:
                A_CN[i][j] = 1 + lmbda
            elif abs(j-i) == 1:
                A_CN[i][j] = -lmbda/2
    # Prepare BCN matrix
    for i in range(0, mx+1):
        for j in range(0, mx+1):
            if i == j:
                B_CN[i][j] = 1 - lmbda
            elif abs(j-i) == 1:
                B_CN[i][j] = lmbda/2

    #TODO log optionalist
    print("ACN to solve for each step: \n{}".format(A_CN))
    print("BCN to solve for each step: \n{}".format(B_CN))

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    # TODO test this part of the function
    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):
        # dependent var to solve
        b = np.dot(B_CN, u_j)
        # Backward Euler timestep solve matrix equation
        u_jp1 = solve(A_CN, b)

        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0

        # Update u_j
        u_j[:] = u_jp1[:]

    # TODO test response
    return [ u_j, x, t ]


if __name__ == "__main__":
    import pylab as pl #specify here to increase load speed

    # solve the heat equation with homogeneous diricelet boundary conditions
    # set problem parameters/functions
    kappa = 1   # diffusion constant
    L=11         # length of spatial domain
    T=1000        # total time to solve for

    # set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    # define initial params
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi*x/L)
        return y

    # solve the heat equation
    [u_j, x, t] = pde_solve(kappa, L, T, u_I, mx, mt)
    print("Solution:\n{}".format(u_j))

    # define this to compare witht the exact solution
    def u_exact(x,t):
        # the exact solution
        y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
        return y

    # plot the final result and exact solution
    pl.plot(x, u_j,'ro',label='num')
    xx = np.linspace(0,L,250)
    pl.plot(xx,u_exact(xx,T),'b-',label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()
