# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi
from scipy.optimize import newton
from numpy.linalg import solve

# set problem parameters/functions
kappa = 1   # diffusion constant
L=11         # length of spatial domain
T=1000        # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000   # number of gridpoints in time


# set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)

# set up the solution variables
u_j = np.zeros(x.size)        # u at current time step
u_jp1 = np.zeros(x.size)      # u at next time step

#set up A_BE matrix
A_CN = np.zeros((mx+1, mx+1))
B_CN = np.zeros((mx+1, mx+1))

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

print("ACN to solve for each step: \n{}".format(A_CN))
print("BCN to solve for each step: \n{}".format(B_CN))

# Set initial condition
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])

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

# plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()
