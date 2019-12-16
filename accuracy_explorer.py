# This file has been written to explore the accuracy of finite
#    difference approaches to 1D diffusion equations

# To ensure a fair comparison the function should be calculated at a fixed time T
from parabolic_pde_solver import pde_solve
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def accuracy_explorer(method, mt_a, mx_a, L, T, u_I,
        heat_source=lambda x, t: [0], f_kappa=lambda x: 1):
    """
    This function can be used to test the accuracy of the:
        Forward Euler
        Backward Euler
        Crank-Nicholson

    On the heat equation

    The error measure used for the comparison is the L2 Norm
        link: https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

    The function calculates the Error vs a variety of stepsizes and can visualise
    the result

    USAGE: accuracy_explorer(U_I, mt, mx, T, X)
    """
    L2_norm_vec = []
    delta_T = []
    delta_X = []
    print('\nmethod=',method)
    # Loop over mt array and calculate trucation error
    for mt in mt_a:
        for mx in mx_a:
            print('deltaT=', T/mt)
            print('deltax=', L/mx)
            # Calculate 1st approx
            [u_h, u_x, u_t] = pde_solve(L, T, u_I, mx, int(mt), logger=True, method=method,
                    heat_source=heat_source, f_kappa=f_kappa)
            #calculate 2nd approx
            [u_h2, u_hx, u_ht] = pde_solve(L, T, u_I, mx, int(mt/2), logger=True, method=method,
                    heat_source=heat_source, f_kappa=f_kappa)
            # Calculate E
            L2_norm  = np.linalg.norm(np.subtract(u_h2, u_h))
            print('Abs err=',L2_norm)
            L2_norm_vec.append(L2_norm)
            delta_T.append(T/mt)
            delta_X.append(L/mx)

    return [L2_norm_vec, delta_T, delta_X]

if __name__ == "__main__":
    # Explore accuracy with change of mt  of the separate methods on heat equation
    L=10   # length of spatial domain
    T=3       # total time to solve for
    # set numerical parameters
    mx = [1e3]     # number of gridpoints in space
    mt_a = np.logspace(1, 4, num=10)   # number of gridpoints in time
    # define initial params
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi*x/L)
        return y
    f_kappa= lambda x : 1e-2

    [E_FE, h_FE, delta_X_FE]=accuracy_explorer('FE', mt_a, mx, L, T, u_I, f_kappa=f_kappa)
    [E_BE, h_BE, delta_X_BE]=accuracy_explorer('BE', mt_a, mx, L, T, u_I, f_kappa=f_kappa)
    [E_CN, h_CN, delta_X_CN]=accuracy_explorer('CN', mt_a, mx, L, T, u_I, f_kappa=f_kappa)
    # Plot log of E vs h
    # plot the final result and exact solution
    plt.plot(h_FE, E_FE,'r-.',label='FE')
    plt.plot(h_BE, E_BE, 'g--', label='BE')
    plt.plot(h_CN, E_CN, 'b:', label='CN')
    plt.xlabel('dT')
    plt.ylabel('E (T=0.5)')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Wave Equation Error Truncation Analysis')
    plt.show()

    # Explore accuracy with change of mt of the separate methods on the heat equation
    L=10        # length of spatial domain
    T=2       # total time to solve for
    # set numerical parameters
    mx = [1e3]     # number of gridpoints in space
    mt_a = np.logspace(1, 4, num=10)   # number of gridpoints in time
    # define initial params
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi*x/L)
        return y

    f_kappa=lambda x: 1e-3

    [E_FE, h_FE, delta_X_FE]=accuracy_explorer('FE', mt_a, mx, L, T, u_I, f_kappa=f_kappa)
    [E_BE, h_BE, delta_X_BE]=accuracy_explorer('BE', mt_a, mx, L, T, u_I, f_kappa=f_kappa)
    [E_CN, h_CN, delta_X_CN]=accuracy_explorer('CN', mt_a, mx, L, T, u_I, f_kappa=f_kappa)
    # Plot log of E vs h
    # plot the final result and exact solution
    plt.plot(h_FE, E_FE,'r-.',label='FE')
    plt.plot(h_BE, E_BE, 'g--', label='BE')
    plt.plot(h_CN, E_CN, 'b:', label='CN')
    plt.xlabel('dT')
    plt.ylabel('E (T=0.5)')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Wave Equation Error Truncation Analysis')
    plt.show()

