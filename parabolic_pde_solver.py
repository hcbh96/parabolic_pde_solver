
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

#set up the function with args
def pde_solve():
    u_j = np.zeros(5)
    return u_j


if __name__ == "__main__":
    # solve the heat equation with homogeneous diricelet boundary conditions
    print("Solution:\n{}".format(pde_solve()))
