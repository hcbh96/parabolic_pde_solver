# This file is designed to perform unit and integration tests on all functions
# in parabolic_pde_solver.py


# Test have been written with TDD in mind where possible
import pytest
import numpy as np
from parabolic_pde_solver import pde_solve

base_kappa = 1   # diffusion coefficient
base_mx = 10     # number of gridpoints in space
base_mt = 1000   # number of gridpoints in time
base_L = 11           # length of spatial domain
base_T = 1000         # total time to solve for
base_U_I = np.ones(5) # initial temperature distribution

def test_ensure_returns():
    # Arrange
    # Act
    res = pde_solve(base_kappa, base_L, base_T, base_U_I, base_mx, base_mt)
    # Assert
    assert res.any() != None


def test_throws_if_not_passed_specified_params():
    #Arrange
    thrown = False
    #Act
    try:
       res = pde_solve()
    except Exception as e:
       thrown = e
    #assert
    assert thrown != False

