# This file is designed to perform unit and integration tests on all functions
# in parabolic_pde_solver.py


# Test have been written with TDD in mind where possible
import pytest
import numpy as np
from math import pi, isclose
from unittest.mock import MagicMock
from parabolic_pde_solver import pde_solve, diags_m

base_kappa = 1   # diffusion coefficient
base_mx = 10     # number of gridpoints in space
base_mt = 1000   # number of gridpoints in time
base_L = 11           # length of spatial domain
base_T = 1000         # total time to solve for
base_u_I = lambda x: x # initial temperature distribution

def test_unit_ensure_returns():
    # Arrange
    # Act
    res = pde_solve(base_kappa, base_L, base_T, base_u_I, base_mx, base_mt)
    # Assert
    assert res != None


def test_unit_throws_if_not_passed_specified_params():
    #Arrange
    thrown = False
    #Act
    try:
       res = pde_solve()
    except Exception as e:
       thrown = e
    #assert
    assert thrown != False

def test_integration_ensures_that_u_I_is_called_multiple_times():
    # Arrange
    mock = MagicMock()
    # Act
    pde_solve(base_kappa, base_L, base_T, mock, base_mx, base_mt)
    # Assert
    assert mock.call_count == 11

def test_unit_ensures_that_u_I_is_called_with_correct_args():
    # Arrange
    def fake_u_I(x):
        assert isinstance(x, float) == True
    # Act
    pde_solve(base_kappa, base_L, base_T, fake_u_I, base_mx, base_mt)
    # Assert



def test_E2E_agaist_exact_solution_to_heat_equation():
    # Arrange
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

    # define this to compare witht the exact solution
    def u_exact(x,t):
        # the exact solution
        y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
        return y
    # Act
    # solve the heat equation
    [u_j, x, t] = pde_solve(kappa, L, T, u_I, mx, mt)

    # Assert
    looped = False
    # compare sol vs u_exact with error threashold
    for i in range(len(u_j)):
        exact = u_exact(x[i], T)
        assert isclose(u_j[i], exact, abs_tol=1e-6)
        looped = True

    assert looped == True


def test_unit_tri_diag_returns_a_grid_of_the_correct_size():
    # Arrange
    m = 2
    n = 1
    # Act
    M = diags_m(m, n)
    # Assert
    assert M.shape == (2,1)


def test_unit_expected_output_1():
    # Arrange
    # Act
    M  = diags_m(2, 2, [-1, 0, 1], [1, 2, 3])
    # Assert
    np.testing.assert_array_equal(M, [[2, 3], [1, 2]])

def test_unit_expected_output_2():
    # Arrange
    # Act
    M  = diags_m(2, 2, [-1, 0, 1], [5, 3, 4])
    # Assert
    np.testing.assert_array_equal(M, [[3, 4], [5, 3]])
