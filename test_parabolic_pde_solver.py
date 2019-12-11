# This file is designed to perform unit and integration tests on all functions
# in parabolic_pde_solver.py


# Test have been written with TDD in mind where possible
import pytest
import numpy as np
from unittest.mock import MagicMock
from parabolic_pde_solver import pde_solve

base_kappa = 1   # diffusion coefficient
base_mx = 10     # number of gridpoints in space
base_mt = 1000   # number of gridpoints in time
base_L = 11           # length of spatial domain
base_T = 1000         # total time to solve for
base_u_I = lambda x: x # initial temperature distribution

def test_ensure_returns():
    # Arrange
    # Act
    res = pde_solve(base_kappa, base_L, base_T, base_u_I, base_mx, base_mt)
    # Assert
    assert res != None


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

def test_ensures_that_u_I_is_called_multiple_times():
    # Arrange
    mock = MagicMock()
    # Act
    pde_solve(base_kappa, base_L, base_T, mock, base_mx, base_mt)
    # Assert
    assert mock.call_count == 11

def test_ensures_that_u_I_is_called_with_correct_args():
    # Arrange
    def fake_u_I(x):
        assert isinstance(x, float) == True
    # Act
    pde_solve(base_kappa, base_L, base_T, fake_u_I, base_mx, base_mt)
    # Assert
