# This file is designed to perform unit and integration tests on all functions
# in parabolic_pde_solver.py


# Test have been written with TDD in mind where possible
import pytest
import numpy as np
from math import pi, isclose
from unittest.mock import MagicMock
from parabolic_pde_solver import pde_solve, diags_m, create_A_CN
from parabolic_pde_solver import create_B_CN, create_A_BE, create_A_FE
from parabolic_pde_solver import solve_BE, solve_FE, solve_CN
from scipy.sparse import diags, csr_matrix

base_mx = 100     # number of gridpoints in space
base_mt = 1000   # number of gridpoints in time
base_L = 11           # length of spatial domain
base_T = 0.5         # total time to solve for
base_u_I = lambda x: np.sin(pi*x/base_L) # initial temperature distriibution

def test_unit_ensure_returns():
    # Arrange
    # Act
    res = pde_solve(base_L, base_T, base_u_I, base_mx, base_mt)
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
    pde_solve(base_L, base_T, mock, base_mx, base_mt)
    # Assert
    assert mock.call_count == 101

def test_unit_ensures_that_u_I_is_called_with_correct_args():
    # Arrange
    def fake_u_I(x):
        assert isinstance(x, float) == True
    # Act
    pde_solve(base_L, base_T, fake_u_I, base_mx, base_mt)
    # Assert

def test_int_ensures_the_bcf_func_is_called_corrrectly():
    # Arrange
    mock = MagicMock(return_value=[0,0])
    # Act
    pde_solve(base_L, base_T, base_u_I, base_mx, base_mt, bcf=mock)
    # Assert
    assert mock.call_count == base_mt + 1


def test_unit_ensures_bcf_is_called_with_the_correct_params():
    # Arrange
    def fake_bcf(t):
        assert type(t) == int
        return [0,0]
    # Act
    pde_solve(base_L, base_T, base_u_I, base_mx, base_mt, bcf=fake_bcf)
    # Assert

def test_E2E_agaist_exact_solution_to_heat_equation():
    # Arrange
    # set problem parameters/functions
    kappa = 1   # diffusion constant
    L=11         # length of spatial domain
    T=0.5        # total time to solve for

    # set numerical parameters
    mx = 100     # number of gridpoints in space
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
    [u_j, x, t] = pde_solve(L, T, u_I, mx, mt, f_kappa=lambda x: 1)

    # Assert
    looped = False
    # compare sol vs u_exact with error threashold
    for i in range(len(u_j)):
        exact = u_exact(x[i], T)
        assert isclose(u_j[i], exact, abs_tol=1e-3)
        looped = True

    assert looped == True


def test_E2E_agaist_heat_equation_varying_bcf():
    # Arrange
    # set problem parameters/functions
    kappa = 1   # diffusion constant
    L=11         # length of spatial domain
    T=0.5        # total time to solve for

    # set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 10   # number of gridpoints in time

    # define initial params
    def u_I(x):
         # initial temperature distribution
         y = np.sin(pi*x/L)
         return y

    def bcf(t):
        return [t, t]

    # Act
    # solve the heat equation
    [u_j, x, t] = pde_solve(L, T, u_I, mx, mt, bcf=bcf)

    # Assert
    # check solution at final value boundary conditions
    assert u_j[0] == 10
    assert u_j[-1] == 10

def test_E2E_agaist_heat_equation_varying_diffusion_coefficient():
    # Arrange
    # set problem parameters/functions
    L=11         # length of spatial domain
    T=0.5        # total time to solve for

    # set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    # define initial params
    def u_I(x):
         # initial temperature distribution
         y = np.sin(pi*x/L)
         return y

    # Act
    # solve the heat equation
    [u_j, x, t] = pde_solve(L, T, u_I, mx, mt, f_kappa=lambda x: x)
    print("u_j {}".format(u_j))
    assert np.isclose(u_j, [0, 0.26789984, 0.40498984, 0.49342122, 0.52927611, 0.5140508,
 0.45421537, 0.36018634, 0.24470462, 0.12086154, 0]).all()


def test_E2E_agaist_heat_equation_varying_diffusion_coefficient():
    # Arrange
    # set problem parameters/functions
    L=11         # length of spatial domain
    T=0.5        # total time to solve for

    # set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    # define initial params
    def u_I(x):
         # initial temperature distribution
         y = np.sin(pi*x/L)
         return y

    def heat_source(x, t):
        res = np.piecewise(x, [x < 5, x >= 5], [-1, 1])
        return res

    # Act
    # solve the heat equation
    [u_j, x, t] = pde_solve(L, T, u_I, mx, mt, heat_source=heat_source)
    print("u_j {}".format(u_j))
    assert np.isclose(u_j, [0, -838.73622915, -977.55407457, -962.87832712, -711.98669888
, 714.00385471, 966.27180176, 996.50604943, 981.83687393, 839.5809771, 0]).all()


""" The below section contains test for diags_m"""

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

"""The above section contains tests for diags_m"""

"""The below contains tests for create_ACN"""
def test_E2E_createACN_outputs_the_correct_matrix1():
    # Arrange
    x=np.ones(10)
    f_kappa=lambda x: 1
    deltax=1
    deltat=1
    # Act
    ACN = create_A_CN(x, f_kappa, deltat, deltax, logger=False)
    # Assert
    looped = False
    dense = ACN.todense()
    [x, y] = dense.shape
    for i in range(x):
        for j in range(y):
            if i==j:
                assert dense.item((i,j)) == 2
                looped = True
            elif abs(i-j) == 1:
                assert dense.item((i,j)) == -0.5
            else:
                assert dense.item((i,j)) == 0

    assert looped == True


def test_E2E_createACN_outputs_the_correct_matrix2():
    # Arrange
    x=np.ones(10)
    f_kappa=lambda x: 2*x
    deltax=1
    deltat=1
    # Act
    ACN = create_A_CN(x, f_kappa, deltat, deltax, logger=False)
    # Assert
    looped = False
    dense = ACN.todense()
    [x, y] = dense.shape
    for i in range(x):
        for j in range(y):
            if i==j:
                assert dense.item((i,j)) == 3
                looped = True
            elif abs(i-j) == 1:
                assert dense.item((i,j)) == -1
            else:
                assert dense.item((i,j)) == 0
    assert looped == True
"""The above contains test for create_ACN"""

"""The below contains tests for create_BCN"""
def test_E2E_create_BCN_outputs_the_correct_matrix1():
    # Arrange
    x=np.ones(10)
    f_kappa=lambda x: 1
    deltax=1
    deltat=1
    # Act
    BCN = create_B_CN(x, f_kappa, deltat, deltax, logger=False)
    # Assert
    looped = False
    dense = BCN.todense()
    [x, y] = dense.shape
    for i in range(x):
        for j in range(y):
            if i==j:
                assert dense.item((i,j)) == 0
                looped = True
            elif abs(i-j) == 1:
                assert dense.item((i,j)) == 0.5
            else:
                assert dense.item((i,j)) == 0
    assert looped == True

def test_E2E_createBCN_outputs_the_correct_matrix2():
    # Arrange
    x=np.ones(10)
    f_kappa=lambda x: 2*x
    deltax=1
    deltat=1
    # Act
    BCN = create_B_CN(x, f_kappa, deltat, deltax, logger=False)
    # Assert
    looped = False
    dense = BCN.todense()
    [x, y] = dense.shape
    for i in range(x):
        for j in range(y):
            if i==j:
                assert dense.item((i,j)) == -1
                looped = True
            elif abs(i-j) == 1:
                assert dense.item((i,j)) == 1
            else:
                assert dense.item((i,j)) == 0
    assert looped == True

"""The above contains tests for create_BCN"""

"""The below contains test for create_ABE"""
def test_E2E_create_ABE_outputs_the_correct_matrix1():
    # Arrange
    x=np.ones(10)
    f_kappa=lambda x: 1
    deltax=1
    deltat=1
    # Act
    ABE = create_A_BE(x, f_kappa, deltat, deltax, logger=False)
    # Assert
    looped = False
    dense = ABE.todense()
    [x, y] = dense.shape
    for i in range(x):
        for j in range(y):
            if i==j:
                assert dense.item((i,j)) == 3
                looped = True
            elif abs(i-j) == 1:
                assert dense.item((i,j)) == -1
            else:
                assert dense.item((i,j)) == 0
    assert looped == True
"""The above contains tests for create_ABE"""

"""The below contains tests for create AFE"""
def test_E2E_create_AFE_outputs_the_correct_matrix1():
    # Arrange
    x=np.ones(10)
    f_kappa=lambda x: 1
    deltax=1
    deltat=1
    # Act
    [AFE, lmbda_v] = create_A_FE(x, f_kappa, deltat, deltax, logger=False)
    # Assert
    looped = False
    dense = AFE.todense()
    [x, y] = dense.shape
    for i in range(x):
        assert lmbda_v[i] == 1
        for j in range(y):
            if i==j:
                assert dense.item((i,j)) == -1
                looped = True
            elif abs(i-j) == 1:
                assert dense.item((i,j)) == 1
            else:
                assert dense.item((i,j)) == 0
    assert looped == True
"""The above contains tests for create_AFE"""

"""The below contains tests for solve_CN"""
def test_E2E_solve_CN_outputs_the_correct_result():
    # Arrange
    u_j=[2, 3, 4]
    A_CN=csr_matrix([[1, -2, 0],[-2, 1, -2],[0, -2, 1]])
    B_CN=csr_matrix([[2,3,6],[5,6,9],[1,2,2]])
    heat_j=[7,1,5]
    # Act
    u_jp1 = solve_CN(u_j, A_CN, B_CN, heat_j)
    # Assert
    assert np.allclose(u_jp1, [ -4.57142857, -23.28571429, -27.57142857])

"""The above contains tests for solve_CN"""

"""The above contains tests for solve_FE"""
def test_E2E_solve_FE_outputs_the_correct_result():
    # Arrange
    u_j=[0,0,0]
    A_BE=csr_matrix([[1, -2, 0],[-2, 1, -2],[0, -2, 1]])
    heat_j=csr_matrix([0,1,0])
    lmbda=8
    bc1=6
    bc2=5
    # Act
    u_jp1 = solve_FE(u_j, A_BE, heat_j, lmbda, bc1,bc2)
    # Assert
    assert np.allclose(u_jp1, [48,  1, 40])

"""The above contains tests for solve_FE"""

"""The below contains tests for solve_BE"""
def test_E2E_solve_BE_outputs_the_correct_result():
    # Arrange
    u_j=[5,8,9]
    A_BE=csr_matrix([[1, -2, 0],[-2, 1, -2],[0, -2, 1]])
    heat_j=[0,1,0]
    # Act
    u_jp1 = solve_BE(u_j, A_BE, heat_j)
    # Assert
    assert np.allclose(u_jp1, [-5.28571429, -4.14285714, -1.28571429])
"""The above contains tests for solve_BE"""
