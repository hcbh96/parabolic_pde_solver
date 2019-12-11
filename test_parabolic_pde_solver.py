# This file is designed to perform unit and integration tests on all functions
# in parabolic_pde_solver.py


# Test have been written with TDD in mind where possible
import pytest

from parabolic_pde_solver import pde_solve

def test_ensure_returns():
    # Arrange
    # Act
    res = pde_solve()
    # Assert
    assert res.any() != None

