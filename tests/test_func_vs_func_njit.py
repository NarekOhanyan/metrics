# %%
"""
Test numpy vs numba functions
"""
import sys
import unittest
import numpy as np
import pandas as pd
import datetime as dt
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
sys.path.append('../')
import metrics.metrics as m

# Tests

# %%

nT = 80
data = np.random.random((nT, 10))

nC, nI = 1, 1
nL, nLx, nLy, nH = 3, 4, 5, 2

Y, X, Z = data[:, :3].T, data[:, 3:7].T, data[:, 7:].T

nY = Y.shape[0]

# %%


class test_ols_h_vs_ols_h_njit(unittest.TestCase):
    """
    Test class
    """

    def test_ols_h(self):
        """
        The test
        """

        try:
            b1, se1, V1, e1, S1 = m.ols_h(Y[0], X)
            b2, se2, V2, e2, S2 = m.ols_h_njit(Y[0], X)

            assert (abs(b1 - b2) < 1e-10).all(), "\nTest for beta failed"
            assert (abs(se1 - se2) < 1e-10).all(), "\nTest for standard errors failed"
            assert (abs(V1 - V2) < 1e-10).all(), "\nTest for beta variance matrix failed"
            assert (abs(e1 - e2) < 1e-10).all(), "\nTest for residuals failed"
            assert (abs(S1 - S2) < 1e-10).all(), "\nTest for mean squared residual failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

# %%


class test_ols_b_h_vs_ols_b_h_njit(unittest.TestCase):
    """
    Test class
    """

    def test_ols_b_h(self):
        """
        The test
        """

        try:
            B1, SE1, V1, E1, S1 = m.ols_b_h(Y, X)
            B2, SE2, V2, E2, S2 = m.ols_b_h_njit(Y, X)

            assert (abs(B1 - B2) < 1e-10).all(), "\nTest for beta failed"
            assert (abs(SE1 - SE2) < 1e-10).all(), "\nTest for standard errors failed"
            assert (abs(V1 - V2) < 1e-10).all(), "\nTest for beta variance matrix failed"
            assert (abs(E1 - E2) < 1e-10).all(), "\nTest for residuals failed"
            assert (abs(S1 - S2) < 1e-10).all(), "\nTest for mean squared residual failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

# %%


class test_fit_ardl_h_vs_fit_ardl_h_njit(unittest.TestCase):
    """
    Test class
    """

    def test_fit_ardl_h(self):
        """
        The test
        """

        try:
            B1, SE1, V1, E1, S1 = m.fit_ardl_h(Y[0:1], X, Z, nC, nI, nLy, nLx)
            B2, SE2, V2, E2, S2 = m.fit_ardl_h_njit(Y[0:1], X, Z, nC, nI, nLy, nLx)

            assert (abs(B1 - B2) < 1e-10).all(), "\nTest for beta failed"
            assert (abs(SE1 - SE2) < 1e-10).all(), "\nTest for standard errors failed"
            assert (abs(V1 - V2) < 1e-10).all(), "\nTest for beta variance matrix failed"
            assert (abs(E1 - E2) < 1e-10).all(), "\nTest for residuals failed"
            assert (abs(S1 - S2) < 1e-10).all(), "\nTest for mean squared residual failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')


# %%


class test_fit_var_h_vs_fit_var_h_njit(unittest.TestCase):
    """
    Test class
    """

    def test_fit_var_h(self):
        """
        The test
        """

        try:
            B1, SE1, V1, E1, S1 = m.fit_var_h(Y, nC, nL)
            B2, SE2, V2, E2, S2 = m.fit_var_h_njit(Y, nC, nL)

            assert (abs(B1 - B2) < 1e-10).all(), "\nTest for beta failed"
            assert (abs(SE1 - SE2) < 1e-10).all(), "\nTest for standard errors failed"
            assert (abs(V1 - V2) < 1e-10).all(), "\nTest for beta variance matrix failed"
            assert (abs(E1 - E2) < 1e-10).all(), "\nTest for residuals failed"
            assert (abs(S1 - S2) < 1e-10).all(), "\nTest for mean squared residual failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')


# %%


class test_lpols_vs_lpols_njit(unittest.TestCase):
    """
    Test class
    """

    def test_lpols(self):
        """
        The test
        """

        try:
            B1, U1, S1 = m.lpols(Xdata=X, Ydata=Y, nL=nL, nH=nL)
            B2, U2, S2 = m.lpols_njit(Xdata=X, Ydata=Y, nL=nL, nH=nL)

            assert (abs(B1 - B2) < 1e-10).all(), "\nTest 1 failed"
            assert (abs(U1 - U2) < 1e-10).all(), "\nTest 2 failed"
            assert (abs(S1 - S2) < 1e-10).all(), "\nTest 3 failed"
        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')


# %%

if __name__ == '__main__':
    unittest.main()

print('')
