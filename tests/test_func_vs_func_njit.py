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
import metrics.metrics as metrics

# Tests

# %%

nT = 80
data = np.random.random((nT, 10))

nC, nI = 1, 1
nL, nLx, nLy, nH = 3, 4, 5, 2

Y, X, Z = data[:, :3].T, data[:, 3:7].T, data[:, 7:].T

nY = Y.shape[0]

# %%


class test_func_vs_func_njit(unittest.TestCase):
    """
    Test class
    """

    def test_ols_h(self):
        """
        The test
        """

        try:
            b1, se1, V1, e1, S1 = metrics.ols_h(Y[0], X)
            b2, se2, V2, e2, S2 = metrics.ols_h_njit(Y[0], X)

            assert np.allclose(b1, b2, atol=1e-10), "\nTest for beta failed"
            assert np.allclose(se1, se2, atol=1e-10), "\nTest for standard errors failed"
            assert np.allclose(V1, V2, atol=1e-10), "\nTest for beta variance matrix failed"
            assert np.allclose(e1, e2, atol=1e-10), "\nTest for residuals failed"
            assert np.allclose(S1, S2, atol=1e-10), "\nTest for mean squared residual failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

    def test_ols_b_h(self):
        """
        The test
        """

        try:
            B1, SE1, V1, E1, S1 = metrics.ols_b_h(Y, X)
            B2, SE2, V2, E2, S2 = metrics.ols_b_h_njit(Y, X)

            assert np.allclose(B1, B2, atol=1e-10), "\nTest for beta failed"
            assert np.allclose(SE1, SE2, atol=1e-10), "\nTest for standard errors failed"
            assert np.allclose(V1, V2, atol=1e-10), "\nTest for beta variance matrix failed"
            assert np.allclose(E1, E2, atol=1e-10), "\nTest for residuals failed"
            assert np.allclose(S1, S2, atol=1e-10), "\nTest for mean squared residual failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

    def test_fit_ardl_h(self):
        """
        The test
        """

        try:
            B1, SE1, V1, E1, S1 = metrics.fit_ardl_h(Y[0:1], X, Z, nC, nI, nLy, nLx)
            B2, SE2, V2, E2, S2 = metrics.fit_ardl_h_njit(Y[0:1], X, Z, nC, nI, nLy, nLx)

            assert np.allclose(B1, B2, atol=1e-10), "\nTest for beta failed"
            assert np.allclose(SE1, SE2, atol=1e-10), "\nTest for standard errors failed"
            assert np.allclose(V1, V2, atol=1e-10), "\nTest for beta variance matrix failed"
            assert np.allclose(E1, E2, atol=1e-10), "\nTest for residuals failed"
            assert np.allclose(S1, S2, atol=1e-10), "\nTest for mean squared residual failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

    def test_fit_var_h(self):
        """
        The test
        """

        try:
            B1, SE1, V1, E1, S1 = metrics.fit_var_h(Y, nC, nL)
            B2, SE2, V2, E2, S2 = metrics.fit_var_h_njit(Y, nC, nL)

            assert np.allclose(B1, B2, atol=1e-10), "\nTest for beta failed"
            assert np.allclose(SE1, SE2, atol=1e-10), "\nTest for standard errors failed"
            assert np.allclose(V1, V2, atol=1e-10), "\nTest for beta variance matrix failed"
            assert np.allclose(E1, E2, atol=1e-10), "\nTest for residuals failed"
            assert np.allclose(S1, S2, atol=1e-10), "\nTest for mean squared residual failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

    def test_lpols(self):
        """
        The test
        """

        try:
            Bc1, Bx1, Bz1, U1, S1 = metrics.lpols(Xdata=X, Ydata=Y, nL=nL, nH=nL)
            Bc2, Bx2, Bz2, U2, S2 = metrics.lpols_(Xdata=X, Ydata=Y, nL=nL, nH=nL)

            assert np.allclose(Bc1, Bc2, atol=1e-10), "\nTest 1 failed"
            assert np.allclose(Bx1, Bx2, atol=1e-10), "\nTest 2 failed"
            assert np.allclose(Bz1, Bz2, atol=1e-10), "\nTest 4 failed"
            assert np.allclose(U1, U2, atol=1e-10), "\nTest 5 failed"
            assert np.allclose(S1, S2, atol=1e-10), "\nTest 6 failed"
        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')


# %%

if __name__ == '__main__':
    unittest.main()

print('')
