# %%
"""
Test fit_var_h vs varols
"""
import sys
import unittest
import numpy as np
import pandas as pd
import datetime as dt
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
sys.path.append('../')
import metrics

# Tests

# %%

nT = 100
a = np.random.random((100, 3))
data = np.random.random((nT, 10))

nC = 1
nL, nH = 3, 2

Y = data[:, :5].T
X = data[:, :].T

df_Ydata = pd.DataFrame(Y.T, index=pd.date_range(start=dt.datetime(2000, 1, 1), periods=100))
df_Xdata = pd.DataFrame(X.T, index=pd.date_range(start=dt.datetime(2000, 1, 1), periods=100))

nY = Y.shape[0]

# %%

print(f'\nnT = {nT}, nL = {nL}, nH = {nH}\n')

# %%


class test_fit_var_h_vs_varols(unittest.TestCase):
    """
    Test class
    """

    def test(self):
        """
        The test
        """

        try:
            B1, _, _, U1, S1 = metrics.fit_var_h(Y, nC, nL)
            c2, Bx2, U2, S2 = metrics.varols(Y, nL=nL)

            c1, Bx1 = metrics.split_C_B(B1, nC, nL, nY)
            c1 = np.squeeze(c1)

            assert (abs(Bx2 - Bx1) < 1e-10).all(), "\nTest 1 failed"
            assert (abs(c2 - c1) < 1e-10).all(), "\nTest 2 failed"
            assert (abs(U2 - U1) < 1e-10).all(), "\nTest 3 failed"
            assert (abs(S2 - S1) < 1e-10).all(), "\nTest 4 failed"
        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

class test_VARm_vs_varm(unittest.TestCase):
    """
    Test class
    """

    def test(self):
        """
        The test
        """

        try:
            Mdl = metrics.VARm(df_Ydata).irf(ci='wbs')
            Mdl1 = metrics.varm(df_Ydata, nL=1)
            Mdl1.irf(method='ch', nH=12, ci='wbs')

            assert (abs(Mdl1.model.parameters.c - np.squeeze(Mdl.Est['Bc'])) < 1e-10).all(), "\nTest 1 failed"
            assert (abs(Mdl1.model.parameters.B - Mdl.Est['Bx']) < 1e-10).all(), "\nTest 2 failed"
            assert (abs(Mdl1.model.irfs.ch.ir.mean[:, 0, 1] - Mdl.Irfs.Irf_m[1, 0, :]) < 1e-10).all(), "\nTest 3 failed"
        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')


class test_ARDLm_vs_ardlm(unittest.TestCase):
    """
    Test class
    """

    def test(self):
        """
        The test
        """

        try:
            Mdl = metrics.ARDLm(df_Ydata, Y_var=0, X_vars=[1, 2], nLy=2, nLx=3).irf()
            Mdl1 = metrics.ardlm(df_Ydata[0], df_Ydata[[1, 2]], nLy=2, nLx=3)
            Mdl1.irf(nH=12, nR=100)

            assert (abs(Mdl1.Est.Bc - np.squeeze(Mdl.Est['Bc'])) < 1e-10).all(), "\nTest 1 failed"
            assert (abs(Mdl1.Est.By.T - np.squeeze(Mdl.Est['By'])) < 1e-10).all(), "\nTest 2 failed"
            assert (abs(Mdl1.Est.Bx.T - np.squeeze(Mdl.Est['Bx'])) < 1e-10).all(), "\nTest 3 failed"
            assert (abs(Mdl1.Irfs.Irf[0] - Mdl1.Irfs.Irf[0]) < 1e-10).all(), "\nTest 4 failed"
        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')


# %%

if __name__ == "__main__":
    unittest.main()

print('')

# %%
