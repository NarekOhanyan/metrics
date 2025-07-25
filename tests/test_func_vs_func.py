# %%
"""
Test different functions against each other
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

df_Ydata = pd.DataFrame(Y.T, index=pd.date_range(start=dt.datetime(2000, 1, 1), periods=nT))
df_Xdata = pd.DataFrame(X.T, index=pd.date_range(start=dt.datetime(2000, 1, 1), periods=nT))

# %%


class test_fit_var_h_vs_varols(unittest.TestCase):
    """
    Test class
    """

    def test_fit_var_h(self):
        """
        The test
        """

        try:
            B1, _, _, U1, S1 = m.fit_var_h(Y, nC, nL)
            c2, Bx2, U2, S2 = m.varols(Y, nL=nL)

            c1, Bx1 = m.split_C_B(B1, nC, nL, nY)
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

    def test_VARm(self):
        """
        The test
        """

        try:
            Mdl = m.VARm(df_Ydata).irf(ci='wbs')
            Mdl1 = m.varm(df_Ydata, nL=1)
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

    def test_ARDLm(self):
        """
        The test
        """

        try:
            Mdl = m.ARDLm(df_Ydata, Y_var=0, X_vars=[1, 2], nLy=2, nLx=3, contemporaneous_impact=False)
            Mdl1 = m.ardlm(df_Ydata[0], df_Ydata[[1, 2]], nLy=2, nLx=3)

            # print(Mdl1.Est.Bc, np.squeeze(Mdl.Est['Bc']))
            # print(Mdl1.Est.By, np.squeeze(Mdl.Est['By']))
            assert (abs(Mdl1.Est.Bc - np.squeeze(Mdl.Est['Bc'])) < 1e-10).all(), "\nTest 1 failed"
            assert (abs(Mdl1.Est.By.T - np.squeeze(Mdl.Est['By'])) < 1e-10).all(), "\nTest 2 failed"
            assert (abs(Mdl1.Est.Bx.T - np.squeeze(Mdl.Est['Bx'])) < 1e-10).all(), "\nTest 3 failed"
            assert (abs(Mdl1.Irfs.Irf[0] - Mdl.Irfs.Irf_m[0]) < 1e-10).all(), "\nTest 4 failed"
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
