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
import metrics

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


class test_func_vs_func(unittest.TestCase):
    """
    Test class
    """

    def test_fit_var_h(self):
        """
        The test
        """

        try:
            B1, _, _, U1, S1 = metrics.fit_var_h(Y, nC, nL)
            c2, Bx2, U2, S2 = metrics.varols(Y, nL=nL)

            c1, Bx1 = metrics.split_C_B(B1, nC, nL, nY)
            c1 = np.squeeze(c1)

            assert np.allclose(Bx1, Bx2, atol=1e-10), "\nTest 1 failed"
            assert np.allclose(c1, c2, atol=1e-10), "\nTest 2 failed"
            assert np.allclose(U1, U2, atol=1e-10), "\nTest 3 failed"
            assert np.allclose(S1, S2, atol=1e-10), "\nTest 4 failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

    def test_VARm(self):
        """
        The test
        """

        try:
            Mdl1 = metrics.VARm(df_Ydata).irf(ci='wbs')
            Mdl2 = metrics.varm(df_Ydata, nL=1)
            Mdl2.irf(method='ch', nH=12, ci='wbs')

            assert np.allclose(np.squeeze(Mdl1.Est['Bc']), Mdl2.model.parameters.c, atol=1e-10), "\nTest 1 failed"
            assert np.allclose(Mdl1.Est['Bx'], Mdl2.model.parameters.B, atol=1e-10), "\nTest 2 failed"
            assert np.allclose(Mdl1.Irfs.Irf_m[1, 0, :], Mdl2.model.irfs.ch.ir.mean[:, 0, 1], atol=1e-10), "\nTest 3 failed"

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')

    def test_ARDLm(self):
        """
        The test
        """

        try:
            Mdl1 = metrics.ARDLm(df_Ydata, Y_var=0, X_vars=[1, 2], nLy=2, nLx=3, contemporaneous_impact=False)
            Mdl2 = metrics.ardlm(df_Ydata[0], df_Ydata[[1, 2]], nLy=2, nLx=3)

            # print(Mdl2.Est.Bc, np.squeeze(Mdl1.Est['Bc']))
            # print(Mdl2.Est.By, np.squeeze(Mdl1.Est['By']))
            assert np.allclose(np.squeeze(Mdl1.Est['Bc']), Mdl2.Est.Bc, atol=1e-10), "\nTest 1 failed"
            assert np.allclose(np.squeeze(Mdl1.Est['By']), Mdl2.Est.By.T, atol=1e-10), "\nTest 2 failed"
            assert np.allclose(np.squeeze(Mdl1.Est['Bx']), Mdl2.Est.Bx.T, atol=1e-10), "\nTest 3 failed"
            assert np.allclose(Mdl1.Irfs.Irf_m[0], Mdl2.Irfs.Irf[0], atol=1e-10), "\nTest 4 failed"

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
