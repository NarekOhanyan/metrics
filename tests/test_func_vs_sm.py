# %%
"""
Test metrics functions against statsmodels
"""
import sys
import unittest
import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm
# import statsmodels.formula.api as smf
import metrics

# Tests

# %%

nT = 80
data = np.random.random((nT, 10))

nC, nI = 1, 1
nL, nLx, nLy, nH = 3, 4, 5, 2

Y, X, Z = data[:, :3].T, data[:, :].T, data[:, 7:].T

nY = Y.shape[0]

df_Ydata = pd.DataFrame(Y.T, index=pd.date_range(start=dt.datetime(2000, 1, 1), periods=nT))
df_Xdata = pd.DataFrame(X.T, index=pd.date_range(start=dt.datetime(2000, 1, 1), periods=nT))

# %%


class test_func_vs_sm(unittest.TestCase):
    """
    Test class
    """

    def test_lpm(self):
        """
        The test
        """

        try:
            LPM = metrics.lpm(data, nL=nL, nH=nH, Y_var_names=["0", "1", "2"])
            # B,U,S = metrics.lpols(X,Y,nL,nH)
            Bx = LPM.model.parameters.Bx

            nX = X.shape[0]
            for iy, y in enumerate(Y):
                for h in range(0, nH + 1):
                    ymat = y[nL-1+h: nT-(nH-h)].T
                    Xmat = np.ones((nT-nL-nH+1, 1))
                    for l in range(0, nL):
                        Xmat = np.column_stack((Xmat, X[:, nL-1-l: nT-nH-l].T))
                    lm = sm.OLS(ymat, exog=Xmat).fit()
                    assert np.allclose(lm.params[1:nX+1], Bx[h, 0, iy], atol=1e-10), '\nTest 1 failed'

        except AssertionError:
            print('Test Failed')
            raise
        else:
            print('Test Passed')


# %%

if __name__ == '__main__':
    unittest.main()

print('')
