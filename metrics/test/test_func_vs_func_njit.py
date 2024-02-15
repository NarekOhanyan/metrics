# %%
"""
Test lpols vs lpols_njit
"""
import sys
import unittest
import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
sys.path.append('../')
import metrics

# Tests

# %%

nT = 80
a = np.random.random((100, 3))
data = np.random.random((nT, 10))

nC = 1
nL, nH = 3, 2

Y = data[:, :5].T
X = data[:, :].T

nY = Y.shape[0]

# %%

print(f'\nnT = {nT}, nL = {nL}, nH = {nH}\n')

# %%


class test_lpols_vs_lpols_njit(unittest.TestCase):
    """
    Test class
    """

    def test(self):
        """
        The test
        """

        try:
            B1, U1, S1 = metrics.lpols(Xdata=X, Ydata=Y, nL=nL, nH=nL)
            B2, U2, S2 = metrics.lpols_njit(Xdata=X, Ydata=Y, nL=nL, nH=nL)

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
