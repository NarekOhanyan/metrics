"""
Test lpols vs lpols_njit
"""
import sys
import unittest
import numpy as np
import statsmodels.api as sm
# import statsmodels.formula.api as smf
sys.path.append('./metrics/')
import metrics

# Tests

# %%

nT = 80
a = np.random.random((100,3))
data = np.random.random((nT,10))

nL, nH = 3, 2

Y=data[:,:5].T
X=data[:,:].T

# %%

print(f'\nnT = {nT}, nL = {nL}, nH = {nH}\n')

# %%

class test_lpols_vs_sm_OLS(unittest.TestCase):
    """
    Test class
    """
    def test(self):
        """
        The test
        """

        try:
            B1,U,S = metrics.lpols(Xdata=X,Ydata=Y,nL=nL,nH=nL)
            B2,U,S = metrics.lpols_njit(Xdata=X,Ydata=Y,nL=nL,nH=nL)

            assert (abs(B1-B2)<1e-10).all(), '\nTest 1 failed'
        except:
            raise
        else:
            print('Test Passed')

# %%

if __name__ == '__main__':
    unittest.main()

print('')
