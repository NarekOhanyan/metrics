"""
Test lpols vs OLS from statsmodels
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
            LPM = metrics.lpm(data,nL=nL,nH=nH,Y_var_names=['0','1','2','3','4'])
            # B,U,S = metrics.lpols(X,Y,nL,nH)
            B = LPM.model.parameters.B

            nX = X.shape[0]
            for iy,y in enumerate(Y):
                for h in range(0,nH+1):
                    ymat = y[nL-1+h:nT-(nH-h)].T
                    Xmat = np.ones((nT-nL-nH+1,1))
                    for l in range(0,nL):
                        Xmat = np.column_stack((Xmat,X[:,nL-1-l:nT-nH-l].T))
                    lm = sm.OLS(ymat,exog=Xmat).fit()
                    assert (abs(lm.params[1:nX+1]-B[h,iy,:])<1e-10).all(), '\nTest 1 failed'
        except:
            raise
        else:
            print('Test Passed')

# %%

if __name__ == '__main__':
    unittest.main()

print('')
