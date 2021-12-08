# %%
"""
Test fit_var_h vs varols
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

nL, nH = 3, 2

Y = data[:, :5].T
X = data[:, :].T

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
            Bc, Bx, SEc, SEx, B1, Se, V, E, S = metrics.fit_var_h(Y, 1, nL)
            c, B2, U, S = metrics.varols(Y, nL=nL)

            assert (abs(Bx - B2) < 1e-10).all(), "\nTest 1 failed"
        except:
            raise
        else:
            print('Test Passed')


# %%

if __name__ == "__main__":
    unittest.main()

print('')

# %%
