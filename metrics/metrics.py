# %% Load packages
import numpy as np
import numba as nb
import pandas as pd
import datetime as dt
import scipy.stats as spstats
import matplotlib.pyplot as mpl

use_numba = True

# %%
class block:
    pass

# %%
class data:
    pass

# %%
class est:
    pass

# %%
class spec:
    pass

# %%
class model:
    def __init__(self):
        self.Spec = spec()
        self.Data = data()
        self.Est = est()

# %%
class irfs:

    def __init__(self, Irf_m, Irfc_m, Irf_sim, Irfc_sim, /, *, irf_spec=None):
        self.Irf_m, self.Irfc_m = Irf_m, Irfc_m
        self.__Irf_sim, self.__Irfc_sim = Irf_sim, Irfc_sim
        self.Spec = irf_spec

    def Irf_q(self, q):
        return np.quantile(self.__Irf_sim, q, axis=0)

    def Irfc_q(self, q):
        return np.quantile(self.__Irfc_sim, q, axis=0)

# %%
class forecasts:

    def __init__(self, Fcs_m, Fcsc_m, Fcs_sim, Fcsc_sim, /, *, forecast_spec=None):
        self.Fcs_m, self.Fcsc_m = Fcs_m, Fcsc_m
        self.__Fcs_sim, self.__Fcsc_sim = Fcs_sim, Fcsc_sim
        self.Spec = forecast_spec

    def Fcs_q(self, q):
        return np.quantile(self.__Fcs_sim, q, axis=0)

    def Fcsc_q(self, q):
        return np.quantile(self.__Fcsc_sim, q, axis=0)

# %%
@nb.njit
def C(arr):
    return np.ascontiguousarray(arr)

# %%
def check_data(Ydata, Xdata, add_constant):
    """
    Check data for NaNs and add a constant if requested

    Args:
        Ydata (nY, n1): Ydata in row format.
        Xdata (nX, n1): Xdata in row format.
        add_constant (bool, optional): Add a constant to Xdata. Defaults to True

    Returns:
        (nY, nN): Ydata without NaNs
        (nX, nN): Xdata without NaNs
        (nC+nX, n1): Xdata with constant if requested
    """
    Ydata = Ydata.reshape((-1, Ydata.shape[1]))
    nY, _ = Ydata.shape
    nC = 1 if add_constant else 0
    cXdata = np.vstack((np.ones((nC, Xdata.shape[1])), Xdata))
    YXdata = np.vstack((Ydata, cXdata))
    YXdata = YXdata[:, ~np.isnan(YXdata).any(axis=0)]
    return YXdata[:nY], YXdata[nY:], cXdata

# %%
def get_NaN_free_idx(Ydata):

    n0, _ = Ydata.shape
    any_NaN = np.isnan(Ydata.values).any(axis=1)
    if not any_NaN.any():                               # if no NaNs
        NaN_free_idx = [0, n0-1]
    elif not any_NaN[0] and not any_NaN[-1]:            # if NaNs in the middle
        raise ValueError('sample should not contain NaNs')
    else:
        any_NaN_change = np.argwhere(any_NaN[1:] != any_NaN[:-1])+1
        if any_NaN_change.shape[0] > 2:                             # if more than two NaN blocks
            raise ValueError('sample should not contain NaNs')
        elif any_NaN[0] and not any_NaN[-1]:                        # if sample starts with NaNs
            NaN_free_idx = [any_NaN_change[0, 0], n0-1]
        elif not any_NaN[0] and any_NaN[-1]:                        # if sample ends with NaNs
            NaN_free_idx = [0, any_NaN_change[0, 0]-1]
        else:                                                       # if starts and ends with NaNs
            NaN_free_idx = [any_NaN_change[0, 0], any_NaN_change[1, 0]-1]

    return NaN_free_idx

# %% ols in row format
def ols_h(y, X, dfc=True):
    """
    OLS with data in row format

    Args:
        y (nN,): y matrix.
        X (nX, nN): X matrix.
        dfc (bool, optional): Degrees of freedom correction. Defaults to True.

    Returns:
        (nX,): parameter estimates
        (nX,): standard errors of parameter estimates
        (nX, nX): variance matrix of parameter estimates
        (nN,): residuals
        (1,): mean squared error
    """
    y, X, Xt = C(y), C(X), C(X.T)
    nX, nN = X.shape
    b = (y@Xt)@np.linalg.inv(X@Xt)
    e = y-b@X
    df = nN-nX if dfc else nN
    S = (1/df)*(e@e.T)
    V = S*np.linalg.inv(X@Xt)
    se = np.sqrt(np.diag(V)).reshape(b.shape)
    return b, se, V, e, S

ols_h_njit = nb.njit(ols_h)

# %% OLS with many dependent variables y
def ols_b_h(Y, X, dfc=True):
    Y, X = C(Y), C(X)
    nY, nN = Y.shape
    nX, nN = X.shape
    Yy = Y.reshape((1, nY*nN))
    Xx = np.kron(np.eye(nY), X)
    if not use_numba:
        b, _, _, e, _ = ols_h(Yy, Xx, dfc)
    else:
        b, _, _, e, _ = ols_h_njit(Yy, Xx, dfc)
    B = b.reshape((nY, nX))
    E = e.reshape((nY, nN))
    df = nN-nX if dfc else nN
    S = (1/df)*(E@E.T)
    invX = C(np.linalg.inv(X@X.T))
    if not use_numba:
        invXx = np.tile(invX, (nY, 1, 1))
    else:
        # invXx = np.repeat(invX[np.newaxis, :, :], nY, axis=0)
        invXx = np.repeat(invX, nY).reshape((nX*nX, nY)).T.reshape((nY, nX, nX))
    V = np.diag(S).reshape((-1, 1, 1))*invXx
    # SE = np.array([np.sqrt(np.diag(V[iY])) for iY in range(nY)])
    SE = np.sqrt(np.outer(np.diag(S), np.diag(invX)))
    return B, SE, V, E, S

ols_b_h_njit = nb.njit(ols_b_h) # doesn't work due to https://github.com/numba/numba/issues/4580

# %%
def split_C_B(B, nC, nL, nY):
    Bc, Bx = np.split(B, np.array([nC]), axis=1)
    Bx = C(Bx).T.reshape((nL, nY, nY)).transpose((0, 2, 1))
    return Bc, Bx

split_C_B_njit = nb.njit(split_C_B)

# %% ARDL-OLS
def fit_ardl_h(y, X, Z, nC, nI, nLy, nLx, dfc=True):
    """
    Function to estimate ARDL(p, q) model with p = nLy, q = nLx using OLS
    """
    y, X, Z = C(y), C(X), C(Z)
    _, n1 = y.shape
    nL = max(nLy, nLx)

    Y = y.reshape(1, -1)

    W = np.ones((nC, n1))
    W = np.vstack((W, Z))
    for l in range(1, nLy+1):
        W = np.vstack((W, np.roll(y, l)))    # np.roll without axis argument performs the operation on a flattened array, but the result is unaffected due to the drop of first nL observations
    for l in range(1-nI, nLx+1):
        W = np.vstack((W, np.roll(X, l)))

    Y, W = Y[:, nL:], W[:, nL:]

    if not use_numba:
        B, SE, V, U, S = ols_h(Y, W, dfc)
    else:
        B, SE, V, U, S = ols_h_njit(Y, W, dfc)

    return B, SE, V, U, S

fit_ardl_h_njit = nb.njit(fit_ardl_h)

# %% VAR-OLS
def fit_var_h(Y, nC, nL, dfc=True):
    """
    Function to estimate VAR(P) model with P = nL using OLS
    """
    _, n1 = Y.shape
    X = np.ones((nC, n1))
    for p in range(1, nL+1):
        X = np.vstack((X, np.roll(Y, p)))    # np.roll without axis argument performs the operation on a flattened array, but the result is unaffected due to the drop of first nL observations
    Y, X = Y[:, nL:], X[:, nL:]
    if not use_numba:
        B, SE, V, U, S = ols_b_h(Y, X, dfc)
    else:
        B, SE, V, U, S = ols_b_h_njit(Y, X, dfc)

    return B, SE, V, U, S

fit_var_h_njit = nb.njit(fit_var_h)

# %% OLS with one dependent variable y
def OLS_h(Ydata, Xdata, add_constant=True, dfc=True):
    Y, X, cXdata = check_data(Ydata, Xdata, add_constant)
    b, se, V, _, S = ols_h(Y, X, dfc)
    e = Ydata-b@cXdata
    return b, se, V, e, S

# %% OLS with many dependent variables y
def OLS_b_h(Ydata, Xdata, add_constant=True, dfc=True):
    assert ~np.isnan(Ydata).any() and ~np.isnan(Xdata).any(), 'data should not contain NaNs'
    Y, X, _ = check_data(Ydata, Xdata, add_constant)
    B, SE, V, E, S = ols_b_h(Y, X, dfc)
    return B, SE, V, E, S

# %%
class ardlm_irfs:

    def __init__(self, irf, std, irfc, stdc):
        self.Irf, self.Std, self.Irfc, self.Stdc = irf, std, irfc, stdc

    def Irf_q(self, q):
        return self.Irf + spstats.norm.ppf(q)*self.Std

    def Irfc_q(self, q):
        return self.Irfc + spstats.norm.ppf(q)*self.Stdc

# %%
class irfs_ARDLm:

    def __init__(self, Irf_m, Irfc_m, Irf_std, Irfc_std, irf_spec):
        self.Irf_m, self.Irfc_m, self.Irf_std, self.Irfc_std = Irf_m, Irfc_m, Irf_std, Irfc_std
        self.Spec = irf_spec

    def Irf_q(self, q):
        return self.Irf_m + spstats.norm.ppf(q)*self.Irf_std

    def Irfc_q(self, q):
        return self.Irfc_m + spstats.norm.ppf(q)*self.Irfc_std

# %%
def get_irfs_ARDLm(By, Bx, /, *, model_spec, irf_spec):

    nI, nLy, nLx, nY, nX = model_spec['nI'], model_spec['nLy'], model_spec['nLx'], model_spec['nY'], model_spec['nX']
    nH = irf_spec['nH']

    Irf = np.full((nX, nH+1), np.nan)

    By = np.pad(By.reshape((nY, nLy)), ((0, 0), (0, max(nH-nLy+1, 0))))
    Bx = np.pad(Bx.reshape((nX, nI+nLx)), ((0, 0), (1-nI, max(nH-nI-nLx+1, 0))))

    for h in range(nH+1):
        for iX in range(nX):
            Irf[iX, h] = Irf[iX, :h][::-1]@By[0, :h] + Bx[iX, h]

    return Irf, np.cumsum(Irf, axis=1)

# %%
def get_irfs_std_ARDLm(B, V, /, *, model_spec, irf_spec):

    nC, nLy, nY, nX, nZ = model_spec['nC'], model_spec['nLy'], model_spec['nY'], model_spec['nX'], model_spec['nZ']
    nH, nR = irf_spec['nH'], irf_spec['nR']

    Irf_R = np.full((nR, nX, nH+1), np.nan)
    Irfc_R = np.full((nR, nX, nH+1), np.nan)

    B_R = np.random.multivariate_normal(B, V, size=nR)
    for r in range(nR):
        B_r = B_R[r].reshape((1, -1))
        _, _, By_r, Bx_r = np.split(B_r, np.cumsum([nC, nZ, nLy*nY]), axis=1)
        irf_r, cirf_r = get_irfs_ARDLm(By_r, Bx_r, model_spec=model_spec, irf_spec=irf_spec)
        Irf_R[r], Irfc_R[r] = irf_r, cirf_r

    Irf_Std = Irf_R.std(axis=0)
    Irfc_Std = Irfc_R.std(axis=0)

    return Irf_Std, Irfc_Std

# %% ARDL model

class ARDLm(model):

    def __init__(self, Data, /, *, Y_var, X_vars, Z_vars=None, sample=None, add_constant=True, contemporaneous_impact=True, nLy=None, nLx=None, dfc=True):
        super().__init__()

        if not isinstance(Data, pd.core.frame.DataFrame):
            raise TypeError('data must be in pandas DataFrame format')

        self.Data.Data = Data[[Y_var, *X_vars, *Z_vars]] if Z_vars else Data[[Y_var, *X_vars]]
        self.Data.Ydata, self.Data.Xdata, self.Data.Zdata = Data[[Y_var]], Data[X_vars], Data[Z_vars] if Z_vars else Data[[]]

        nC = 1 if add_constant else 0
        nI = 1 if contemporaneous_impact else 0

        self.Irfs = None
        self.Forecasts = None

        self.set_sample(Y_var, X_vars, Z_vars, sample, nC, nI, nLy, nLx, dfc)

    def fit(self):

        sample_idx, nC, nI, nLy, nLx, nY, nX, nZ, dfc = self.Spec['sample_idx'], self.Spec['nC'], self.Spec['nI'], self.Spec['nLy'], self.Spec['nLx'], self.Spec['nY'], self.Spec['nX'], self.Spec['nZ'], self.Spec['dfc']
        Ydata, Xdata, Zdata = self.Data.Ydata, self.Data.Xdata, self.Data.Zdata
        nL = max(nLy, nLx)

        y, X, Z = Ydata.iloc[sample_idx[0]-nL:sample_idx[1]+1].values.T, Xdata.iloc[sample_idx[0]-nL:sample_idx[1]+1].values.T, Zdata.iloc[sample_idx[0]-nL:sample_idx[1]+1].values.T

        if not use_numba:
            B, SE, V, U, S = fit_ardl_h(y, X, Z, nC, nI, nLy, nLx, dfc)
        else:
            B, SE, V, U, S = fit_ardl_h_njit(y, X, Z, nC, nI, nLy, nLx, dfc)

        B, SE = np.squeeze(B), np.squeeze(SE)

        Bc, Bz, By, Bx = np.split(B, np.cumsum([nC, nZ, nLy*nY]))
        SEc, SEz, SEy, SEx = np.split(SE, np.cumsum([nC, nZ, nLy*nY]))

        Bx = np.squeeze(Bx.reshape((nI+nLx, nX)))
        SEx = np.squeeze(SEx.reshape((nI+nLx, nX)))

        self.Est = {'Bc': Bc, 'Bz': Bz if Bz.size>0 else None, 'By': By, 'Bx': Bx, 'SEc': SEc, 'SEz': SEz if SEz.size>0 else None, 'SEy': SEy, 'SEx': SEx, 'B': B, 'SE': SE, 'V': V, 'U': U, 'S': S}

        return self

    def irf(self, **irf_spec):

        irf_spec_default = self.Irfs.Spec if hasattr(self.Irfs, 'Spec') else {'nH': 12, 'ci': 'bs', 'nR': 100}
        irf_spec = {**irf_spec_default, **irf_spec} if irf_spec else irf_spec_default

        model_spec = self.Spec
        By, Bx, B, V = self.Est['By'], self.Est['Bx'], self.Est['B'], self.Est['V']

        # IRF at means
        Irf_m, Irfc_m = get_irfs_ARDLm(By, Bx, model_spec=model_spec, irf_spec=irf_spec)

        # IRF standard deviations
        Irf_std, Irfc_std = get_irfs_std_ARDLm(B, V, model_spec=model_spec, irf_spec=irf_spec)

        Irfs = irfs_ARDLm(Irf_m, Irfc_m, Irf_std, Irfc_std, irf_spec)

        self.Irfs = Irfs

        return self

    def set_sample(self, Y_var, X_vars, Z_vars, sample, nC, nI, nLy, nLx, dfc):

        Ydata, Xdata, Zdata = self.Data.Ydata, self.Data.Xdata, self.Data.Zdata

        (_, nY), (_, nX), (_, nZ) = Ydata.shape, Xdata.shape, Zdata.shape
        nL = max(nLy, nLx)

        NaN_free_idx = get_NaN_free_idx(self.Data.Data)

        if sample is None:
            sample_idx = [nL+NaN_free_idx[0], NaN_free_idx[1]]
        elif Ydata.index.get_loc(sample[0]) >= nL+NaN_free_idx[0] and NaN_free_idx[1] <= Ydata.index.get_loc(sample[1]):
            sample_idx = [Ydata.index.get_loc(sample[0]), Ydata.index.get_loc(sample[1])]
        else:
            raise KeyError('Provided sample is outside of the available data range')

        nT = int(sample_idx[1] - sample_idx[0] + 1)
        sample = [Ydata.index[sample_idx[0]].strftime('%Y-%m-%d'), Ydata.index[sample_idx[1]].strftime('%Y-%m-%d')]

        self.Spec = {'Y_var': Y_var, 'X_vars': X_vars, 'Z_vars': Z_vars, 'sample': sample, 'sample_idx': sample_idx, 'nC': nC, 'nI': nI, 'nLy': nLy, 'nLx': nLx, 'nY': nY, 'nX': nX, 'nZ': nZ, 'nT': nT, 'dfc': dfc}
        self.__Default_Spec = {**self.Spec}

        self.fit()
        self.irf()

        return self

    def change_sample(self, sample=None):

        default_sample = self.__Default_Spec['sample']
        sample = default_sample if sample is None else sample

        Ydata = self.Data.Ydata

        Ydata_sample = Ydata.loc[sample[0]:sample[1]]

        if dt.datetime.strptime(default_sample[0], '%Y-%m-%d') <= Ydata_sample.index[0] and Ydata_sample.index[-1] <= dt.datetime.strptime(default_sample[1], '%Y-%m-%d'):
            sample_idx = [Ydata.index.get_loc(Ydata_sample.iloc[0].name), Ydata.index.get_loc(Ydata_sample.iloc[-1].name)]
        else:
            raise KeyError('Provided sample is outside of the available data range')

        nT = int(sample_idx[1] - sample_idx[0] + 1)
        sample = [Ydata.index[sample_idx[0]].strftime('%Y-%m-%d'), Ydata.index[sample_idx[1]].strftime('%Y-%m-%d')]

        self.Spec['nT'] = nT
        self.Spec['sample'] = sample
        self.Spec['sample_idx'] = sample_idx

        self.fit()
        self.irf()

        return self

# %% ARDL
class ardlm(model):

    def __init__(self, Ydata, Xdata, /, *, Zdata=None, nLy=None, nLx=None, nH=12, nR=100, add_constant=True):

        super().__init__()
        Ydata, Xdata, Zdata = self.do_data(Ydata, Xdata, Zdata)
        self.Data.Ydata, self.Data.Xdata, self.Data.Zdata = Ydata, Xdata, Zdata
        self.add_constant = add_constant
        self.Spec.nLy, self.Spec.nLx = nLy, nLx
        self.Spec.nC = 1 if add_constant else 0

        self.fit()
        self.irf(nH, nR)

    def fit(self):

        nC, nLy, nLx = self.Spec.nC, self.Spec.nLy, self.Spec.nLx
        Ydata, Xdata, Zdata = self.Data.Ydata, self.Data.Xdata, self.Data.Zdata

        y, X, Z = Ydata.values.T, Xdata.values.T, Zdata.values.T

        n1 = y.shape[1]
        nL = max(nLy, nLx)
        nT = n1 - nL

        Y = y.reshape(1, -1)

        W = np.ones((nC, n1))
        W = np.vstack((W, Z))
        for l in range(1, nLy+1):
            W = np.vstack((W, np.roll(y, l, axis=1)))
        for l in range(1, nLx+1):
            W = np.vstack((W, np.roll(X, l, axis=1)))

        (nY,_), (nX,_), (nZ,_), (nW,_) = Y.shape, X.shape, Z.shape, W.shape

        Y, W = Y[:, nL:], W[:, nL:]

        B, SE, V, E, S = ols_h(Y, W)

        B, SE = np.squeeze(B), np.squeeze(SE)

        Bc, Bz, By, Bx = np.split(B, np.cumsum([nC, nZ, nLy*nY]))
        SEc, SEz, SEy, SEx = np.split(SE, np.cumsum([nC, nZ, nLy*nY]))

        Bx = np.squeeze(Bx.reshape((nLx, nX)).T)

        self.Est.B, self.Est.SE, self.Est.V, self.Est.E, self.Est.S = B, SE, V, E, S
        self.Est.Bc, self.Est.Bz, self.Est.By, self.Est.Bx = Bc, Bz, By, Bx
        self.Est.SEc, self.Est.SEz, self.Est.SEy, self.Est.SEx = SEc, SEz, SEy, SEx
        self.Spec.nY, self.Spec.nX, self.Spec.nZ, self.Spec.nW, self.Spec.nT = nY, nX, nZ, nW, nT

        return self

    def irf(self, nH, nR):

        B, By, Bx, V = self.Est.B, self.Est.By, self.Est.Bx, self.Est.V
        nLy, nLx = self.Spec.nLy, self.Spec.nLx
        nC, nY, nX, nZ = self.Spec.nC, self.Spec.nY, self.Spec.nX, self.Spec.nZ

        Irf, Irfc = self.get_irf(By, Bx, nLy, nLx, nY, nX, nH)
        Irf_std, Irfc_std = self.get_irf_std(B, V, nLy, nLx, nC, nY, nX, nZ, nH, nR)

        self.Irfs = ardlm_irfs(Irf, Irf_std, Irfc, Irfc_std)
        self.Irfs.nH, self.Irfs.nR = nH, nR

        return self

    def get_irf(self, By, Bx, nLy, nLx, nY, nX, nH):

        Irf = np.full((nX, nH+1), np.nan)

        By = np.pad(By.reshape((nY, nLy)),((0, 0),(0, nH-nLy+1)))
        Bx = np.pad(Bx.reshape((nX, nLx)),((0, 0),(0, nH-nLx+1)))

        Bx = np.hstack((np.zeros((nX, 1)),Bx))
        for h in range(nH+1):
            for iX in range(nX):
                Irf[iX, h] = Irf[iX, :h][::-1]@By[0, :h] + Bx[iX, h]

        return Irf, np.cumsum(Irf, axis=1)

    def get_irf_std(self, B, V, nLy, nLx, nC, nY, nX, nZ, nH, nR):

        Irf_R = np.full((nR, nX, nH+1), np.nan)
        Irfc_R = np.full((nR, nX, nH+1), np.nan)

        B_R = np.random.multivariate_normal(B,V, size=nR)
        for r in range(nR):
            B_r = B_R[r].reshape((1, -1))
            _, _, By_r, Bx_r = np.split(B_r, np.cumsum([nC, nZ, nLy*nY]), axis=1)
            irf_r, cirf_r = self.get_irf(By_r, Bx_r, nLy, nLx, nY, nX, nH)
            Irf_R[r], Irfc_R[r] = irf_r, cirf_r

        Irf_Std = Irf_R.std(axis=0)
        Irfc_Std = Irfc_R.std(axis=0)

        return Irf_Std, Irfc_Std

    def do_data(self, Ydata, Xdata, Zdata):

        def make_df(Data):
            if isinstance(Data, np.ndarray) or isinstance(Data, pd.Series):
                Data = pd.DataFrame(Data)
            assert Data.shape[0] > Data.shape[1], 'data must be in column format'
            return Data

        if Zdata is None:
            Zdata = np.full((Ydata.shape[0], 0), np.nan)

        Ydata, Xdata, Zdata = make_df(Ydata), make_df(Xdata), make_df(Zdata)

        assert Ydata.shape[1] == 1, 'Ydata must contain only one column'
        assert Ydata.shape[0] == Xdata.shape[0] == Zdata.shape[0], 'data must have the same length'

        return Ydata, Xdata, Zdata

# %% Nelson-Siegel model
class nsm:

    def __init__(self, yields, tau, lam, classic=True):

        if len(yields.shape) == 1:
            yields = yields[None, :]

        if yields.shape[1] != tau.shape[0]:
            raise SyntaxError('yields and tau must have the same length')

        self.yields = pd.DataFrame(yields)
        self.yields.columns = tau
        self.tau = tau
        self.lam = lam
        self.classic = classic
        self.fit()

    def getLoadings(self, tau, lam):
        if self.classic:
            b1l = np.ones_like(tau)
            b2l = np.array((1-np.exp(-lam*tau))/(lam*tau))
            b3l = np.array((1-np.exp(-lam*tau))/(lam*tau)-np.exp(-lam*tau))
        else:
            b1l = np.array((1-np.exp(-lam*tau))/(lam*tau))
            b2l = np.array((1-np.exp(-lam*tau))/(lam*tau)-np.exp(-lam*tau))
            b3l = np.ones_like(tau)-np.array((1-np.exp(-lam*tau))/(lam*tau))
        return np.hstack((b1l, b2l, b3l))

    def olsproj(self, yin, Xin):
        y = yin.copy()
        X = Xin.copy()
        if len(y.shape) == 1:
            y = y[:, None]
        yX = np.hstack((y, X))
        yXnan = np.isnan(yX).any(axis=1)
        y[yXnan, :], X[yXnan, :] = 0, 0
        b = np.linalg.solve(X.T@X, X.T@y)
        return b

    def fit(self):
        yields = self.yields.values
        tau = self.tau
        lam = self.lam
        X = self.getLoadings(tau, lam)

        betasT = np.full((3, yields.shape[0]), np.nan)
        for t, yld in enumerate(yields):
            betasT[:, t, None] = self.olsproj(yld.T, X)

        self.X = X
        self.betas = pd.DataFrame(betasT.T, index=self.yields.index, columns=['beta1', 'beta2', 'beta3'])
        self.predict(np.array(range(1, tau[-1]+1)))

    def predict(self, ptau=None):
        if ptau is None:
            ptau = self.tau
        lam = self.lam
        betas = self.betas.values
        X = self.getLoadings(ptau, lam)

        self.curve = pd.DataFrame(betas@X.T, index=self.yields.index, columns=ptau)
        self.ptau = ptau

    def plot(self, index):
        tau = self.tau
        ptau = self.ptau
        mpl.scatter(tau, self.yields.loc[index].values)
        mpl.plot(ptau, self.curve.loc[index].values)

# %% VAR model
class varms:

    def __init__(self, Data, nP):
        if Data.shape[0] > Data.shape[1]:
            Data = Data.T
        n0, n1 = Data.shape
        self.Data = Data
        self.nP = nP
        self.n0 = n0
        self.n1 = n1
        self.nT = n1 - nP
        self.nK = n0
        self.model.irfs = block()

    def fit(self):
        Data = self.Data
        _, n1 = Data.shape
        nP = self.nP
        nK = self.nK
        nT = n1 - nP
        Z = np.ones((1, n1))

        for p in range(1, 1+nP):
            Z = np.vstack((Z, np.roll(Data, p)))

        Z = Z[:, nP:]
        Y = Data[:, nP:]

        cB = (Y@Z.T)@(np.linalg.inv(Z@Z.T))
        c = cB[:, 0]
        B = cB[:, 1:].T.reshape((nP, nK, nK)).swapaxes(1, 2)
        U = Y-cB@Z
        S = (1/(nT-nP*nK-1))*(U@U.T)

        self.parameters = block()
        self.parameters.c = c
        self.parameters.B = B
        self.parameters.S = S
        self.residuals = block()
        self.residuals.rd = U

    def irf(self, nH, method='cholesky', idv=None, ins_names=None):

        self.nH = nH
        nT = self.nT
        nP = self.nP
        nK = self.nK
        B = self.parameters.B
        S = self.parameters.S
        U = self.residuals.rd

        Psi = np.zeros((nH, nK, nK))
        Psi[0] = np.eye(nK)
        for h in range(1, nH):
            for i in range(min(h, nP)):
                Psi[h] += Psi[h-i-1]@B[i]

        self.model.irfs.rd = Psi
        self.model.irfs.rdc = np.cumsum(Psi, 0)
        self.parameters.A0inv = block()

        if method == 'cholesky':
            A0inv = np.linalg.cholesky(S)
            self.model.irfs.ch = Psi@A0inv
            self.model.irfs.chc = np.cumsum(Psi@A0inv, 0)
            self.parameters.A0inv.ch = A0inv
            self.residuals.ch = np.linalg.inv(A0inv)@U
        if method == 'iv':
            if idv is None or ins_names is None:
                raise SyntaxError('Please provide an instrument for SVAR-IV identification')
            instrument = self.data[ins_names]
            A0inv = np.zeros((nK, nK))
            for v, ins in zip(idv, instrument.T):
#                 print(np.cov(U, ins[-nT:].T))
                A0inv[:, v] = np.cov(np.vstack((ins[-nT:], U)))[0, 1:]
                A0inv[:, v] = A0inv[:, v]/A0inv[v, v]
            self.model.irfs.iv = Psi@A0inv
            self.model.irfs.ivc = np.cumsum(Psi@A0inv, 0)
            self.parameters.A0inv.iv = A0inv
            self.iv = block()
            self.iv.idv = idv
            self.iv.ins_names = ins_names

# %% VAR-OLS
def varols(Data, nL):
    """
    Function to estimate VAR(P) model with P = nL using OLS
    """
    n0, n1 = Data.shape
    nT = n1 - nL
    nY = n0
    Z = np.ones((1, n1))

    for p in range(1, nL+1):
        Z = np.vstack((Z, np.roll(Data, p)))    # np.roll without axis argument performs the operation on a flattened array, but the result is unaffected due to the drop of first nL observations

    Z = C(Z[:, nL:])
    Y = C(Data[:, nL:])

    cB = np.linalg.solve(Z@C(Z.T), Z@C(Y.T)).T

    c = cB[:, 0]
    B = C(cB[:, 1:].T).reshape((nL, nY, nY)).transpose((0, 2, 1))
    U = Y-cB@Z
    S = (1/(nT-nL*nY-1))*(U@U.T)
    return c, B, U, S

varols_njit = nb.njit(varols)

# %% VAR simulate
def varsim(c, B, U, Y0):
    (nY, nT) = U.shape
    (_, nL) = Y0.shape
    Y = np.full(((nY, nL+nT)), np.nan)
    Y[:, :nL] = Y0

    for t in range(nL, nL+nT):
        # The methods (a) and (b) are equivalent
        ## (a)
        BB = C(B.transpose((0, 2, 1))).reshape((nL*nY, nY)).T
        Y[:, t] = c + (BB@C(Y[:, t-nL:t][:, ::-1].T).reshape((nL*nY, 1))).reshape((-1, )) + U[:, t-nL]
        ## (b)
        # Y_t = c + U[:, t-nL]
        # for l in range(nL):
        #     Y_t += B[l]@Y[:, t-l-1]
        # Y[:, t] = Y_t
    return Y

varsim_njit = nb.njit(varsim)

# %% simulate VAR
def sim_var(Y0, Bc, Bx, U):
    Y0, Bc, Bx, U = C(Y0), C(Bc), C(Bx), C(U)
    nY, nL = Y0.shape
    nY, nT = U.shape
    Y = np.full(((nY, nL+nT)), np.nan)
    Y[:, :nL] = Y0
    BX = C(Bx.transpose((0, 2, 1))).reshape((nL*nY, nY)).T
    for t in range(nL, nL+nT):
        # The methods (a) and (b) are equivalent
        ## (a)
        Y[:, t] = Bc.reshape(-1) + BX@C(Y[:, t-nL:t][:, ::-1].T).reshape(-1) + U[:, t-nL]
        ## (b)
        # Y_t = c + U[:, t-nL]
        # for l in range(nL):
        #     Y_t += B[l]@Y[:, t-l-1]
        # Y[:, t] = Y_t
    return Y

sim_var_njit = nb.njit(sim_var)

# %% simulate VAR
def sim_var_b(Y0, Bc, Bx, U):
    (nS, nY, nT) = U.shape
    (_, nL) = Y0.shape
    Yy = np.full(((nS, nY, nL+nT)), np.nan)
    Yy[:, :, :nL] = Y0

    BX = C(Bx.transpose((0, 2, 1))).reshape((nL*nY, nY)).T

    for t in range(nL, nL+nT):
        # The methods (a) and (b) are equivalent
        ## (a)
        Yy[:, :, t] = np.repeat(Bc[np.newaxis, :], nS, axis=0) + (BX@C(Yy[:, :, t-nL:t][:, :, ::-1].T).reshape((nS, nL*nY)).T).T + U[:, :, t-nL]
        ## (b)
        # for s in range(nS):
        #     Y_t = c + U[:, t-nL]
        #     for l in range(nL):
        #         Y_t += B[l]@Yy[:, :, t-l-1]
        #     Yy[s, :, t] = Y_t
    return Yy

sim_var_b_njit = nb.njit(sim_var_b)

# %% get Psi from B
def get_Psi_from_Bx(B, nH):
    (nL, nY, _) = B.shape
    Psi = np.zeros((nH+1, nY, nY))
    Psi[0] = np.eye(nY)
    for h in range(1, nH+1):
        for i in range(min(h, nL)):
            Psi[h] += C(Psi[h-i-1])@C(B[i])
    return Psi

get_Psi_from_Bx_njit = nb.njit(get_Psi_from_Bx)

# %% Get A0inv
def get_A0inv(method=None, U=None, S=None, idv=None, M=None):
    nY, _ = U.shape
    if method is None:
        A0inv = np.eye(nY)
    if method == 'ch':
        A0inv = np.linalg.cholesky(S)
    if method == 'iv':
        method_ = 1
        A0inv = np.sqrt(np.diag(np.diag(S)))
#         A0inv = np.zeros((nY, nY))
        if method_ == 0:
            for v, m in zip(idv, M):
                mU = np.vstack((m, U))
                mU_nan = np.isnan(mU)
                mU = mU[:, ~mU_nan.any(axis=0)]
                if mU.shape[1] < 10:
                    raise ValueError('Not enough observations to perform SVAR-IV identification')
                centered = False
                if centered:
                    S_mU = np.cov(mU)
                else:
                    S_mU = (1/mU.shape[1])*(mU@mU.T)
                method__ = 'regression'

                if method__ == 'regression':
                    X = np.vstack((np.ones((1, mU.shape[1])), mU[0:1, :]))
                    Y = mU[1:, :]
                    beta1 = np.linalg.solve(X@X.T, X@Y.T)[1, :]

                if method__ == 'moments':
                    beta1 = S_mU[1:, 0]

                # normalize
                beta1 = beta1[:]/beta1[v]
    #             A0inv[:, v] = (insUcov[1:, 0]/insUstd[0]).T # st. dev. of explained part
    #             A0inv[:, v] = (insUcov[1:, 0]/insUcov[v+1, 0]).T # st. dev. of residual
    #             A0inv[:, v] = (insUcov[1:, 0]/(insUcov[v+1, 0]/insUstd[v+1])).T # st. dev. of residual
    #             A0inv[:, v] = A0inv[:, v]/A0inv[v, v] # unit
                A0inv[:, v] = beta1.T

        if method_ == 1:
            nM = M.shape[0]
            not_idv = np.array([_ for _ in range(nY) if _ not in idv])

            # Reorder instrumented residuals first
            U_ = np.vstack((U[idv, :], U[not_idv, :]))
            MU = np.vstack((M, U_))
            # Remove time periods with NaNs
            MU_nan = np.isnan(MU)
            MU = MU[:, ~MU_nan.any(axis=0)]
            if MU.shape[1] < 10:
                raise ValueError('Not enough observations to perform SVAR-IV identification')

            b11, b21 = iv_block_njit(MU, nM)

            idv_array = np.array(idv)
            not_idv_array = np.array(not_idv)

            A0inv[idv_array[:, None], idv_array] = b11
            A0inv[not_idv_array[:, None], idv_array] = b21

    return A0inv

# %% get A0inv
@nb.njit # not used
def get_A0inv_njit(method=None, U=None, S=None, idv=None, M=None):
    nY, _ = U.shape
    if method is None:
        A0inv = np.eye(nY)
    if method == 'ch':
        A0inv = np.linalg.cholesky(S)
    if method == 'iv':
        method_ = 1
        A0inv = np.sqrt(np.diag(np.diag(S)))
#         A0inv = np.zeros((nY, nY))
        if method_ == 0:
            for v, m in zip(idv, M):
                mU = np.vstack((m, U))
                mU_nan = np.isnan(mU)
                mU = mU[:, ~mU_nan.any(axis=0)]
                if mU.shape[1] < 10:
                    raise ValueError('Not enough observations to perform SVAR-IV identification')

                centered = False
                if centered:
                    S_mU = np.cov(mU)
                else:
                    S_mU = (1/mU.shape[1])*(mU@mU.T)

                method__ = 'regression'
                if method__ == 'regression':
                    X = np.vstack((np.ones((1, mU.shape[1])), mU[0:1, :]))
                    Y = mU[1:, :]
                    beta1 = np.linalg.solve(X@X.T, X@Y.T)[1, :]
                if method__ == 'moments':
                    beta1 = S_mU[1:, 0]

                # normalize
                beta1 = beta1[:]/beta1[v]
    #             A0inv[:, v] = (insUcov[1:, 0]/insUstd[0]).T # st. dev. of explained part
    #             A0inv[:, v] = (insUcov[1:, 0]/insUcov[v+1, 0]).T # st. dev. of residual
    #             A0inv[:, v] = (insUcov[1:, 0]/(insUcov[v+1, 0]/insUstd[v+1])).T # st. dev. of residual
    #             A0inv[:, v] = A0inv[:, v]/A0inv[v, v] # unit
                A0inv[:, v] = beta1.T

        if method_ == 1:
            nM = M.shape[0]
            idv1 = []
            for _ in idv:
                idv1.append(_)
            not_idv = []
            for _ in range(nY):
                in_idv = False
                for _1 in idv:
                    if _ == _1:
                        break
                if not in_idv:
                    not_idv.append(_)

            # Reorder instrumented residuals first
            U_ = np.full_like(U, np.nan)
            for (iU_, iU) in enumerate([_ for _ in idv]+[_ for _ in not_idv]):
                U_[iU_, :] = U[iU, :]
            MU = np.vstack((M, U_))
            # Remove time periods with NaNs
            MU_nan = np.isnan(MU)
            MU = MU[:, ~MU_nan.any(axis=0)]
            if MU.shape[1] < 10:
                raise ValueError('Not enough observations to perform SVAR-IV identification')

            b11, b21 = iv_block_njit(MU, nM)

            for (ib, iA) in enumerate(idv):
                for (jb, jA) in enumerate(idv):
                    A0inv[iA, jA] = b11[ib, jb]
            # A0inv[idv[:, None], idv] = b11
#                 print(A0inv[idv, :][:, idv])
            for (ib, iA) in enumerate(not_idv):
                for (jb, jA) in enumerate(idv):
                    A0inv[iA, jA] = b21[ib, jb]
            # A0inv[not_idv[:, None], idv] = b21
#                 print(A0inv)
    return A0inv

# ### IV-identification

def iv_block(MU, nM):
    # The formulas from Mertens & Ravn (2013) Appendix A
    S_mumu = (1/MU.shape[1])*(MU@MU.T)
    S_uu = S_mumu[nM:, nM:]
    S_mu = S_mumu[:nM, nM:]
    S_mu1 = S_mu[:, :nM]
    S_mu2 = S_mu[:, nM:]
    S11 = S_uu[:nM, :nM]
    S21 = S_uu[nM:, :nM]
    S22 = S_uu[nM:, nM:]
    b21_b11_1 = (np.linalg.inv(S_mu1)@S_mu2).T
#     b21_b11_1 = np.linalg.solve(S_mu1, S_mu2).T
    Z = b21_b11_1@S11@b21_b11_1.T-(S21@b21_b11_1.T+b21_b11_1@S21.T)+S22
    b12_b12_T = (S21-b21_b11_1@S11).T@np.linalg.inv(Z)@(S21-b21_b11_1@S11)
#     b12_b12_T = (S21-b21_b11_1@S11).T@np.linalg.solve(Z, S21-b21_b11_1@S11)
    b22_b22_T = S22+b21_b11_1@(b12_b12_T-S11)@b21_b11_1.T
    b12_b22_1 = (b12_b12_T@b21_b11_1.T+(S21-b21_b11_1@S11).T)@b22_b22_T
    b11_b11_T = S11-b12_b12_T
    S1_S1_T = (np.eye(nM)-b12_b22_1@b21_b11_1)@b11_b11_T@(np.eye(nM)-b12_b22_1@b21_b11_1).T
    S1 = np.linalg.cholesky(S1_S1_T)
    b11_S1_1 = np.linalg.inv(np.eye(nM)-b12_b22_1@b21_b11_1)
#     b11_S1_1 = np.linalg.solve(np.eye(nM)-b21_b11_1.T@b12_b22_1.T, np.eye(nM)).T
    b21_S1_1 = b21_b11_1@np.linalg.inv(np.eye(nM)-b12_b22_1@b21_b11_1)
#     b21_S1_1 = np.linalg.solve(np.eye(nM)-b21_b11_1.T@b12_b22_1.T, b21_b11_1.T).T
    b11 = b11_S1_1@S1
    b21 = b21_S1_1@S1
    return b11, b21


@nb.njit
def iv_block_njit(MU, nM):
    # The formulas from Mertens & Ravn (2013) Appendix A
    MU = np.ascontiguousarray(MU)
    MU_T = np.ascontiguousarray(MU.T)
    S_mumu = (1/MU.shape[1])*(MU@MU_T)
    S_uu = S_mumu[nM:, nM:]
    S_mu = S_mumu[:nM, nM:]
    S_mu1 = np.ascontiguousarray(S_mu[:, :nM])
    S_mu2 = np.ascontiguousarray(S_mu[:, nM:])
    S11 = np.ascontiguousarray(S_uu[:nM, :nM])
    S21 = np.ascontiguousarray(S_uu[nM:, :nM])
    S22 = np.ascontiguousarray(S_uu[nM:, nM:])
#     b21_b11_1 = (np.linalg.inv(S_mu1)@S_mu2).T
    b21_b11_1 = np.linalg.solve(S_mu1, S_mu2).T
    Z = b21_b11_1@S11@b21_b11_1.T-(S21@b21_b11_1.T+b21_b11_1@S21.T)+S22
#     b12_b12_T = (S21-b21_b11_1@S11).T@np.linalg.inv(Z)@(S21-b21_b11_1@S11)
    b12_b12_T = (S21-b21_b11_1@S11).T@np.linalg.solve(Z, S21-b21_b11_1@S11)
    b22_b22_T = S22+b21_b11_1@(b12_b12_T-S11)@b21_b11_1.T
    b12_b22_1 = (b12_b12_T@b21_b11_1.T+(S21-b21_b11_1@S11).T)@b22_b22_T
    b11_b11_T = S11-b12_b12_T
    S1_S1_T = (np.eye(nM)-b12_b22_1@b21_b11_1)@b11_b11_T@(np.eye(nM)-b12_b22_1@b21_b11_1).T
    S1 = np.linalg.cholesky(S1_S1_T)
#     b11_S1_1 = np.linalg.inv(np.eye(nM)-b12_b22_1@b21_b11_1)
    b11_S1_1 = np.linalg.solve(np.eye(nM)-b21_b11_1.T@b12_b22_1.T, np.eye(nM)).T
#     b21_S1_1 = b21_b11_1@np.linalg.inv(np.eye(nM)-b12_b22_1@b21_b11_1)
    b21_S1_1 = np.linalg.solve(np.eye(nM)-b21_b11_1.T@b12_b22_1.T, b21_b11_1.T).T
    b11 = b11_S1_1@S1
    b21 = b21_S1_1@S1
    return b11, b21

# %% get SIRF from IRF
def get_sirf_from_irf(Psi, A0inv, impulse='unit'):
    if impulse == 'unit':
        impulse_scale = np.diag(1/np.diag(A0inv))
    elif impulse == '1sd':
        impulse_scale = np.eye(A0inv.shape[0])
    Impact = A0inv@impulse_scale
    if not use_numba:
        Irf, Irfc = Psi@Impact, np.cumsum(Psi@Impact, 0)
    else:
        Irf, Irfc = np.full_like(Psi, np.nan), np.full_like(Psi, np.nan)
        for h in range(Psi.shape[0]):
            Irf[h] = C(Psi[h])@C(Impact)
            if h == 0:
                Irfc[h] = Irf[h]
            else:
                Irfc[h] = Irfc[h-1] + Irf[h]
    return Irf, Irfc

get_sirf_from_irf_njit = nb.njit(get_sirf_from_irf)

# %% Bootstrap
def bs_irf(Y, U, B, /, *, model_spec, irf_spec, bs_dist = 'Rademacher'):
    nC, nL, nY, nT, dfc = model_spec['nC'], model_spec['nL'], model_spec['nY'], model_spec['nT'], model_spec['dfc']
    nH, ci = irf_spec['nH'], irf_spec['ci']
    Y0 = Y[:, :nL]
    Bc, Bx = split_C_B(B, nC, nL, nY)
    if ci == 'bs':
        idx_r = np.random.choice(nT, size=nT)
        U_ = U[:, idx_r]
    if ci == 'wbs':
        if bs_dist == 'Rademacher':
            rescale = np.random.choice((-1, 1), size=(1, nT))
        if bs_dist == 'Normal':
            rescale = np.random.normal(size=(1, nT))
        U_ = U*rescale
    if not use_numba:
        Y_ = sim_var(Y0, Bc, Bx, U_)
        B_, _, _, _, S_ = fit_var_h(Y_, nC, nL, dfc)
        _, Bx_ = split_C_B(B_, nC, nL, nY)
        A0inv_ = np.linalg.cholesky(S_)
        ir_, irc_ = get_irfs_VARm(Bx_, A0inv_, nH)
    else:
        Y_ = sim_var_njit(Y0, Bc, Bx, U_)
        B_, _, _, _, S_ = fit_var_h_njit(Y_, nC, nL, dfc)
        _, Bx_ = split_C_B_njit(B_, nC, nL, nY)
        A0inv_ = np.linalg.cholesky(S_)
        ir_, irc_ = get_irfs_VARm_njit(Bx_, A0inv_, nH)
    return ir_, irc_

# %% Bootstrap
def bs_fcs(Y, U, B, /, *, model_spec, forecast_spec, bs_dist = 'Rademacher'):
    nC, nL, nY, nT = model_spec['nC'], model_spec['nL'], model_spec['nY'], model_spec['nT']
    nF, ci = forecast_spec['nF'], forecast_spec['ci']
    Y0 = Y[:, -nL:]
    Bc, Bx = split_C_B(B, nC, nL, nY)
    if ci == 'bs':
        idx_r = np.random.choice(nT, size=nF)
        U_ = U[:, idx_r]
    if ci == 'wbs':
        if bs_dist == 'Rademacher':
            rescale = np.random.choice((-1, 1), size=(1, nF))
        if bs_dist == 'Normal':
            rescale = np.random.normal(size=(1, nF))
        idx_r = np.random.choice(nT, size=nF)
        U_ = U[:, idx_r]*rescale
    if not use_numba:
        fcs_ = sim_var(Y0, Bc, Bx, U_)
    else:
        fcs_ = sim_var_njit(Y0, Bc, Bx, U_)
    fcs_, fcsc_ = fcs_[:, nL-1:], np.cumsum(fcs_[:, nL-1:], 1)
    return fcs_, fcsc_

# %% Bootstrap
def get_bs(Y, c, B, U, S, UM, nL, nY, nH, nT, /, *, method=None, impulse=None, cl=None, ci=None, idv=None, M=None):
    Y0_r = Y[:, :nL]
    if ci == 'bs':
        idx_r = np.random.choice(nT, size=nT)
        rescale = np.ones((1, nT))
        UM_r = UM[:, idx_r]*rescale
    if ci == 'wbs':
        bs_dist = 'Rademacher'
        if bs_dist == 'Rademacher':
            rescale = np.random.choice((-1, 1), size=(1, nT))
        if bs_dist == 'Normal':
            rescale = np.random.normal(size=(1, nT))
        UM_r = UM[:, :]*rescale
    U_r = UM_r[:nY, :]
    M_r = UM_r[nY:, :]
    if not use_numba:
        Y_r = varsim(c, B, U_r, Y0_r)
        _, B_r_, U_r_, S_r_ = varols(Y_r, nL)
        Psi_ = get_Psi_from_Bx(B_r_, nH)
        A0inv_ = get_A0inv(method=method, U=U_r_, S=S_r_, idv=idv, M=M_r)
        ir_, irc_ = get_sirf_from_irf(Psi_, A0inv_, impulse)
    else:
        Y_r = varsim_njit(c, B, U_r, Y0_r)
        _, B_r_, U_r_, S_r_ = varols_njit(Y_r, nL)
        Psi_ = get_Psi_from_Bx_njit(B_r_, nH)
        A0inv_ = get_A0inv(method=method, U=U_r_, S=S_r_, idv=idv, M=M_r)
        ir_, irc_ = get_sirf_from_irf_njit(Psi_, A0inv_, impulse)
    return ir_, irc_

get_bs_njit = nb.njit(get_bs)

# %% get IRFs
def get_irfs(Y, c, B, U, S, /, *, nH, method=None, impulse=None, cl=None, ci=None, nR=1000, idv=None, M=None):
    nL, nY, _ = B.shape
    _, n1 = Y.shape
    nT = n1 - nL

#         Psi = get_Psi_from_Bx(B, nH)
    Psi = get_Psi_from_Bx_njit(B, nH)
    A0inv = get_A0inv(method=method, U=U, S=S, idv=idv, M=M)
    irm, irmc = get_sirf_from_irf(Psi, A0inv, impulse)
    ir = block()
    irc = block()
    ir.mean = irm
    irc.mean = irmc

    if ci is not None:
        IR = np.full((nR, nH+1, nY, nY), np.nan)
        IRC = np.full((nR, nH+1, nY, nY), np.nan)
        if method == 'ch':
            M = [0 for _ in range(nT)]
        UM = np.vstack((U, M))
        for r in range(nR):
            if (r+1) % 100 == 0:
                print(f'\r Bootstrap {r+1}/{nR}', end='\r', flush=True)
            ir_, irc_ = get_bs(Y, c, B, U, S, UM, nL, nY, nH, nT, method=method, impulse=impulse, cl=cl, ci=ci, idv=idv, M=M)
#             ir_, irc_ = bs_njit(Y, c,B, U, S, UM, nL, nY, nH, nT, method=method, impulse=impulse, cl=cl, ci=ci, idv=idv, M=M)
            IR[r], IRC[r] = ir_, irc_
        print(end='\n')
        # ir.q500 = np.quantile(IR, 0.500, axis=0)
        # irc.q500 = np.quantile(IRC, 0.500, axis=0)
        if 0.99 in cl:
            ir.q005 = np.quantile(IR, 0.005, axis=0)
            ir.q995 = np.quantile(IR, 0.995, axis=0)
            irc.q005 = np.quantile(IRC, 0.005, axis=0)
            irc.q995 = np.quantile(IRC, 0.995, axis=0)
        if 0.95 in cl:
            ir.q025 = np.quantile(IR, 0.025, axis=0)
            ir.q975 = np.quantile(IR, 0.975, axis=0)
            irc.q025 = np.quantile(IRC, 0.025, axis=0)
            irc.q975 = np.quantile(IRC, 0.975, axis=0)
        if 0.90 in cl:
            ir.q050 = np.quantile(IR, 0.050, axis=0)
            ir.q950 = np.quantile(IR, 0.950, axis=0)
            irc.q050 = np.quantile(IRC, 0.050, axis=0)
            irc.q950 = np.quantile(IRC, 0.950, axis=0)
        if 0.80 in cl:
            ir.q100 = np.quantile(IR, 0.100, axis=0)
            ir.q900 = np.quantile(IR, 0.900, axis=0)
            irc.q100 = np.quantile(IRC, 0.100, axis=0)
            irc.q900 = np.quantile(IRC, 0.900, axis=0)
        if 0.68 in cl:
            ir.q160 = np.quantile(IR, 0.160, axis=0)
            ir.q840 = np.quantile(IR, 0.840, axis=0)
            irc.q160 = np.quantile(IRC, 0.160, axis=0)
            irc.q840 = np.quantile(IRC, 0.840, axis=0)
        if 0.50 in cl:
            ir.q250 = np.quantile(IR, 0.250, axis=0)
            ir.q750 = np.quantile(IR, 0.750, axis=0)
            irc.q250 = np.quantile(IRC, 0.250, axis=0)
            irc.q750 = np.quantile(IRC, 0.750, axis=0)

    return ir, irc, Psi, A0inv

# %%
def get_irfs_VARm(Bx, A0inv, nH):
    Psi = get_Psi_from_Bx(Bx, nH) if not use_numba else get_Psi_from_Bx_njit(Bx, nH)
    Irf, Irfc = get_sirf_from_irf(Psi, A0inv) if not use_numba else get_sirf_from_irf_njit(Psi, A0inv)
    Irf, Irfc = Irf.transpose((2, 1, 0)), Irfc.transpose((2, 1, 0))
    return Irf, Irfc

get_irfs_VARm_njit = nb.njit(get_irfs_VARm)

# %%

def get_forecasts_VARm(Y, B, /, *, model_spec, forecast_spec):
    nC, nL, nY = model_spec['nC'], model_spec['nL'], model_spec['nY']
    nF = forecast_spec['nF']
    _, n1 = Y.shape
    Bc, Bx = split_C_B(B, nC, nL, nY)
    Fcs = np.zeros((n1+nF, nY))
    Fcs[:n1, :] = Y.T
    for f in range(n1, n1+nF):
        for l in range(nL):
            Fcs[f] += Bx[l]@Fcs[f-l-1]
        Fcs[f] += Bc[:, 0]
    Fcs, Fcsc = Fcs[n1-1:].T, np.cumsum(Fcs[n1-1:].T, 1)
    return Fcs, Fcsc

# %%
def get_irfs_sim_VARm(Y, B, U, /, *, model_spec, irf_spec):

    nY, _ = Y.shape
    nH, ci, nR = irf_spec['nH'], irf_spec['ci'], irf_spec['nR']

    if ci in ['bs', 'wbs']:
        Irf_sim, Irfc_sim = np.full((nR, nY, nY, nH+1), np.nan), np.full((nR, nY, nY, nH+1), np.nan)
        for r in range(nR):
            Irf_sim[r], Irfc_sim[r] = bs_irf(Y, U, B, model_spec=model_spec, irf_spec=irf_spec)

    return Irf_sim, Irfc_sim

    # # Confidence intervals
    # if ci['ci'] in ['pbs', 'sim']:

    #     nR = ci['nR']

    #     Irf_sim = np.full((nR, nY, nY, nH+1), np.nan)
    #     Irfc_sim = np.full((nR, nY, nY, nH+1), np.nan)

    #     Bv = B.reshape((-1))
    #     Vv = sp.linalg.block_diag(*V)
    #     B_sim = np.random.multivariate_normal(Bv, Vv, size=nR)
    #     for r in range(nR):
    #         B_r = B_sim[r].reshape((nY, -1))
    #         _, Bx_r = split_C_B(B_r, nC, nL, nY)
    #         A0inv = np.linalg.cholesky(S)
    #         Irf_r, Irfc_r = get_irfs_VARm(Bx_r, A0inv, nH)
    #         Irf_sim[r], Irfc_sim[r] = Irf_r, Irfc_r

    #     return Irf, Irfc, Irf_sim, Irfc_sim
    # else:
    #     return Irf, Irfc, None, None

# %%
def get_fcs_sim_VARm(Y, B, U, /, *, model_spec, forecast_spec):

    nY, _ = Y.shape
    nF, ci, nR = forecast_spec['nF'], forecast_spec['ci'], forecast_spec['nR']

    if ci in ['bs', 'wbs']:
        Fcs_sim, Fcsc_sim = np.full((nR, nY, nF+1), np.nan), np.full((nR, nY, nF+1), np.nan)
        for r in range(nR):
            Fcs_sim[r], Fcsc_sim[r] = bs_fcs(Y, U, B, model_spec=model_spec, forecast_spec=forecast_spec)

    return Fcs_sim, Fcsc_sim

# %%
class irfs_VARm(irfs):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

# %%
class forecasts_VARm(forecasts):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

# %% VAR

class VARm(model):

    def __init__(self, Ydata, /, *, var_names=None, sample=None, add_constant=True, nL=1, dfc=True):

        super().__init__()

        if not isinstance(Ydata, pd.core.frame.DataFrame):
            raise TypeError('data must be in pandas DataFrame format')

        if var_names is None:
            var_names = Ydata.columns
        Ydata = Ydata[var_names]

        # Ydata, Xdata, Zdata = self.do_data(Ydata, Xdata, Zdata)
        self.Data.Ydata = Ydata
        nC = 1 if add_constant else 0

        self.Irfs = None
        self.Forecasts = None

        self.set_sample(var_names, sample, nC, nL, dfc)

    def fit(self):

        sample_idx, nC, nL, nY, dfc = self.Spec['sample_idx'], self.Spec['nC'], self.Spec['nL'], self.Spec['nY'], self.Spec['dfc']
        Ydata = self.Data.Ydata

        Y = Ydata.iloc[sample_idx[0]-nL:sample_idx[1]+1].values.T

        if not use_numba:
            B, SE, V, U, S = fit_var_h(Y, nC, nL, dfc)
        else:
            B, SE, V, U, S = fit_var_h_njit(Y, nC, nL, dfc)

        Bc, Bx = split_C_B(B, nC, nL, nY)
        SEc, SEx = split_C_B(SE, nC, nL, nY)

        self.Est = {'Bc': Bc, 'Bx': Bx, 'SEc': SEc, 'SEx': SEx, 'B': B, 'SE': SE, 'V': V, 'U': U, 'S': S}

        return self

    def check_stability(self):

        nL, nY = self.Spec['nL'], self.Spec['nY']
        Bx = self.Est['Bx']

        # companion matrix
        BB = np.zeros((nL*nY, nL*nY))
        BB[:nY, :] = np.hstack(Bx)
        BB[nY:, :-nY] = np.eye(nY*(nL-1))

        stable = np.all(np.abs(np.linalg.eigvals(BB)) < 1)

        self.Est['Stable'] = stable

        return self

    def irf(self, **irf_spec):

        irf_spec_default = self.Irfs.Spec if hasattr(self.Irfs, 'Spec') else {'nH': 12, 'ci': 'bs', 'nR': 100}
        irf_spec = {**irf_spec_default, **irf_spec} if irf_spec else irf_spec_default

        model_spec = self.Spec
        sample_idx, nL = self.Spec['sample_idx'], self.Spec['nL']
        nH = irf_spec['nH']
        Bx, B, U, S = self.Est['Bx'], self.Est['B'], self.Est['U'], self.Est['S']
        Ydata = self.Data.Ydata

        Y = Ydata.iloc[sample_idx[0]-nL:sample_idx[1]+1].values.T

        A0inv = np.linalg.cholesky(S)

        # IRF at means
        Irf_m, Irfc_m = get_irfs_VARm(Bx, A0inv, nH)

        # IRF simulations
        Irf_sim, Irfc_sim = get_irfs_sim_VARm(Y, B, U, model_spec=model_spec, irf_spec=irf_spec)

        Irfs = irfs_VARm(Irf_m, Irfc_m, Irf_sim, Irfc_sim, irf_spec=irf_spec)

        self.Irfs = Irfs

        return self

    def forecast(self, **forecast_spec):

        forecast_spec_default = self.Forecasts.Spec if hasattr(self.Forecasts, 'Spec') else {'nF': 12, 'ci': 'bs', 'nR': 100, 'period': None}
        forecast_spec = {**forecast_spec_default, **forecast_spec} if forecast_spec else forecast_spec_default

        nL = self.Spec['nL']
        B, U = self.Est['B'], self.Est['U']
        sample_idx = self.Spec['sample_idx']
        model_spec = self.Spec
        Ydata = self.Data.Ydata
        if forecast_spec['period'] is not None:
            period_idx = [sample_idx[0], Ydata.index.get_loc(forecast_spec['period'])]
        else:
            period_idx = sample_idx
        forecast_spec['period'] = Ydata.index[period_idx[1]].strftime('%Y-%m-%d')

        Y = Ydata.iloc[period_idx[0]-nL:period_idx[1]+1].values.T

        # Mean forecasts
        Fcs_m, Fcsc_m = get_forecasts_VARm(Y, B, model_spec=model_spec, forecast_spec=forecast_spec)

        # Simulated forecasts
        Fcs_sim, Fcsc_sim = get_fcs_sim_VARm(Y, B, U, model_spec=model_spec, forecast_spec=forecast_spec)

        Forecasts = forecasts_VARm(Fcs_m, Fcsc_m, Fcs_sim, Fcsc_sim, forecast_spec=forecast_spec)

        self.Forecasts = Forecasts

        return self

    def set_sample(self, var_names, sample, nC, nL, dfc):

        Ydata = self.Data.Ydata

        _, nY = Ydata.shape

        NaN_free_idx = get_NaN_free_idx(Ydata)

        if sample is None:
            sample_idx = [nL+NaN_free_idx[0], NaN_free_idx[1]]
        else:
            sample = [dt.datetime.strptime(sample[0], '%Y-%m-%d'), dt.datetime.strptime(sample[1], '%Y-%m-%d')]
            if nL+NaN_free_idx[0] <= Ydata.index.get_loc(sample[0]) and Ydata.index.get_loc(sample[1]) <= NaN_free_idx[1]:
                sample_idx = [Ydata.index.get_loc(sample[0]), Ydata.index.get_loc(sample[1])]
            else:
                raise KeyError(f'Provided sample {sample[0].strftime("%Y-%m-%d")} - {sample[1].strftime("%Y-%m-%d")} is outside of the available data range {Ydata.index[nL+NaN_free_idx[0]].strftime("%Y-%m-%d")} - {Ydata.index[NaN_free_idx[1]].strftime("%Y-%m-%d")}')

        nT = int(sample_idx[1] - sample_idx[0] + 1)
        sample = [Ydata.index[sample_idx[0]].strftime('%Y-%m-%d'), Ydata.index[sample_idx[1]].strftime('%Y-%m-%d')]

        self.Spec = {'var_names': var_names, 'sample': sample, 'sample_idx': sample_idx, 'nC': nC, 'nL': nL, 'nY': nY, 'nT': nT, 'dfc': dfc}
        self.__Default_Spec = {**self.Spec}

        self.fit()
        self.check_stability()

        return self

    def change_sample(self, sample=None):

        default_sample = self.__Default_Spec['sample']
        sample = default_sample if sample is None else sample

        Ydata = self.Data.Ydata

        Ydata_sample = Ydata.loc[sample[0]:sample[1]]

        if dt.datetime.strptime(default_sample[0], '%Y-%m-%d') <= Ydata_sample.index[0] and Ydata_sample.index[-1] <= dt.datetime.strptime(default_sample[1], '%Y-%m-%d'):
            sample_idx = [Ydata.index.get_loc(Ydata_sample.iloc[0].name), Ydata.index.get_loc(Ydata_sample.iloc[-1].name)]
        else:
            raise KeyError('Provided sample is outside of the available data range')

        nT = int(sample_idx[1] - sample_idx[0] + 1)
        sample = [Ydata.index[sample_idx[0]].strftime('%Y-%m-%d'), Ydata.index[sample_idx[1]].strftime('%Y-%m-%d')]

        self.Spec['nT'] = nT
        self.Spec['sample'] = sample
        self.Spec['sample_idx'] = sample_idx

        self.fit()
        self.check_stability()

        return self

# %%
class varm:

    # ==============================================================================================

    def __init__(self, Data,/,*, nL=None, var_names=None, sample=None):

        if Data.shape[0] < Data.shape[1]:
            Data = Data.T

        if isinstance(Data, pd.DataFrame):
            pass
        elif isinstance(Data,(pd.Series, np.ndarray)):
            Data = pd.DataFrame(Data)
            Data.columns = [str(i) for i in Data.columns]

        if var_names is None:
            var_names = Data.columns

        self.Data = Data
        self.model = block()
        self.model.spec = block()
        self.model.spec.var_names = var_names
        self.model.spec.nL = nL
        self.model.spec.nY = len(var_names)
        self.model.svar = block()
        self.model.irfs = block()
        self.model.irfs.spec = block()

        self.set_sample(sample)

    # ==============================================================================================

    def fit(self):
        isample = self.model.spec.isample
        nL = self.model.spec.nL
        var_names = self.model.spec.var_names
        Data = self.Data[var_names].iloc[isample[0]-nL:isample[1]+1, :].values

#         c, B, U, S = varols(Data.T, nL)
        c, B, U, S = varols_njit(Data.T, nL)

        self.model.parameters = block()
        self.model.parameters.c = c
        self.model.parameters.B = B
        self.model.parameters.S = S
        self.model.parameters.A0inv = block()
        self.model.residuals = block()
        self.model.residuals.rd = U.T

        if hasattr(self.model.svar, 'ch'):
            self.irf(method='ch')
        if hasattr(self.model.svar, 'iv'):
            self.irf(method='iv')

    # ==============================================================================================

    def irf(self, nH=None, method=None, impulse=None, cl=None, ci=None, nR=None, idv=None, ins_names=None):

        if nH is None:
            if hasattr(self.model.irfs.spec, 'nH'):
                nH = self.model.irfs.spec.nH
            else:
                nH = 48

        if impulse is None:
            if hasattr(self.model.irfs.spec, 'impulse'):
                impulse = self.model.irfs.spec.impulse
            else:
                impulse = 'unit'

        if cl is None:
            if hasattr(self.model.irfs.spec, 'cl'):
                cl = self.model.irfs.spec.cl
            else:
                cl = 0.95

        if ci is None:
            if hasattr(self.model.irfs.spec, 'ci'):
                ci = self.model.irfs.spec.ci
            else:
                ci = None

        if nR is None:
            if hasattr(self.model.irfs.spec, 'nR'):
                nR = self.model.irfs.spec.nR
            else:
                nR = 1000

        self.model.irfs.spec.nH = nH
        self.model.irfs.spec.impulse = impulse

        nL = self.model.spec.nL
        nY = self.model.spec.nY
        c = self.model.parameters.c
        B = self.model.parameters.B
        S = self.model.parameters.S
        U = self.model.residuals.rd.T
        isample = self.model.spec.isample
        var_names = self.model.spec.var_names
        Data = self.Data[var_names].iloc[isample[0]-nL:isample[1]+1, :].values
        Y = Data.T

        if isinstance(cl, float):
            cl = [cl]

        if isinstance(impulse, list) or isinstance(impulse, np.ndarray):
            impulse = np.diag(impulse)
        if isinstance(impulse, float):
            impulse = impulse*np.eye(nY)


        if method == 'ch':
            idv = None
            ins_names = None
            M = None

        if method == 'iv':
            if idv is None or ins_names is None:
                if hasattr(self.model.svar, 'iv'):
                    if hasattr(self.model.svar.iv, 'idv') and hasattr(self.model.svar.iv, 'ins_names'):
                        idv = self.model.svar.iv.idv
                        ins_names = self.model.svar.iv.ins_names
                    else:
                        raise SyntaxError('Please provide an instrument for SVAR-IV identification')
                else:
                    raise SyntaxError('Please provide an instrument for SVAR-IV identification')
            else:
                if isinstance(idv, int):
                    idv = [idv]
                if isinstance(ins_names, str):
                    ins_names = [ins_names]
            if len(idv) != len(ins_names):
                raise SyntaxError('The number of instruments must be equal the number of instrumented variables')

            instruments = self.Data[ins_names].iloc[isample[0]:isample[1]+1, :].values
            M = instruments.T

        if method in ('ch', 'iv'):
            ir, irc, Psi, A0inv = get_irfs(Y, c,B, U, S, nH=nH, method=method, impulse=impulse, cl=cl, ci=ci, nR=nR, idv=idv, M=M)
            self.model.irfs.spec.cl = cl
            if ci in ('bs', 'wbs'):
                self.model.irfs.spec.ci = ci
                self.model.irfs.spec.nR = nR
            irfs = block()
            irfs.ir = ir
            irfs.irc = irc

        if method == 'ch':
            self.model.irfs.ch = irfs
            self.model.parameters.A0inv.ch = A0inv
            self.model.residuals.ch = U.T@np.linalg.inv(A0inv)
            self.model.svar.ch = block()

        if method == 'iv':
            self.model.irfs.iv = irfs
            self.model.parameters.A0inv.iv = A0inv
            self.model.residuals.iv = U.T@np.linalg.inv(A0inv)
            self.model.svar.iv = block()
            self.model.svar.iv.idv = idv
            self.model.svar.iv.ins_names = ins_names

        ir, irc, Psi, A0inv = get_irfs(Y, c,B, U, S, nH=nH, impulse=impulse)
        self.model.irfs.rd = block()
        self.model.irfs.rd.ir = ir
        self.model.irfs.rd.irc = irc

    # ==============================================================================================

    def set_sample(self, sample=None):

        var_names = self.model.spec.var_names
        Data = self.Data[var_names].values
        datarownan = np.isnan(Data).any(axis=1)
        if not datarownan.any():
            nanoffset = [0, 0]
        else:
            datarownanchange = np.argwhere(datarownan[1:]!=datarownan[:-1])+1
            if datarownanchange.shape[0] == 1:
                if datarownan[0]:
                    nanoffset = [datarownanchange[0, 0], 0]
                else:
                    nanoffset = [0, Data.shape[0]-datarownanchange[0, 0]]
            elif datarownanchange.shape[0] == 2:
                nanoffset = [datarownanchange[0, 0], Data.shape[0]-datarownanchange[1, 0]]
            elif datarownanchange.shape[0] > 2:
                raise ValueError('Sample should not contain NaNs')

        nL = self.model.spec.nL
        if sample is None:
            isample = (nL+nanoffset[0], self.Data.shape[0]-1-nanoffset[1])
        else:
            try:
                isample = (max(self.Data.index.get_loc(sample[0]), nL+nanoffset[0]), min(self.Data.index.get_loc(sample[1]), self.Data.shape[0]-1-nanoffset[1]))
                sample = (self.Data.index[isample[0]].strftime('%Y-%m-%d'), self.Data.index[isample[1]].strftime('%Y-%m-%d'))
            except KeyError:
                raise KeyError('Provided sample is outside of the available data range')


#         sample = (self.data.index[isample[0]], self.data.index[isample[1]])

        self.model.spec.sample = sample
        self.model.spec.isample = isample
        self.model.spec.nT = isample[1] - isample[0] + 1

        self.fit()

    # ==============================================================================================

    def set_lag_length(self, nL):

        self.model.spec.nL = nL
        self.fit()
        if hasattr(self.model.svar, 'ch'):
            self.irf(method='ch')
        if hasattr(self.model.svar, 'iv'):
            self.irf(method='iv')

    # ==============================================================================================

    def set_horizon(self, nH):

        self.model.irfs.spec.nH = nH
        if hasattr(self.model.svar, 'ch'):
            self.irf(nH=nH, method='ch')
        if hasattr(self.model.svar, 'iv'):
            self.irf(nH=nH, method='iv')

# MM = varm(data, var_names=['0', '1', '2'], nL=4)
# MM.irf(method='iv', ci='bs', idv=2, ins_names='4')
# MM = varm(data, var_names=['0', '1', '3', '2', '4', '5'], nL=4)
# MM.irf(method='iv', ci='wbs', nR=100, idv=[1], ins_names=['6'])
# %load_ext line_profiler
# data.shape
# %lprun -f bs varm(data, var_names=['0', '1', '3', '2', '4', '5'], nL=4).irf(method='iv', ci='wbs', nR=1000, idv=[1], ins_names=['6'])
# print(MM.model.irfs.iv.ir.mean[0])
# print(MM.model.irfs.iv.ir.mean[0])



# print(MM. model.parameters.B[0])
# print(MM. model.parameters.B[0])
# ## LP model
# ### LP
# ### LP-OLS

# %% lpols
def lpols(Ydata, Xdata=None, Zdata=None, Wdata=None, nL=1, nH=0):
    """
    Function to estimate LP(nL, nH) model using OLS
    """

    if Xdata is None: Xdata = Ydata
    if Zdata is None: Zdata = np.full((0, Ydata.shape[1]), np.nan)
    if Wdata is None: Wdata = np.full((0, Ydata.shape[1]), np.nan)

    nY, n1 = Ydata.shape
    nX, _ = Xdata.shape
    nZ, _ = Zdata.shape
    nT = n1 - nL - nH + 1
    nK = nL - 1

    Y = np.vstack([np.roll(Ydata, -h) for h in range(0, nH+1)])
    X = np.vstack([Xdata, Zdata])
    Z = np.vstack([np.ones((1, n1))]+[np.roll(Xdata, l) for l in range(1, nK+1)]+[Wdata])

    X, Y, Z = C(X[:, nK:n1-nH]), C(Y[:, nK:n1-nH]), C(Z[:, nK:n1-nH])

    Mz = np.eye(nT) - Z.T@np.linalg.inv(Z@Z.T)@Z
    B = np.linalg.solve(X@Mz@X.T, X@Mz@Y.T).T
    U = Y@Mz - B@X@Mz
    U = U.reshape((nH+1, nY, nT))
    B = B.T.reshape((nX+nZ, nH+1, nY)).transpose((1, 2, 0)) #.swapaxes(1, 2)
    S = (1/nT)*(U[1]@U[1].T)
    return B, U, S

# %% lpols

@nb.njit
def lpols_njit(Ydata, Xdata=None, Zdata=None, Wdata=None, nL=1, nH=0):
    """
    Function to estimate LP(nL, nH) model using OLS
    """

    if Xdata is None: Xdata = Ydata

    nY, n1 = Ydata.shape
    nX, _ = Xdata.shape
    nZ = 0 if Zdata is None else Zdata.shape[0]
    nT = n1 - nL - nH + 1
    nK = nL - 1

    Y = np.full((0, n1), np.nan)
    for h in range(0, nH+1): Y = np.vstack((Y, np.roll(Ydata, -h)))

    X = np.asarray(Xdata)
    if Zdata is not None: X = np.vstack((X, Zdata))

    Z = np.ones((1, n1))
    for l in range(1, nK+1): Z = np.vstack((Z, np.roll(Xdata, l)))
    if Wdata is not None: Z = np.vstack((Z, Wdata))

    X, Y, Z = C(X[:, nK:n1-nH]), C(Y[:, nK:n1-nH]), C(Z[:, nK:n1-nH])

    Mz = np.eye(nT) - Z.T@np.linalg.inv(Z@Z.T)@Z
    B = np.linalg.solve(X@Mz@X.T, X@Mz@Y.T).T
    U = Y@Mz - B@X@Mz
    U = U.reshape((nH+1, nY, nT))
    B = B.T.reshape((nX+nZ, nH+1, nY)).transpose((1, 2, 0)) #.swapaxes(1, 2)
    S = (1/nT)*(U[1]@U[1].T)
    return B, U, S


# %%

def get_lp_irfs(B, U, S, /, *, method=None, impulse=None, idv=None, M=None):
    _, _, nX = B.shape

#         Psi = get_Psi_from_Bx(B, nH)
    Psi = np.array(B)
#     A0inv = get_A0inv(method=method, U=U, S=S, idv=idv, M=M)
    A0inv = np.eye(nX)
    irm, irmc = get_sirf_from_irf(Psi, A0inv, impulse)
    ir = block()
    irc = block()
    ir.mean = irm
    irc.mean = irmc
    return ir, irc, Psi, A0inv


class lpm:

    # ==============================================================================================

    def __init__(self, Data, /, *, nL=None, nH=None, Y_var_names=None, X_var_names=None, sample=None):

        if Data.shape[0] < Data.shape[1]:
            Data = Data.T

        if isinstance(Data, pd.DataFrame):
            pass
        elif isinstance(Data,(pd.Series, np.ndarray)):
            Data = pd.DataFrame(Data)
            Data.columns = [str(i) for i in Data.columns]

        if Y_var_names is None:
            Y_var_names = Data.columns.tolist()
        if X_var_names is None:
            X_var_names = Data.columns.tolist()

        self.Data = Data
        self.model = block()
        self.model.spec = block()
        self.model.spec.Y_var_names = Y_var_names
        self.model.spec.X_var_names = X_var_names
        self.model.spec.var_names = X_var_names + [name for name in Y_var_names if name not in X_var_names]
        self.model.spec.nL = nL
        self.model.spec.nH = nH
        self.model.spec.nY = len(Y_var_names)
        self.model.spec.nX = len(X_var_names)
        self.model.slp = block()
        self.model.irfs = block()
        self.model.irfs.spec = block()

        self.set_sample(sample)

    # ==============================================================================================

    def fit(self):
        isample = self.model.spec.isample
        nL = self.model.spec.nL
        nH = self.model.spec.nH
        Y_var_names = self.model.spec.Y_var_names
        X_var_names = self.model.spec.X_var_names
        var_names = self.model.spec.var_names
        Y_var_indices = [i for i, name in enumerate(self.Data.columns) if name in Y_var_names]
        X_var_indices = [i for i, name in enumerate(self.Data.columns) if name in X_var_names]
        Data = self.Data[var_names].iloc[isample[0]-nL:isample[1]+nH, :].values

        B, U, S = lpols_njit(Ydata=Data[:, Y_var_indices].T, Xdata=Data[:, X_var_indices].T, nL=nL, nH=nH)

# #         offset = nK

# #         print(n0, n1, nT, nL, nH)
# #         print(Y_var_indices, X_var_indices)
# #         y = np.full((nK+nH+1, nT, nY), np.nan)
# #         Y = np.full((nH+1, nT, nY), np.nan)
# #         X = np.full((nK+1, nT, nY), np.nan)

# #         # Creating y
# #         for idj in range(-nK, nH+1):
# #             y[offset+idj] = data[offset+idj:offset+nT+idj, Y_var_indices]
# #         # Creating Y
# #         for idj in range(1, nH+1):
# #             Y[idj] = data[offset+idj:offset+nT+idj, Y_var_indices] #y[offset+1:offset:idh, :, :]
# #         # Creating X
# #         for idj in range(0, nK):
# #             X[idj] = data[offset-idj:offset+nT-idj, X_var_indices] #y[offset+1:offset:idh, :, :]
# #         # Creating Z
# #         for idj in range(1, nK):
# #             Z[idj] = np.vstack((np.ones((1, n0)), np.roll(data.T, p)))  X[] data[offset-idj:offset+nT-idj, X_var_indices] #y[offset+1:offset:idh, :, :]



#         Y = np.full((0, n0), np.nan)
#         for h in range(1, nH+1):
#             Y = np.vstack((Y, np.roll(data[:, Y_var_indices].T,-h)))

#         X = data[:, X_var_indices].T

#         Z = np.ones((1, n0))
#         for l in range(1, nK+1):
#             Z = np.vstack((Z, np.roll(data[:, X_var_indices].T, l)))

# #         print(Y.shape, X.shape, Z.shape)

#         X = X[:, nK:-nH].T
#         Y = Y[:, nK:-nH].T
#         Z = Z[:, nK:-nH].T
# #         print(Y.shape, X.shape, Z.shape)
# #         print(Y)
#         Mz = np.eye(nT) - Z@np.linalg.inv(Z.T@Z)@Z.T
#         B = np.linalg.inv(X.T@Mz@X)@(X.T@Mz@Y)
# #         print(Y)
# #         print(B)
# #         cB = (Y@Z.T)@(np.linalg.inv(Z@Z.T))
#         U = Mz@Y - Mz@X@B
#         U = U.T.reshape((nH, nX, nT))
#         S = (1/nT)*(U[0]@U[0].T)
#         B = B.reshape((nX, nH, nX)).swapaxes(0, 1) #.swapaxes(1, 2)


        self.model.parameters = block()
        self.model.parameters.B = B
        self.model.parameters.S = S
        self.model.parameters.A0inv = block()
        self.model.residuals = block()
        self.model.residuals.rd = U

        if hasattr(self.model.slp, 'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp, 'iv'):
            self.irf(method='iv')

    # ==============================================================================================

    def irf(self, method=None, impulse=None, idv=None, ins_names=None):

        if impulse is None:
            if hasattr(self.model.irfs.spec, 'impulse'):
                impulse = self.model.irfs.spec.impulse
            else:
                impulse = 'unit'

        self.model.irfs.spec.impulse = impulse
        self.model.irfs.spec.nH = self.model.spec.nH

        nY = self.model.spec.nY
        B = self.model.parameters.B
        S = self.model.parameters.S
        U = self.model.residuals.rd
        isample = self.model.spec.isample

        if isinstance(impulse, list) or isinstance(impulse, np.ndarray):
            impulse = np.diag(impulse)
        if isinstance(impulse, float):
            impulse = impulse*np.eye(nY)

        if method == 'ch':
            idv = None
            ins_names = None
            M = None

        if method == 'iv':
            if idv is None or ins_names is None:
                if hasattr(self.model.slp, 'iv'):
                    if hasattr(self.model.slp.iv, 'idv') and hasattr(self.model.slp.iv, 'ins_names'):
                        idv = self.model.slp.iv.idv
                        ins_names = self.model.slp.iv.ins_names
                    else:
                        raise SyntaxError('Please provide an instrument for SLP-IV identification')
                else:
                    raise SyntaxError('Please provide an instrument for SLP-IV identification')
            else:
                if isinstance(idv, int):
                    idv = [idv]
                if isinstance(ins_names, str):
                    ins_names = [ins_names]
            if len(idv) != len(ins_names):
                raise SyntaxError('The number of instruments must be equal the number of instrumented variables')

            instruments = self.Data[ins_names].iloc[isample[0]:isample[1]+1, :].values
            M = instruments.T

        if method in ('ch', 'iv'):
            ir, irc, Psi, A0inv = get_lp_irfs(B, U, S, method=method, impulse=impulse, idv=idv, M=M)
            irfs = block()
            irfs.ir = ir
            irfs.irc = irc

        if method == 'ch':
            self.model.irfs.ch = irfs
            self.model.parameters.A0inv.ch = A0inv
            self.model.slp.ch = block()

        if method == 'iv':
            self.model.irfs.iv = irfs
            self.model.parameters.A0inv.iv = A0inv
            self.model.slp.iv = block()
            self.model.slp.iv.idv = idv
            self.model.slp.iv.ins_names = ins_names

        ir, irc, Psi, A0inv = get_lp_irfs(B, U, S, impulse=impulse)
        self.model.irfs.rd = block()
        self.model.irfs.rd.ir = ir
        self.model.irfs.rd.irc = irc

    # ==============================================================================================

    def set_sample(self, sample=None):

        var_names = self.model.spec.var_names
        Data = self.Data[var_names].values
        datarownan = np.isnan(Data).any(axis=1)
        if not datarownan.any():
            nanoffset = [0, 0]
        else:
            datarownanchange = np.argwhere(datarownan[1:]!=datarownan[:-1])+1
            if datarownanchange.shape[0] == 1:
                if datarownan[0]:
                    nanoffset = [datarownanchange[0, 0], 0]
                else:
                    nanoffset = [0, Data.shape[0]-datarownanchange[0, 0]]
            elif datarownanchange.shape[0] == 2:
                nanoffset = [datarownanchange[0, 0], Data.shape[0]-datarownanchange[1, 0]]
            elif datarownanchange.shape[0] > 2:
                raise ValueError('Sample should not contain NaNs')

        nL = self.model.spec.nL
        nH = self.model.spec.nH
        if sample is None:
            isample = (nL+nanoffset[0], self.Data.shape[0]-nH-nanoffset[1])
        else:
            isample = (max(self.Data.index.get_loc(sample[0]), nL+nanoffset[0]), min(self.Data.index.get_loc(sample[1]), self.Data.shape[0]-nH-nanoffset[1]))
            sample = (self.Data.index[isample[0]].strftime('%Y-%m-%d'), self.Data.index[isample[1]].strftime('%Y-%m-%d'))


#         sample = (self.data.index[isample[0]], self.data.index[isample[1]])

        self.model.spec.sample = sample
        self.model.spec.isample = isample
        self.model.spec.nT = isample[1]-isample[0]+1

        self.fit()

    # ==============================================================================================

    def set_lag_length(self, nL):

        self.model.spec.nL = nL
        self.fit()
        if hasattr(self.model.slp, 'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp, 'iv'):
            self.irf(method='iv')

    # ==============================================================================================

    def set_horizon(self, nH):

        self.model.spec.nH = nH
        self.fit()
        if hasattr(self.model.slp, 'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp, 'iv'):
            self.irf(method='iv')

# ## SFM model
# ### SFM

class sfm:
    def __init__(self, Data, tcodes, nF=None, make_fig=False):

        if isinstance(Data, pd.DataFrame):
            pass
        elif isinstance(Data,(pd.Series, np.ndarray)):
            Data = pd.DataFrame(Data)

        Data = Data.values

        self.Data = Data
        self.tcodes = tcodes

        Datatrans = self.transform_by_tcode(Data, tcodes)
        DataTrans = pd.DataFrame(Datatrans, columns=Data.columns, index=Data.index)
        X = Datatrans[2:, :]

        Xcolnan = np.isnan(X).any(axis=0)
        X_nonan = X[:, ~Xcolnan]

        X_S, X_Mu, X_StD = self.standardize(X_nonan)
        fac, lam = self.getFactors(X_S, nF, make_fig)

        Factors = pd.DataFrame(fac, columns=[('Factor_'+str(f)) for f in range(1, fac.shape[1]+1)], index=Data.index[2:])
        Lambda = pd.DataFrame(lam, columns=Data.columns[~Xcolnan], index=range(fac.shape[1]))

        self.DataTrans = DataTrans
        self.X = X_nonan
        self.X_S = X_S
        self.X_Mu = X_Mu
        self.X_StD = X_StD
        self.Factors = Factors
        self.Lambda = Lambda
        self.F = fac
        self.L = lam
        self.nT = Data.shape[0]
        self.nN = Data.shape[1]
        self.nF = fac.shape[1]

    def transform_by_tcode(self, rawdata, tcodes):
        def transxf(xin, tcode):
            x = xin.copy()
            y = np.full_like(x, np.nan)
            if tcode == 1:
                y = x
            elif tcode == 2:
                y[1:] = x[1:]-x[:-1]
            elif tcode == 3:
                y[2:] = (x[2:]-x[1:-1])-(x[1:-1]-x[:-2])
            elif tcode == 4:
                y = np.log(x)
            elif tcode == 5:
                y[1:] = np.log(x[1:])-np.log(x[:-1])
            elif tcode == 6:
                y[2:] = (np.log(x[2:])-np.log(x[1:-1]))-(np.log(x[1:-1])-np.log(x[:-2]))
            elif tcode == 7:
                y[1:] = (x[1:]-x[:-1])/x[:-1]
            return y
        transformed_data = np.full_like(rawdata, np.nan)
        for (idv, x), tcode in zip(enumerate(rawdata.T), tcodes):
            transformed_data[:, idv] = transxf(x, tcode)
        return transformed_data

    def standardize(self, X):
        X_Mu = np.mean(X, axis=0)
        X_StD = np.std(X, axis=0, ddof=1)
        X_S = (X-X_Mu)/X_StD
        return X_S, X_Mu, X_StD

    def getFactors(self, X, nF=None, make_fig=False):
        nT, nN = X.shape
        S_xx = (1/nT)*(X.T@X)
        eigVal, eigVec = np.linalg.eigh(S_xx)
        # sort eigenvalues in descending order
        idx = np.argsort(eigVal)[::-1]
        eigVal = eigVal[idx]
        eigVec = eigVec[:, idx]

        if nF is None:
            r_max = int(np.floor(np.sqrt(nN)))
            V = np.full(r_max, np.nan)  # sum of squared residuals
            IC = np.full(r_max, np.nan) # information criterion

            for r in range(1, r_max+1):
                lam = np.sqrt(nN)*eigVec[:, :r].T
                fac = (1/nN)*(X@lam.T)
                V[r-1] = (1/(nN*nT))*np.trace((X-fac@lam).T@(X-fac@lam))
                IC[r-1] = np.log(V[r-1]) + r*(nN+nT)/(nN*nT)*np.log(min(nN, nT))

            if make_fig:
                _, axes = mpl.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(range(1, r_max+1), IC, '-o')
                axes[0].set_xticks(range(1, r_max+1))
                axes[0].set_xlim((0, r_max+1))
                axes[0].set_title('Bai & Ng Criterion')
                axes[1].plot(range(1, r_max+1), eigVal[:r_max], '-o')
                axes[1].set_xticks(range(1, r_max+1))
                axes[1].set_xlim((0, r_max+1))
                axes[1].set_title('Eigenvalues')
            nFF = np.argmin(IC)+1
            print(f'Number of factors selected by Bai & Ng criterion is {nFF}')
        else:
            nFF = nF
        lam = np.sqrt(nN)*eigVec[:, :nFF].T
        fac = (1/nN)*(X@lam.T)
        return fac, lam


# %% VAR model forecast evaluation

class FE_VARm:

    def __init__(self, df, /, *, var_names, nL, nF, sample=None, period=None, in_sample=False):

        freq = df.index.freq

        Mdl_full = VARm(df, nL=nL, var_names=var_names, sample=sample)
        nY, sample = Mdl_full.Spec['nY'], Mdl_full.Spec['sample']

        df_F = pd.DataFrame(df[var_names]).reindex(pd.date_range(start=df.index[0], periods=df.shape[0]+nF, freq=freq))

        if period is None:
            period = sample if in_sample else [df.index[df.index.get_loc(sample[0])+(df.index.get_loc(sample[1])-df.index.get_loc(sample[0]))//2].strftime('%Y-%m-%d'), sample[1]]

        if in_sample:
            Mdl = Mdl_full

        for t in df.loc[period[0]:period[1]].index:
            if not in_sample:
                Mdl = Mdl_full.change_sample(sample=(sample[0], t))
            Mdl = Mdl.forecast(nF=nF, period=t)
            Forecast = Mdl.Forecasts.Fcs_m
            df_f = pd.DataFrame(Forecast.T if Mdl.Est['Stable'] else np.full_like(Forecast.T, np.nan), index=pd.date_range(t, periods=nF+1, freq=freq), columns=['_f_'+var_name+'_'+t.strftime('%Y%m%d') for var_name in var_names])
            df_F = df_F.join(df_f, how='outer')

        df_E = df_F[[c for c in df_F.columns if c.startswith('_f_')]]               # take forecasts
        df_E = df_E.apply(lambda x: df_F[x.name[3:len(x.name)-9]]-x)                # calculate forecast errors
        df_E = df_E.loc[period[0]:period[1]]                                        # take only necessary values

        E = df_E.values                                                             # organize forecast errors
        row_roll = np.arange(nY*(E.shape[0]-1), -nY, -nY)
        row_roll = np.arange(nY*(E.shape[0]-1), -nY, -nY)
        rows, columns = np.ogrid[:E.shape[0], :E.shape[1]]
        columns_new = columns - row_roll[:, np.newaxis]
        E = E[rows, columns_new]

        column_roll = np.repeat(list(range(E.shape[0]-1, -1, -1)), nY)
        rows, columns = np.ogrid[:E.shape[0], :E.shape[1]]
        rows_new = rows + column_roll[np.newaxis, :]
        rows_new[rows_new>=E.shape[0]] -= E.shape[0]
        E = E[rows_new, columns]

        E = np.hstack((np.fliplr(E)[:, :nY*(nF+1)], np.full((E.shape[0], max(0, nY*(nF+1)-df_E.shape[1])), np.nan)))

        df_E = pd.DataFrame(E, index=df_E.index, columns=['_e_'+var_name+f'_f{iF}' for iF in range(0, nF+1) for var_name in var_names[::-1]])
        df_E = df_E[['_e_'+var_name+f'_f{iF}' for iF in range(0, nF+1) for var_name in var_names]]

        RMSFE = pd.DataFrame([df_E.map(np.square).apply(np.mean).apply(np.sqrt)[[c for c in df_E.columns if varname in c]].values for varname in var_names], index=var_names, columns=range(0, nF+1))

        STD = df.loc[period[0]:period[1], var_names].std()

        self.var_names = var_names
        self.sample = sample
        self.period = period
        self.nL = nL
        self.nF = nF
        self.df_F = df_F
        self.df_E = df_E
        self.RMSFE = RMSFE
        self.STD = STD

# %%
