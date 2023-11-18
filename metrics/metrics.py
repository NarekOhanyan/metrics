# %% Load packages
import numpy as np
import numba as nb
import pandas as pd
import scipy.stats as spstats
import matplotlib.pyplot as mpl

use_numba = False

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

    def __init__(self, Irf_m, Irfc_m, Irf_Sim=None, Irfc_Sim=None, irf_spec=None):
        self.Irf_m, self.Irfc_m = Irf_m, Irfc_m
        self.__Irf_Sim, self.__Irfc_Sim = Irf_Sim, Irfc_Sim
        self.Spec = irf_spec

    def Irf_q(self, q):
        Irf_Sim = self.__Irf_Sim
        return np.quantile(Irf_Sim, q, axis=0)

    def Irfc_q(self, q):
        Irfc_Sim = self.__Irfc_Sim
        return np.quantile(Irfc_Sim, q, axis=0)

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
        add_constant (bool,optional): Add a constant to Xdata. Defaults to True

    Returns:
        (nY, nN): Ydata without NaNs
        (nX, nN): Xdata without NaNs
        (nC+nX, n1): Xdata with constant if requested
    """
    Ydata = Ydata.reshape((-1,Ydata.shape[1]))
    nY, _ = Ydata.shape
    nC = 1 if add_constant else 0
    cXdata = np.row_stack((np.ones((nC,Xdata.shape[1])),Xdata))
    YXdata = np.row_stack((Ydata,cXdata))
    YXdata = YXdata[:,~np.isnan(YXdata).any(axis=0)]
    return YXdata[:nY], YXdata[nY:], cXdata

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
    if use_numba:
        # invXx = np.repeat(invX[np.newaxis, :, :], nY, axis=0)
        invXx = np.repeat(invX, nY).reshape((nX*nX,nY)).T.reshape((nY,nX,nX))
    else:
        invXx = np.tile(invX, (nY, 1, 1))
    V = np.diag(S).reshape((-1, 1, 1))*invXx
    # Se = np.array([np.sqrt(np.diag(V[iY])) for iY in range(nY)])
    Se = np.sqrt(np.outer(np.diag(S), np.diag(invX)))
    return B, Se, V, E, S

ols_b_h_njit = nb.njit(ols_b_h) # doesn't work due to https://github.com/numba/numba/issues/4580

# %%
def split_C_B(B, nC, nL, nY):
    Bc, Bx = np.split(B, np.array([nC]), axis=1)
    Bx = Bx.T.reshape((nL, nY, nY)).transpose((0, 2, 1))
    return Bc, Bx

# %% VAR-OLS
def fit_var_h(Y, nC, nL, dfc=True):
    """
    Function to estimate VAR(P) model with P = nL using OLS
    """
    nY, n1 = Y.shape
    X = np.ones((nC, n1))
    for p in range(1, nL+1):
        X = np.row_stack((X, np.roll(Y, p)))
    Y, X = Y[:, nL:], X[:, nL:]
    if not use_numba:
        B, Se, V, U, S = ols_b_h(Y, X, dfc)
    else:
        B, Se, V, U, S = ols_b_h_njit(Y, X, dfc)

    return B, Se, V, U, S

fit_var_h_njit = nb.njit(fit_var_h)

# %% OLS with one dependent variable y
def OLS_h(Ydata, Xdata, add_constant=True, dfc=True):
    Y, X, cXdata = check_data(Ydata,Xdata,add_constant)
    b, se, V, _, S = ols_h(Y, X, dfc)
    e = Ydata-b@cXdata
    return b, se, V, e, S

# %% OLS with many dependent variables y
def OLS_b_h(Ydata, Xdata, add_constant=True, dfc=True):
    assert ~np.isnan(Ydata).any() and ~np.isnan(Xdata).any(), 'data should not contain NaNs'
    Y, X, _ = check_data(Ydata,Xdata,add_constant)
    B, Se, V, E, S = ols_b_h(Y, X, dfc)
    return B, Se, V, E, S

# %%
class ardlm_irfs:

    def __init__(self, irf, std, irfc, stdc):
        self.Irf, self.Std, self.Irfc, self.Stdc = irf, std, irfc, stdc

    def Irf_q(self,q):
        return self.Irf + spstats.norm.ppf(q)*self.Std

    def Irfc_q(self,q):
        return self.Irfc + spstats.norm.ppf(q)*self.Stdc

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
        self.irf(nH,nR)

    def fit(self):

        nC, nLy, nLx = self.Spec.nC, self.Spec.nLy, self.Spec.nLx
        Ydata, Xdata, Zdata = self.Data.Ydata, self.Data.Xdata, self.Data.Zdata

        y, X, Z = Ydata.values.T, Xdata.values.T, Zdata.values.T

        n1 = y.shape[1]
        nL = max(nLy,nLx)
        nT = n1 - nL

        Y = y.reshape(1,-1)

        W = np.ones((nC,n1))
        W = np.row_stack((W,Z))
        for l in range(1,nLy+1):
            W = np.row_stack((W,np.roll(y,l, axis=1)))
        for l in range(1,nLx+1):
            W = np.row_stack((W,np.roll(X,l, axis=1)))

        (nY,_), (nX,_), (nZ,_), (nW,_) = Y.shape, X.shape, Z.shape, W.shape

        Y, W = Y[:,nL:], W[:,nL:]

        B, Se, V, E, S = ols_h(Y, W)

        B, Se = np.squeeze(B), np.squeeze(Se)

        Bc, Bz, By, Bx = np.split(B,np.cumsum([nC,nZ,nLy*nY]))
        SEc, SEz, SEy, SEx = np.split(Se,np.cumsum([nC,nZ,nLy*nY]))

        Bx = np.squeeze(Bx.reshape((nLx,nX)).T)

        self.Est.B, self.Est.Se, self.Est.V, self.Est.E, self.Est.S = B, Se, V, E, S
        self.Est.Bc, self.Est.Bz, self.Est.By, self.Est.Bx = Bc, Bz, By, Bx
        self.Est.SEc, self.Est.SEz, self.Est.SEy, self.Est.SEx = SEc, SEz, SEy, SEx
        self.Spec.nY, self.Spec.nX, self.Spec.nZ, self.Spec.nW, self.Spec.nT = nY, nX, nZ, nW, nT

        return self

    def irf(self,nH,nR):

        B, By, Bx, V = self.Est.B, self.Est.By, self.Est.Bx, self.Est.V
        nLy, nLx = self.Spec.nLy, self.Spec.nLx
        nC, nY, nX, nZ = self.Spec.nC, self.Spec.nY, self.Spec.nX, self.Spec.nZ

        Irf, Irfc = self.get_irf(By, Bx, nLy, nLx, nY, nX, nH)
        Irf_std, Irfc_std = self.get_irf_std(B, V, nLy, nLx, nC, nY, nX, nZ, nH, nR)

        self.Irfs = ardlm_irfs(Irf, Irf_std, Irfc, Irfc_std)
        self.Irfs.nH, self.Irfs.nR = nH, nR

        return self

    def get_irf(self, By, Bx, nLy, nLx, nY, nX, nH):

        Irf = np.full((nX,nH+1),np.nan)

        By = np.pad(By.reshape((nY,nLy)),((0,0),(0,nH-nLy+1)))
        Bx = np.pad(Bx.reshape((nX,nLx)),((0,0),(0,nH-nLx+1)))

        Bx = np.column_stack((np.zeros((nX,1)),Bx))
        for h in range(nH+1):
            for iX in range(nX):
                Irf[iX,h] = Irf[iX,:h][::-1]@By[0,:h] + Bx[iX,h]

        return Irf, np.cumsum(Irf,axis=1)

    def get_irf_std(self, B, V, nLy, nLx, nC, nY, nX, nZ, nH, nR):

        Irf_R = np.full((nR,nX,nH+1),np.nan)
        Irfc_R = np.full((nR,nX,nH+1),np.nan)

        B_R = np.random.multivariate_normal(B,V,size=nR)
        for r in range(nR):
            B_r = B_R[r].reshape((1,-1))
            _, _, By_r, Bx_r = np.split(B_r,np.cumsum([nC,nZ,nLy*nY]),axis=1)
            irf_r, cirf_r = self.get_irf(By_r, Bx_r, nLy, nLx, nY, nX, nH)
            Irf_R[r], Irfc_R[r] = irf_r, cirf_r

        Irf_Std = Irf_R.std(axis=0)
        Irfc_Std = Irfc_R.std(axis=0)

        return Irf_Std, Irfc_Std

    def do_data(self, Ydata, Xdata, Zdata):

        def make_df(data):
            if isinstance(data,np.ndarray) or isinstance(data,pd.Series):
                data = pd.DataFrame(data)
            assert data.shape[0] > data.shape[1], 'data must be in column format'
            return data

        if Zdata is None:
            Zdata = np.full((Ydata.shape[0],0),np.nan)

        Ydata, Xdata, Zdata = make_df(Ydata), make_df(Xdata), make_df(Zdata)

        assert Ydata.shape[1] == 1, 'Ydata must contain only one column'
        assert Ydata.shape[0] == Xdata.shape[0] == Zdata.shape[0], 'data must have the same length'

        return Ydata, Xdata, Zdata

# %% Nelson-Siegel model
class nsm:

    def __init__(self,yields,tau,lam,classic=True):

        if len(yields.shape) == 1:
            yields = yields[None,:]

        if yields.shape[1] != tau.shape[0]:
            raise SyntaxError('yields and tau must have the same length')

        self.yields = pd.DataFrame(yields)
        self.yields.columns = tau
        self.tau = tau
        self.lam = lam
        self.classic = classic
        self.fit()

    def getLoadings(self,tau,lam):
        if self.classic:
            b1l = np.ones_like(tau)
            b2l = np.array((1-np.exp(-lam*tau))/(lam*tau))
            b3l = np.array((1-np.exp(-lam*tau))/(lam*tau)-np.exp(-lam*tau))
        else:
            b1l = np.array((1-np.exp(-lam*tau))/(lam*tau))
            b2l = np.array((1-np.exp(-lam*tau))/(lam*tau)-np.exp(-lam*tau))
            b3l = np.ones_like(tau)-np.array((1-np.exp(-lam*tau))/(lam*tau))
        return np.column_stack((b1l,b2l,b3l))

    def olsproj(self,yin,Xin):
        y = yin.copy()
        X = Xin.copy()
        if len(y.shape) == 1:
            y = y[:,None]
        yX = np.column_stack((y,X))
        yXnan = np.isnan(yX).any(axis=1)
        y[yXnan,:],X[yXnan,:] = 0,0
        b = np.linalg.solve(X.T@X,X.T@y)
        return b

    def fit(self):
        yields = self.yields.values
        tau = self.tau
        lam = self.lam
        X = self.getLoadings(tau,lam)

        betasT = np.full((3,yields.shape[0]),np.nan)
        for t,yld in enumerate(yields):
            betasT[:,t,None] = self.olsproj(yld.T,X)

        self.X = X
        self.betas = pd.DataFrame(betasT.T,index=self.yields.index,columns=['beta1','beta2','beta3'])
        self.predict(np.array(range(1,tau[-1]+1)))

    def predict(self,ptau=None):
        if ptau is None:
            ptau = self.tau
        lam = self.lam
        betas = self.betas.values
        X = self.getLoadings(ptau,lam)

        self.curve = pd.DataFrame(betas@X.T,index=self.yields.index,columns=ptau)
        self.ptau = ptau

    def plot(self,index):
        tau = self.tau
        ptau = self.ptau
        mpl.scatter(tau,self.yields.loc[index].values)
        mpl.plot(ptau,self.curve.loc[index].values)

# %% VAR model
class varms:

    def __init__(self,data,nP):
        if data.shape[0] > data.shape[1]:
            data = data.T
        (n0,n1) = data.shape
        self.data = data
        self.nP = nP
        self.n0 = n0
        self.n1 = n1
        self.nT = n1 - nP
        self.nK = n0
        self.model.irfs = block()

    def fit(self):
        data = self.data
        (n0,n1) = data.shape
        nP = self.nP
        nK = self.nK
        nT = n1 - nP
        Z = np.ones((1,n1))

        for p in range(1,1+nP):
            Z = np.row_stack((Z,np.roll(data,p)))

        Z = Z[:,nP:]
        Y = data[:,nP:]

        cB = (Y@Z.T)@(np.linalg.inv(Z@Z.T))
        c = cB[:,0]
        B = cB[:,1:].T.reshape((nP,nK,nK)).swapaxes(1,2)
        U = Y-cB@Z
        S = (1/(nT-nP*nK-1))*(U@U.T)

        self.parameters = block()
        self.parameters.c = c
        self.parameters.B = B
        self.parameters.S = S
        self.residuals = block()
        self.residuals.rd = U

    def irf(self,nH,method='cholesky',idv=None,ins_names=None):

        self.nH = nH
        nT = self.nT
        nP = self.nP
        nK = self.nK
        B = self.parameters.B
        S = self.parameters.S
        U = self.residuals.rd

        Psi = np.zeros((nH,nK,nK))
        Psi[0] = np.eye(nK)
        for h in range(1,nH):
            for i in range(min(h,nP)):
                Psi[h] += Psi[h-i-1]@B[i]

        self.model.irfs.rd = Psi
        self.model.irfs.rdc = np.cumsum(Psi,0)
        self.parameters.A0inv = block()

        if method == 'cholesky':
            A0inv = np.linalg.cholesky(S)
            self.model.irfs.ch = Psi@A0inv
            self.model.irfs.chc = np.cumsum(Psi@A0inv,0)
            self.parameters.A0inv.ch = A0inv
            self.residuals.ch = np.linalg.inv(A0inv)@U
        if method == 'iv':
            if idv is None or ins_names is None:
                raise SyntaxError('Please provide an instrument for SVAR-IV identification')
            instrument = self.data[ins_names]
            A0inv = np.zeros((nK,nK))
            for v,ins in zip(idv,instrument.T):
#                 print(np.cov(U,ins[-nT:].T))
                A0inv[:,v] = np.cov(np.row_stack((ins[-nT:],U)))[0,1:]
                A0inv[:,v] = A0inv[:,v]/A0inv[v,v]
            self.model.irfs.iv = Psi@A0inv
            self.model.irfs.ivc = np.cumsum(Psi@A0inv,0)
            self.parameters.A0inv.iv = A0inv
            self.iv = block()
            self.iv.idv = idv
            self.iv.ins_names = ins_names

# %% VAR-OLS
def varols(data, nL):
    """
    Function to estimate VAR(P) model with P = nL using OLS
    """
    (n0, n1) = data.shape
    nT = n1 - nL
    nY = n0
    Z = np.ones((1, n1))

    for p in range(1, nL+1):
        Z = np.row_stack((Z, np.roll(data, p)))

    Z = C(Z[:, nL:])
    Y = C(data[:, nL:])

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
    Y[:,:nL] = Y0

    for t in range(nL, nL+nT):
        # The methods (a) and (b) are equivalent
        ## (a)
        BB = C(B.transpose((0, 2, 1))).reshape((nL*nY, nY)).T
        Y[:,t] = c + (BB@C(Y[:, t-nL:t][:, ::-1].T).reshape((nL*nY, 1))).reshape((-1, )) + U[:, t-nL]
        ## (b)
        # Y_t = c + U[:,t-nL]
        # for l in range(nL):
        #     Y_t += B[l]@Y[:,t-l-1]
        # Y[:,t] = Y_t
    return Y

varsim_njit = nb.njit(varsim)

# %% simulate VAR
def sim_var(Y0, Bc, Bx, U):
    Y0, Bc, Bx, U = C(Y0), C(Bc), C(Bx), C(U)
    nY, nL = Y0.shape
    nY, nT = U.shape
    Y = np.full(((nY, nL+nT)), np.nan)
    Y[:,:nL] = Y0
    BX = C(Bx.transpose((0, 2, 1))).reshape((nL*nY, nY)).T
    for t in range(nL, nL+nT):
        # The methods (a) and (b) are equivalent
        ## (a)
        Y[:,t] = Bc.reshape(-1) + BX@C(Y[:, t-nL:t][:, ::-1].T).reshape(-1) + U[:, t-nL]
        ## (b)
        # Y_t = c + U[:,t-nL]
        # for l in range(nL):
        #     Y_t += B[l]@Y[:, t-l-1]
        # Y[:,t] = Y_t
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
        Yy[:, :, t] = np.repeat(Bc[np.newaxis,:], nS, axis=0) + (BX@C(Yy[:, :, t-nL:t][:, :, ::-1].T).reshape((nS, nL*nY)).T).T + U[:, :, t-nL]
        ## (b)
        # for s in range(nS):
        #     Y_t = c + U[:,t-nL]
        #     for l in range(nL):
        #         Y_t += B[l]@Yy[:, :, t-l-1]
        #     Yy[s, :, t] = Y_t
    return Yy

sim_var_b_njit = nb.njit(sim_var_b)

# %% get Psi from B
def get_Psi_from_B(B, nH):
    (nL, nY, _) = B.shape
    Psi = np.zeros((nH+1, nY, nY))
    Psi[0] = np.eye(nY)
    for h in range(1, nH+1):
        for i in range(min(h, nL)):
            Psi[h] += C(Psi[h-i-1])@C(B[i])
    return Psi

get_Psi_from_B_njit = nb.njit(get_Psi_from_B)

# %% Get A0inv
def get_A0inv(method=None,U=None,S=None,idv=None,M=None):
    (nY,nT) = U.shape
    if method == None:
        A0inv = np.eye(nY)
    if method == 'ch':
        A0inv = np.linalg.cholesky(S)
    if method == 'iv':
        method_ = 1
        A0inv = np.sqrt(np.diag(np.diag(S)))
#         A0inv = np.zeros((nY,nY))
        if method_ == 0:
            for v,m in zip(idv,M):
                mU = np.row_stack((m,U))
                mU_nan = np.isnan(mU)
                mU = mU[:,~mU_nan.any(axis=0)]
                if mU.shape[1] < 10:
                    raise ValueError('Not enough observations to perform SVAR-IV identification')
                centered = False
                if centered:
                    S_mU = np.cov(mU)
                else:
                    S_mU = (1/mU.shape[1])*(mU@mU.T)
                method__ = 'regression'

                if method__ == 'regression':
                    X = np.row_stack((np.ones((1,mU.shape[1])),mU[0:1,:]))
                    Y = mU[1:,:]
                    beta1 = np.linalg.solve(X@X.T,X@Y.T)[1,:]

                if method__ == 'moments':
                    beta1 = S_mU[1:,0]

                # normalize
                beta1 = beta1[:]/beta1[v]
    #             A0inv[:,v] = (insUcov[1:,0]/insUstd[0]).T # st. dev. of explained part
    #             A0inv[:,v] = (insUcov[1:,0]/insUcov[v+1,0]).T # st. dev. of residual
    #             A0inv[:,v] = (insUcov[1:,0]/(insUcov[v+1,0]/insUstd[v+1])).T # st. dev. of residual
    #             A0inv[:,v] = A0inv[:,v]/A0inv[v,v] # unit
                A0inv[:,v] = beta1.T

        if method_ == 1:
            nM = M.shape[0]
            not_idv = np.array([_ for _ in range(nY) if _ not in idv])

            # Reorder instrumented residuals first
            U_ = np.row_stack((U[idv,:],U[not_idv,:]))
            MU = np.row_stack((M,U_))
            # Remove time periods with nans
            MU_nan = np.isnan(MU)
            MU = MU[:,~MU_nan.any(axis=0)]
            if MU.shape[1] < 10:
                raise ValueError('Not enough observations to perform SVAR-IV identification')

            b11,b21 = iv_block_njit(MU,nM)

            idv_array = np.array(idv)
            not_idv_array = np.array(not_idv)

            A0inv[idv_array[:,None],idv_array] = b11
            A0inv[not_idv_array[:,None],idv_array] = b21

    return A0inv

# %% get A0inv
@nb.njit # not used
def get_A0inv_njit(method=None,U=None,S=None,idv=None,M=None):
    (nY,nT) = U.shape
    if method == None:
        A0inv = np.eye(nY)
    if method == 'ch':
        A0inv = np.linalg.cholesky(S)
    if method == 'iv':
        method_ = 1
        A0inv = np.sqrt(np.diag(np.diag(S)))
#         A0inv = np.zeros((nY,nY))
        if method_ == 0:
            for v,m in zip(idv,M):
                mU = np.row_stack((m,U))
                mU_nan = np.isnan(mU)
                mU = mU[:,~mU_nan.any(axis=0)]
                if mU.shape[1] < 10:
                    raise ValueError('Not enough observations to perform SVAR-IV identification')

                centered = False
                if centered:
                    S_mU = np.cov(mU)
                else:
                    S_mU = (1/mU.shape[1])*(mU@mU.T)

                method__ = 'regression'
                if method__ == 'regression':
                    X = np.row_stack((np.ones((1,mU.shape[1])),mU[0:1,:]))
                    Y = mU[1:,:]
                    beta1 = np.linalg.solve(X@X.T,X@Y.T)[1,:]
                if method__ == 'moments':
                    beta1 = S_mU[1:,0]

                # normalize
                beta1 = beta1[:]/beta1[v]
    #             A0inv[:,v] = (insUcov[1:,0]/insUstd[0]).T # st. dev. of explained part
    #             A0inv[:,v] = (insUcov[1:,0]/insUcov[v+1,0]).T # st. dev. of residual
    #             A0inv[:,v] = (insUcov[1:,0]/(insUcov[v+1,0]/insUstd[v+1])).T # st. dev. of residual
    #             A0inv[:,v] = A0inv[:,v]/A0inv[v,v] # unit
                A0inv[:,v] = beta1.T

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
                        inn = False
                        break
                if not in_idv:
                    not_idv.append(_)

            # Reorder instrumented residuals first
            U_ = np.full_like(U,np.nan)
            for (iU_,iU) in enumerate([_ for _ in idv]+[_ for _ in not_idv]):
                U_[iU_,:] = U[iU,:]
            MU = np.row_stack((M,U_))
            # Remove time periods with nans
            MU_nan = np.isnan(MU)
            MU = MU[:,~MU_nan.any(axis=0)]
            if MU.shape[1] < 10:
                raise ValueError('Not enough observations to perform SVAR-IV identification')

            b11,b21 = iv_block_njit(MU,nM)

            for (ib,iA) in enumerate(idv):
                for (jb,jA) in enumerate(idv):
                    A0inv[iA,jA] = b11[ib,jb]
            # A0inv[idv[:,None],idv] = b11
#                 print(A0inv[idv,:][:,idv])
            for (ib,iA) in enumerate(not_idv):
                for (jb,jA) in enumerate(idv):
                    A0inv[iA,jA] = b21[ib,jb]
            # A0inv[not_idv[:,None],idv] = b21
#                 print(A0inv)
    return A0inv

# ### IV-identification

def iv_block(MU,nM):
    # The formulas from Mertens & Ravn (2013) Appendix A
    S_mumu = (1/MU.shape[1])*(MU@MU.T)
    S_uu = S_mumu[nM:,nM:]
    S_mu = S_mumu[:nM,nM:]
    S_mu1 = S_mu[:,:nM]
    S_mu2 = S_mu[:,nM:]
    S11 = S_uu[:nM,:nM]
    S21 = S_uu[nM:,:nM]
    S22 = S_uu[nM:,nM:]
    b21_b11_1 = (np.linalg.inv(S_mu1)@S_mu2).T
#     b21_b11_1 = np.linalg.solve(S_mu1,S_mu2).T
    Z = b21_b11_1@S11@b21_b11_1.T-(S21@b21_b11_1.T+b21_b11_1@S21.T)+S22
    b12_b12_T = (S21-b21_b11_1@S11).T@np.linalg.inv(Z)@(S21-b21_b11_1@S11)
#     b12_b12_T = (S21-b21_b11_1@S11).T@np.linalg.solve(Z,S21-b21_b11_1@S11)
    b22_b22_T = S22+b21_b11_1@(b12_b12_T-S11)@b21_b11_1.T
    b12_b22_1 = (b12_b12_T@b21_b11_1.T+(S21-b21_b11_1@S11).T)@b22_b22_T
    b11_b11_T = S11-b12_b12_T
    S1_S1_T = (np.eye(nM)-b12_b22_1@b21_b11_1)@b11_b11_T@(np.eye(nM)-b12_b22_1@b21_b11_1).T
    S1 = np.linalg.cholesky(S1_S1_T)
    b11_S1_1 = np.linalg.inv(np.eye(nM)-b12_b22_1@b21_b11_1)
#     b11_S1_1 = np.linalg.solve(np.eye(nM)-b21_b11_1.T@b12_b22_1.T,np.eye(nM)).T
    b21_S1_1 = b21_b11_1@np.linalg.inv(np.eye(nM)-b12_b22_1@b21_b11_1)
#     b21_S1_1 = np.linalg.solve(np.eye(nM)-b21_b11_1.T@b12_b22_1.T,b21_b11_1.T).T
    b11 = b11_S1_1@S1
    b21 = b21_S1_1@S1
    return b11,b21


@nb.njit
def iv_block_njit(MU,nM):
    # The formulas from Mertens & Ravn (2013) Appendix A
    MU = np.ascontiguousarray(MU)
    MU_T = np.ascontiguousarray(MU.T)
    S_mumu = (1/MU.shape[1])*(MU@MU_T)
    S_uu = S_mumu[nM:,nM:]
    S_mu = S_mumu[:nM,nM:]
    S_mu1 = np.ascontiguousarray(S_mu[:,:nM])
    S_mu2 = np.ascontiguousarray(S_mu[:,nM:])
    S11 = np.ascontiguousarray(S_uu[:nM,:nM])
    S21 = np.ascontiguousarray(S_uu[nM:,:nM])
    S22 = np.ascontiguousarray(S_uu[nM:,nM:])
#     b21_b11_1 = (np.linalg.inv(S_mu1)@S_mu2).T
    b21_b11_1 = np.linalg.solve(S_mu1,S_mu2).T
    Z = b21_b11_1@S11@b21_b11_1.T-(S21@b21_b11_1.T+b21_b11_1@S21.T)+S22
#     b12_b12_T = (S21-b21_b11_1@S11).T@np.linalg.inv(Z)@(S21-b21_b11_1@S11)
    b12_b12_T = (S21-b21_b11_1@S11).T@np.linalg.solve(Z,S21-b21_b11_1@S11)
    b22_b22_T = S22+b21_b11_1@(b12_b12_T-S11)@b21_b11_1.T
    b12_b22_1 = (b12_b12_T@b21_b11_1.T+(S21-b21_b11_1@S11).T)@b22_b22_T
    b11_b11_T = S11-b12_b12_T
    S1_S1_T = (np.eye(nM)-b12_b22_1@b21_b11_1)@b11_b11_T@(np.eye(nM)-b12_b22_1@b21_b11_1).T
    S1 = np.linalg.cholesky(S1_S1_T)
#     b11_S1_1 = np.linalg.inv(np.eye(nM)-b12_b22_1@b21_b11_1)
    b11_S1_1 = np.linalg.solve(np.eye(nM)-b21_b11_1.T@b12_b22_1.T,np.eye(nM)).T
#     b21_S1_1 = b21_b11_1@np.linalg.inv(np.eye(nM)-b12_b22_1@b21_b11_1)
    b21_S1_1 = np.linalg.solve(np.eye(nM)-b21_b11_1.T@b12_b22_1.T,b21_b11_1.T).T
    b11 = b11_S1_1@S1
    b21 = b21_S1_1@S1
    return b11,b21

# %% get SIRF from IRF
def get_sirf_from_irf(Psi, A0inv, impulse='unit'):
    if impulse == 'unit':
        impulse_scale = np.diag(1/np.diag(A0inv))
    elif impulse == '1sd':
        impulse_scale = np.eye(A0inv.shape[0])
    else:
        impulse_scale = impulse*np.diag(1/np.diag(A0inv))
    Impact = A0inv@impulse_scale
    Irf = Psi@Impact
    Irfc = np.cumsum(Irf, 0)
    return Irf, Irfc

# get_sirf_from_irf_njit = nb.njit(get_sirf_from_irf)

# %% Bootstrap
def bs(Y, U, B, /, *, model_spec, irf_spec):
    nC, nL, nY, nT, dfc = model_spec['nC'], model_spec['nL'], model_spec['nY'], model_spec['nT'], model_spec['dfc']
    nH, ci = irf_spec['nH'], irf_spec['ci']
    Y0 = Y[:, :nL]
    Bc, Bx = split_C_B(B, nC, nL, nY)
    if ci == 'bs':
        idx_r = np.random.choice(nT, size=nT)
        U_ = U[:, idx_r]
    if ci == 'wbs':
        bs_dist = 'Rademacher'
        if bs_dist == 'Rademacher':
            rescale = np.random.choice((-1, 1), size=(1, nT))
        if bs_dist == 'Normal':
            rescale = np.random.normal(size=(1, nT))
        U_ = U*rescale
    U_ = U_[:nY, :]
    if use_numba:
        Y_ = sim_var_njit(Y0, Bc, Bx, U_)
        B_, _, _, _, S_ = fit_var_h_njit(Y_, nC, nL, dfc)
        _, Bx_ = split_C_B(B_, nC, nL, nY)
        A0inv_ = np.linalg.cholesky(S_)
        ir_, irc_ = get_irfs_VARm(Bx_, A0inv_, nH)
    else:
        Y_ = sim_var(Y0, Bc, Bx, U_)
        B_, _, _, _, S_ = fit_var_h(Y_, nC, nL, dfc)
        _, Bx_ = split_C_B(B_, nC, nL, nY)
        A0inv_ = np.linalg.cholesky(S_)
        ir_, irc_ = get_irfs_VARm(Bx_, A0inv_, nH)
    return ir_, irc_

bs_njit = nb.njit(bs)

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
    if use_numba:
        Y_r = varsim_njit(c, B, U_r, Y0_r)
        c_r_, B_r_, U_r_, S_r_ = varols_njit(Y_r, nL)
        Psi_ = get_Psi_from_B_njit(B_r_, nH)
        A0inv_ = get_A0inv(method=method, U=U_r_, S=S_r_, idv=idv, M=M_r)
        ir_, irc_ = get_sirf_from_irf(Psi_, A0inv_, impulse)
    else:
        Y_r = varsim(c, B, U_r, Y0_r)
        c_r_, B_r_, U_r_, S_r_ = varols(Y_r, nL)
        Psi_ = get_Psi_from_B(B_r_, nH)
        A0inv_ = get_A0inv(method=method, U=U_r_, S=S_r_, idv=idv, M=M_r)
        ir_, irc_ = get_sirf_from_irf(Psi_, A0inv_, impulse)
    return ir_, irc_

get_bs_njit = nb.njit(get_bs)

# %% get IRFs
def get_irfs(Y,c,B,U,S,/,*,nH,method=None,impulse=None,cl=None,ci=None,nR=1000,idv=None,M=None):
    (nL,nY,_) = B.shape
    (_,n1) = Y.shape
    nT = n1 - nL

#         Psi = get_Psi_from_B(B,nH)
    Psi = get_Psi_from_B_njit(B,nH)
    A0inv = get_A0inv(method=method,U=U,S=S,idv=idv,M=M)
    irm,irmc = get_sirf_from_irf(Psi,A0inv,impulse)
    ir = block()
    irc = block()
    ir.mean = irm
    irc.mean = irmc

    if ci is not None:
        IR = np.full((nR,nH+1,nY,nY),np.nan)
        IRC = np.full((nR,nH+1,nY,nY),np.nan)
        if method == 'ch':
            M = [0 for _ in range(nT)]
        UM = np.row_stack((U,M))
        for r in range(nR):
            if (r+1) % 100 == 0:
                print('\r Bootstrap {}/{}'.format(r+1,nR),end='\r',flush=True)
            ir_,irc_ = get_bs(Y,c,B,U,S,UM,nL,nY,nH,nT,method=method,impulse=impulse,cl=cl,ci=ci,idv=idv,M=M)
#             ir_,irc_ = bs_njit(Y,c,B,U,S,UM,nL,nY,nH,nT,method=method,impulse=impulse,cl=cl,ci=ci,idv=idv,M=M)
            IR[r],IRC[r] = ir_,irc_
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

    return ir,irc,Psi,A0inv

# %%
def get_irfs_VARm(Bx, A0inv, nH):
    Psi = get_Psi_from_B(Bx, nH)
    Irf, Irfc = get_sirf_from_irf(Psi, A0inv)
    Irf = Irf.transpose((2, 1, 0))
    Irfc = Irfc.transpose((2, 1, 0))
    return Irf, Irfc

# %%
def get_irfs_sim_VARm(Y, B, U, /, *, model_spec, irf_spec):

    nY, _ = Y.shape
    nH, ci, nR = irf_spec['nH'], irf_spec['ci'], irf_spec['nR']

    if ci in ['bs','wbs']:
        Irf_Sim = np.full((nR, nY, nY, nH+1), np.nan)
        Irfc_Sim = np.full((nR, nY, nY, nH+1), np.nan)
        for r in range(nR):
            Irf_Sim[r], Irfc_Sim[r] = bs(Y, U, B, model_spec=model_spec, irf_spec=irf_spec)

    return Irf_Sim, Irfc_Sim

    # # Confidence intervals
    # if ci['ci'] in ['pbs','sim']:

    #     nR = ci['nR']

    #     Irf_Sim = np.full((nR, nY, nY, nH+1), np.nan)
    #     Irfc_Sim = np.full((nR, nY, nY, nH+1), np.nan)

    #     Bv = B.reshape((-1))
    #     Vv = sp.linalg.block_diag(*V)
    #     B_Sim = np.random.multivariate_normal(Bv, Vv, size=nR)
    #     for r in range(nR):
    #         B_r = B_Sim[r].reshape((nY,-1))
    #         _, Bx_r = split_C_B(B_r, nC, nL, nY)
    #         A0inv = np.linalg.cholesky(S)
    #         Irf_r, Irfc_r = get_irfs_VARm(Bx_r, A0inv, nH)
    #         Irf_Sim[r], Irfc_Sim[r] = Irf_r, Irfc_r

    #     return Irf, Irfc, Irf_Sim, Irfc_Sim
    # else:
    #     return Irf, Irfc, None, None

# %%
class irfs_VARm(irfs):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

# %% VAR

class VARm(model):

    def __init__(self, Ydata, /, *, var_names=None, add_constant=True, nL=1, dfc=True):

        super().__init__()

        if not isinstance(Ydata, pd.core.frame.DataFrame):
            raise TypeError('data must be in pandas DataFrame format')

        if var_names is None:
            var_names = Ydata.columns
        Ydata = Ydata[var_names]

        # Ydata, Xdata, Zdata = self.do_data(Ydata, Xdata, Zdata)
        self.Data.Ydata = Ydata
        nC = 1 if add_constant else 0
        self.Spec = dict(nC=nC, nL=nL, dfc=dfc)

        self.Irfs = None
        self.fit()
        self.irf()

    def fit(self):

        nC, nL, nY, dfc = self.Spec['nC'], self.Spec['nL'], self.Spec['nY'], self.Spec['dfc']
        Ydata = self.Data.Ydata

        Y = Ydata.values.T

        nY, n1 = Y.shape
        nT = n1-nL

        B, SE, V, U, S = fit_var_h(Y, nC, nL, dfc)

        Bc, Bx = split_C_B(B, nC, nL, nY)
        SEc, SEx = split_C_B(SE, nC, nL, nY)

        self.Spec['nY'], self.Spec['nT'] = nY, nT
        self.Est = {'Bc': Bc, 'Bx': Bx, 'SEc': SEc, 'SEx': SEx, 'B': B, 'SE': SE, 'V': V, 'U': U, 'S': S}

        return self

    def irf(self, **irf_spec):

        irf_spec_default = self.Irfs.Spec if hasattr(self.Irfs, 'Spec') else {'nH': 12, 'ci': 'bs', 'nR': 100}
        irf_spec = {**irf_spec_default, **irf_spec} if irf_spec else irf_spec_default

        model_spec = self.Spec
        nH = irf_spec['nH']
        Bx, B, U, S = self.Est['Bx'], self.Est['B'], self.Est['U'], self.Est['S']
        Ydata = self.Data.Ydata

        Y = Ydata.values.T

        A0inv = np.linalg.cholesky(S)

        # IRF at means
        Irf_m, Irfc_m = get_irfs_VARm(Bx, A0inv, nH)

        # IRF simulations
        Irf_Sim, Irfc_Sim = get_irfs_sim_VARm(Y, B, U, model_spec=model_spec, irf_spec=irf_spec)

        Irfs = irfs_VARm(Irf_m, Irfc_m, Irf_Sim, Irfc_Sim, irf_spec)

        self.Irfs = Irfs

        return self

# %%
class varm:

    # ==============================================================================================

    def __init__(self,data,/,*,nL=None,var_names=None,sample=None):

        if data.shape[0] < data.shape[1]:
            data = data.T

        if isinstance(data,pd.DataFrame):
            pass
        elif isinstance(data,(pd.Series,np.ndarray)):
            data = pd.DataFrame(data)
            data.columns = [str(i) for i in data.columns]

        if var_names is None:
            var_names = data.columns

        (n0,n1) = data.shape
        self.data = data
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
        nY = self.model.spec.nY
        var_names = self.model.spec.var_names
        data = self.data[var_names].iloc[isample[0]-nL:isample[1]+1,:].values

#         c, B, U, S = varols(data.T,nL)
        c, B, U, S = varols_njit(data.T,nL)

        self.model.parameters = block()
        self.model.parameters.c = c
        self.model.parameters.B = B
        self.model.parameters.S = S
        self.model.parameters.A0inv = block()
        self.model.residuals = block()
        self.model.residuals.rd = U.T

        if hasattr(self.model.svar,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.svar,'iv'):
            self.irf(method='iv')

    # ==============================================================================================

    def irf(self,nH=None,method=None,impulse=None,cl=None,ci=None,nR=None,idv=None,ins_names=None):

        if nH is None:
            if hasattr(self.model.irfs.spec,'nH'):
                nH = self.model.irfs.spec.nH
            else:
                nH = 48

        if impulse is None:
            if hasattr(self.model.irfs.spec,'impulse'):
                impulse = self.model.irfs.spec.impulse
            else:
                impulse = 'unit'

        if cl is None:
            if hasattr(self.model.irfs.spec,'cl'):
                cl = self.model.irfs.spec.cl
            else:
                cl = 0.95

        if ci is None:
            if hasattr(self.model.irfs.spec,'ci'):
                ci = self.model.irfs.spec.ci
            else:
                ci = None

        if nR is None:
            if hasattr(self.model.irfs.spec,'nR'):
                nR = self.model.irfs.spec.nR
            else:
                nR = 1000

        self.model.irfs.spec.nH = nH
        self.model.irfs.spec.impulse = impulse

        nT = self.model.spec.nT
        nL = self.model.spec.nL
        nY = self.model.spec.nY
        c = self.model.parameters.c
        B = self.model.parameters.B
        S = self.model.parameters.S
        U = self.model.residuals.rd.T
        isample = self.model.spec.isample
        var_names = self.model.spec.var_names
        data = self.data[var_names].iloc[isample[0]-nL:isample[1]+1,:].values
        Y = data.T

        if isinstance(cl,float):
            cl = [cl]

        if isinstance(impulse,list) or isinstance(impulse,np.ndarray):
            impulse = np.diag(impulse)
        if isinstance(impulse,float):
            impulse = impulse*np.eye(nY)


        if method == 'ch':
            idv = None
            ins_names = None
            M = None

        if method == 'iv':
            if idv is None or ins_names is None:
                if hasattr(self.model.svar,'iv'):
                    if hasattr(self.model.svar.iv,'idv') and hasattr(self.model.svar.iv,'ins_names'):
                        idv = self.model.svar.iv.idv
                        ins_names = self.model.svar.iv.ins_names
                    else:
                        raise SyntaxError('Please provide an instrument for SVAR-IV identification')
                else:
                    raise SyntaxError('Please provide an instrument for SVAR-IV identification')
            else:
                if isinstance(idv,int):
                    idv = [idv]
                if isinstance(ins_names,str):
                    ins_names = [ins_names]
            if len(idv) != len(ins_names):
                raise SyntaxError('The number of instruments must be equal the number of instrumented variables')

            instruments = self.data[ins_names].iloc[isample[0]:isample[1]+1,:].values
            M = instruments.T
            (nM,_) = M.shape

        if method in ('ch','iv'):
            ir,irc,Psi,A0inv = get_irfs(Y,c,B,U,S,nH=nH,method=method,impulse=impulse,cl=cl,ci=ci,nR=nR,idv=idv,M=M)
            self.model.irfs.spec.cl = cl
            if ci in ('bs','wbs'):
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

        ir,irc,Psi,A0inv = get_irfs(Y,c,B,U,S,nH=nH,impulse=impulse)
        self.model.irfs.rd = block()
        self.model.irfs.rd.ir = ir
        self.model.irfs.rd.irc = irc

    # ==============================================================================================

    def set_sample(self,sample=None):

        var_names = self.model.spec.var_names
        data = self.data[var_names].values
        datarownan = np.isnan(data).any(axis=1)
        if not datarownan.any():
            nanoffset = [0,0]
        else:
            datarownanchange = np.argwhere(datarownan[1:]!=datarownan[:-1])+1
            if datarownanchange.shape[0] == 1:
                if datarownan[0]:
                    nanoffset = [datarownanchange[0,0],0]
                else:
                    nanoffset = [0,data.shape[0]-datarownanchange[0,0]]
            elif datarownanchange.shape[0] == 2:
                nanoffset = [datarownanchange[0,0],data.shape[0]-datarownanchange[1,0]]
            elif datarownanchange.shape[0] > 2:
                raise ValueError('Sample should not contain NaNs')

        nL = self.model.spec.nL
        if sample is None:
            isample = (nL+nanoffset[0],self.data.shape[0]-1-nanoffset[1])
        else:
            try:
                isample = (max(self.data.index.get_loc(sample[0]),nL+nanoffset[0]),min(self.data.index.get_loc(sample[1]),self.data.shape[0]-1-nanoffset[1]))
                sample = (self.data.index[isample[0]].strftime('%Y-%m-%d'),self.data.index[isample[1]].strftime('%Y-%m-%d'))
            except KeyError:
                raise KeyError('Provided sample is outside of the available data range')


#         sample = (self.data.index[isample[0]],self.data.index[isample[1]])

        self.model.spec.sample = sample
        self.model.spec.isample = isample
        self.model.spec.nT = isample[1] - isample[0] + 1

        self.fit()

    # ==============================================================================================

    def set_lag_length(self,nL):

        self.model.spec.nL = nL
        self.fit()
        if hasattr(self.model.svar,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.svar,'iv'):
            self.irf(method='iv')

    # ==============================================================================================

    def set_horizon(self,nH):

        self.model.irfs.spec.nH = nH
        if hasattr(self.model.svar,'ch'):
            self.irf(nH=nH,method='ch')
        if hasattr(self.model.svar,'iv'):
            self.irf(nH=nH,method='iv')

# MM = varm(data,var_names=['0','1','2'],nL=4)
# MM.irf(method='iv',ci='bs',idv=2,ins_names='4')
# MM = varm(data,var_names=['0','1','3','2','4','5'],nL=4)
# MM.irf(method='iv',ci='wbs',nR=100,idv=[1],ins_names=['6'])
# %load_ext line_profiler
# data.shape
# %lprun -f bs varm(data,var_names=['0','1','3','2','4','5'],nL=4).irf(method='iv',ci='wbs',nR=1000,idv=[1],ins_names=['6'])
# print(MM.model.irfs.iv.ir.mean[0])
# print(MM.model.irfs.iv.ir.mean[0])



# print(MM. model.parameters.B[0])
# print(MM. model.parameters.B[0])
# ## LP model
# ### LP
# ### LP-OLS

# %% lpols
def lpols(Xdata=None,Ydata=None,Zdata=None,Wdata=None,nL=None,nH=None):
    """
    Function to estimate LP(nL,nH) model using OLS
    """
    if Xdata is None and Ydata is None:
        raise ValueError('No data provided')
    else:
        if Xdata is None:
            Xdata = Ydata
        if Ydata is None:
            Ydata = Xdata
    if Zdata is not None:
        (n0,n1) = Zdata.shape
        nZ = n0
    else:
        nZ = 0

    (n0,n1) = Ydata.shape
    nY = n0
    (n0,n1) = Xdata.shape
    nT = n1 - nL - nH + 1
    nX = n0
    nK = nL - 1

    Y = np.full((0,n1),np.nan)
    for h in range(0,nH+1):
        Y = np.row_stack((Y,np.roll(Ydata,-h)))

    X = np.asarray(Xdata)
    if Zdata is not None:
        X = np.row_stack((X,Zdata))

    Z = np.ones((1,n1))
    for l in range(1,nK+1):
        Z = np.row_stack((Z,np.roll(Xdata,l)))
    if Wdata is not None:
        Z = np.row_stack((Z,Wdata))

    X = X[:,nK:n1-nH]
    Y = Y[:,nK:n1-nH]
    Z = Z[:,nK:n1-nH]

    Mz = np.eye(nT) - Z.T@np.linalg.inv(Z@Z.T)@Z
    B = (Y@Mz@X.T)@np.linalg.inv(X@Mz@X.T)
    U = Y@Mz - B@X@Mz
    U = U.reshape((nH+1,nY,nT))
    B = B.T.reshape((nX+nZ,nH+1,nY)).transpose((1,2,0)) #.swapaxes(1,2)
    S = (1/nT)*(U[1]@U[1].T)
    return B, U, S


@nb.njit
def lpols_njit(Xdata=None,Ydata=None,Zdata=None,Wdata=None,nL=None,nH=None):
    """
    Function to estimate LP(nL,nH) model using OLS
    """
    if Xdata is None and Ydata is None:
        raise ValueError('No data provided')
    else:
        if Xdata is None:
            Xdata = Ydata
        if Ydata is None:
            Ydata = Xdata
    if Zdata is not None:
        (n0,n1) = Zdata.shape
        nZ = n0
    else:
        nZ = 0

    (n0,n1) = Ydata.shape
    nY = n0
    (n0,n1) = Xdata.shape
    nT = n1 - nL - nH + 1
    nX = n0
    nK = nL - 1

    Y = np.full((0,n1),np.nan)
    for h in range(0,nH+1):
        Y = np.row_stack((Y,np.roll(Ydata,-h)))

    X = np.asarray(Xdata)
    if Zdata is not None:
        X = np.row_stack((X,Zdata))

    Z = np.ones((1,n1))
    for l in range(1,nK+1):
        Z = np.row_stack((Z,np.roll(Xdata,l)))
    if Wdata is not None:
        Z = np.row_stack((Z,Wdata))

    X = np.ascontiguousarray(X[:,nK:n1-nH])
    Y = np.ascontiguousarray(Y[:,nK:n1-nH])
    Z = np.ascontiguousarray(Z[:,nK:n1-nH])

    Mz = np.eye(nT) - Z.T@np.linalg.inv(Z@Z.T)@Z
    B = np.linalg.solve(X@Mz@X.T,X@Mz@Y.T).T
    U = Y@Mz - B@X@Mz
    U = U.reshape((nH+1,nY,nT))
    B = B.T.reshape((nX+nZ,nH+1,nY)).transpose((1,2,0)) #.swapaxes(1,2)
    S = (1/nT)*(U[1]@U[1].T)
    return B, U, S


def get_lp_irfs(B,U,S,/,*,method=None,impulse=None,idv=None,M=None):
    (_,nY,nX) = B.shape

#         Psi = get_Psi_from_B(B,nH)
    Psi = np.array(B)
#     A0inv = get_A0inv(method=method,U=U,S=S,idv=idv,M=M)
    A0inv = np.eye(nX)
    irm,irmc = get_sirf_from_irf(Psi,A0inv,impulse)
    ir = block()
    irc = block()
    ir.mean = irm
    irc.mean = irmc
    return ir,irc,Psi,A0inv


class lpm:

    # ==============================================================================================

    def __init__(self,data,/,*,nL=None,nH=None,Y_var_names=None,X_var_names=None,sample=None):

        if data.shape[0] < data.shape[1]:
            data = data.T

        if isinstance(data,pd.DataFrame):
            pass
        elif isinstance(data,(pd.Series,np.ndarray)):
            data = pd.DataFrame(data)
            data.columns = [str(i) for i in data.columns]

        if Y_var_names is None:
            Y_var_names = data.columns.tolist()
        if X_var_names is None:
            X_var_names = data.columns.tolist()

        (n0,n1) = data.shape
        self.data = data
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
        nY = self.model.spec.nY
        nX = self.model.spec.nX
        Y_var_names = self.model.spec.Y_var_names
        X_var_names = self.model.spec.X_var_names
        var_names = self.model.spec.var_names
        Y_var_indices = [i for i,name in enumerate(self.data.columns) if name in Y_var_names]
        X_var_indices = [i for i,name in enumerate(self.data.columns) if name in X_var_names]
        data = self.data[var_names].iloc[isample[0]-nL:isample[1]+nH,:].values
        (n0,n1) = data.shape
        nK = nL - 1

        B, U, S = lpols_njit(Xdata=data[:,X_var_indices].T,Ydata=data[:,Y_var_indices].T,nL=nL,nH=nH)

# #         offset = nK

# #         print(n0,n1,nT,nL,nH)
# #         print(Y_var_indices,X_var_indices)
# #         y = np.full((nK+nH+1,nT,nY),np.nan)
# #         Y = np.full((nH+1,nT,nY),np.nan)
# #         X = np.full((nK+1,nT,nY),np.nan)

# #         # Creating y
# #         for idj in range(-nK,nH+1):
# #             y[offset+idj] = data[offset+idj:offset+nT+idj,Y_var_indices]
# #         # Creating Y
# #         for idj in range(1,nH+1):
# #             Y[idj] = data[offset+idj:offset+nT+idj,Y_var_indices] #y[offset+1:offset:idh,:,:]
# #         # Creating X
# #         for idj in range(0,nK):
# #             X[idj] = data[offset-idj:offset+nT-idj,X_var_indices] #y[offset+1:offset:idh,:,:]
# #         # Creating Z
# #         for idj in range(1,nK):
# #             Z[idj] = np.row_stack((np.ones((1,n0)),np.roll(data.T,p)))  X[] data[offset-idj:offset+nT-idj,X_var_indices] #y[offset+1:offset:idh,:,:]



#         Y = np.full((0,n0),np.nan)
#         for h in range(1,nH+1):
#             Y = np.row_stack((Y,np.roll(data[:,Y_var_indices].T,-h)))

#         X = data[:,X_var_indices].T

#         Z = np.ones((1,n0))
#         for l in range(1,nK+1):
#             Z = np.row_stack((Z,np.roll(data[:,X_var_indices].T,l)))

# #         print(Y.shape,X.shape,Z.shape)

#         X = X[:,nK:-nH].T
#         Y = Y[:,nK:-nH].T
#         Z = Z[:,nK:-nH].T
# #         print(Y.shape,X.shape,Z.shape)
# #         print(Y)
#         Mz = np.eye(nT) - Z@np.linalg.inv(Z.T@Z)@Z.T
#         B = np.linalg.inv(X.T@Mz@X)@(X.T@Mz@Y)
# #         print(Y)
# #         print(B)
# #         cB = (Y@Z.T)@(np.linalg.inv(Z@Z.T))
#         U = Mz@Y - Mz@X@B
#         U = U.T.reshape((nH,nX,nT))
#         S = (1/nT)*(U[0]@U[0].T)
#         B = B.reshape((nX,nH,nX)).swapaxes(0,1) #.swapaxes(1,2)


        self.model.parameters = block()
        self.model.parameters.B = B
        self.model.parameters.S = S
        self.model.parameters.A0inv = block()
        self.model.residuals = block()
        self.model.residuals.rd = U

        if hasattr(self.model.slp,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp,'iv'):
            self.irf(method='iv')

    # ==============================================================================================

    def irf(self,method=None,impulse=None,idv=None,ins_names=None):

        if impulse is None:
            if hasattr(self.model.irfs.spec,'impulse'):
                impulse = self.model.irfs.spec.impulse
            else:
                impulse = 'unit'

        self.model.irfs.spec.impulse = impulse
        self.model.irfs.spec.nH = self.model.spec.nH

        nT = self.model.spec.nT
        nL = self.model.spec.nL
        nX = self.model.spec.nX
        nH = self.model.spec.nH
        B = self.model.parameters.B
        S = self.model.parameters.S
        U = self.model.residuals.rd
        isample = self.model.spec.isample

        if isinstance(impulse,list) or isinstance(impulse,np.ndarray):
            impulse = np.diag(impulse)
        if isinstance(impulse,float):
            impulse = impulse*np.eye(nY)

        if method == 'ch':
            idv = None
            ins_names = None
            M = None

        if method == 'iv':
            if idv is None or ins_names is None:
                if hasattr(self.model.slp,'iv'):
                    if hasattr(self.model.slp.iv,'idv') and hasattr(self.model.slp.iv,'ins_names'):
                        idv = self.model.slp.iv.idv
                        ins_names = self.model.slp.iv.ins_names
                    else:
                        raise SyntaxError('Please provide an instrument for SLP-IV identification')
                else:
                    raise SyntaxError('Please provide an instrument for SLP-IV identification')
            else:
                if isinstance(idv,int):
                    idv = [idv]
                if isinstance(ins_names,str):
                    ins_names = [ins_names]
            if len(idv) != len(ins_names):
                raise SyntaxError('The number of instruments must be equal the number of instrumented variables')

            instruments = self.data[ins_names].iloc[isample[0]:isample[1]+1,:].values
            M = instruments.T
            (nM,_) = M.shape

        if method in ('ch','iv'):
            ir,irc,Psi,A0inv = get_lp_irfs(B,U,S,method=method,impulse=impulse,idv=idv,M=M)
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

        ir,irc,Psi,A0inv = get_lp_irfs(B,U,S,impulse=impulse)
        self.model.irfs.rd = block()
        self.model.irfs.rd.ir = ir
        self.model.irfs.rd.irc = irc

    # ==============================================================================================

    def set_sample(self,sample=None):

        var_names = self.model.spec.var_names
        data = self.data[var_names].values
        datarownan = np.isnan(data).any(axis=1)
        if not datarownan.any():
            nanoffset = [0,0]
        else:
            datarownanchange = np.argwhere(datarownan[1:]!=datarownan[:-1])+1
            if datarownanchange.shape[0] == 1:
                if datarownan[0]:
                    nanoffset = [datarownanchange[0,0],0]
                else:
                    nanoffset = [0,data.shape[0]-datarownanchange[0,0]]
            elif datarownanchange.shape[0] == 2:
                nanoffset = [datarownanchange[0,0],data.shape[0]-datarownanchange[1,0]]
            elif datarownanchange.shape[0] > 2:
                raise ValueError('Sample should not contain NaNs')

        nL = self.model.spec.nL
        nH = self.model.spec.nH
        if sample is None:
            isample = (nL+nanoffset[0],self.data.shape[0]-nH-nanoffset[1])
        else:
            isample = (max(self.data.index.get_loc(sample[0]),nL+nanoffset[0]),min(self.data.index.get_loc(sample[1]),self.data.shape[0]-nH-nanoffset[1]))
            sample = (self.data.index[isample[0]].strftime('%Y-%m-%d'),self.data.index[isample[1]].strftime('%Y-%m-%d'))


#         sample = (self.data.index[isample[0]],self.data.index[isample[1]])

        self.model.spec.sample = sample
        self.model.spec.isample = isample
        self.model.spec.nT = isample[1] - isample[0] + 1

        self.fit()

    # ==============================================================================================

    def set_lag_length(self,nL):

        self.model.spec.nL = nL
        self.fit()
        if hasattr(self.model.slp,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp,'iv'):
            self.irf(method='iv')

    # ==============================================================================================

    def set_horizon(self,nH):

        self.model.spec.nH = nH
        self.fit()
        if hasattr(self.model.slp,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp,'iv'):
            self.irf(method='iv')

# ## SFM model
# ### SFM

class sfm:
    def __init__(self,data,tcodes,nF=None,make_fig=False):

        if isinstance(data,pd.DataFrame):
            Data = data
        elif isinstance(data,(pd.Series,np.ndarray)):
            Data = pd.DataFrame(data)

        data = Data.values

        self.Data = Data
        self.tcodes = tcodes

        datatrans = self.transform_by_tcode(data,tcodes)
        DataTrans = pd.DataFrame(datatrans,columns=Data.columns,index=Data.index)
        X = datatrans[2:,:]

        Xcolnan = np.isnan(X).any(axis=0)
        X_nonan = X[:,~Xcolnan]
        tcodes_nonan = tcodes[~Xcolnan]

        X_S,X_Mu,X_StD = self.standardize(X_nonan)
        fac,lam = self.getFactors(X_S,nF,make_fig)

        Factors = pd.DataFrame(fac,columns=[('Factor_'+str(f)) for f in range(1,fac.shape[1]+1)],index=Data.index[2:])
        Lambda = pd.DataFrame(lam,columns=Data.columns[~Xcolnan],index=range(fac.shape[1]))

        self.DataTrans = DataTrans
        self.X = X_nonan
        self.X_S = X_S
        self.X_Mu = X_Mu
        self.X_StD = X_StD
        self.Factors = Factors
        self.Lambda = Lambda
        self.F = fac
        self.L = lam
        self.nT = data.shape[0]
        self.nN = data.shape[1]
        self.nF = fac.shape[1]

    def transform_by_tcode(self,rawdata,tcodes):
        def transxf(xin,tcode):
            x = xin.copy()
            nT = len(x)
            y = np.full_like(x,np.nan)
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
        transformed_data = np.full_like(rawdata,np.nan)
        for (idv,x),tcode in zip(enumerate(rawdata.T),tcodes):
            transformed_data[:,idv] = transxf(x,tcode)
        return transformed_data

    def standardize(self,X):
        X_Mu = np.mean(X,axis=0)
        X_StD = np.std(X,axis=0,ddof=1)
        X_S = (X-X_Mu)/X_StD
        return X_S,X_Mu,X_StD

    def getFactors(self,X,nF=None,make_fig=False):
        (nT,nN) = X.shape
        S_xx = (1/nT)*(X.T@X)
        eigVal, eigVec = np.linalg.eigh(S_xx)
        # sort eigenvalues in descending order
        idx = np.argsort(eigVal)[::-1]
        eigVal = eigVal[idx]
        eigVec = eigVec[:,idx]

        if nF is None:
            r_max = int(np.floor(np.sqrt(nN)))
            V = np.full(r_max,np.nan)  # sum of squared residuals
            IC = np.full(r_max,np.nan) # information criterion

            for r in range(1,r_max+1):
                lam = np.sqrt(nN)*eigVec[:,:r].T
                fac = (1/nN)*(X@lam.T)
                V[r-1] = (1/(nN*nT))*np.trace((X-fac@lam).T@(X-fac@lam))
                IC[r-1] = np.log(V[r-1]) + r*(nN+nT)/(nN*nT)*np.log(min(nN,nT))

            if make_fig:
                fig,axes = mpl.subplots(1,2,figsize=(12,4))
                axes[0].plot(range(1,r_max+1),IC,'-o')
                axes[0].set_xticks(range(1,r_max+1))
                axes[0].set_xlim((0,r_max+1))
                axes[0].set_title('Bai & Ng Criterion')
                axes[1].plot(range(1,r_max+1),eigVal[:r_max],'-o')
                axes[1].set_xticks(range(1,r_max+1))
                axes[1].set_xlim((0,r_max+1))
                axes[1].set_title('Eigenvalues')
            nFF = np.argmin(IC)+1
            print(f'Number of factors selected by Bai & Ng criterion is {nFF}')
        else:
            nFF = nF
        lam = np.sqrt(nN)*eigVec[:,:nFF].T
        fac = (1/nN)*(X@lam.T)
        return fac, lam

# %%
