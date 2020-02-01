#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl


# a = np.random.random((100,3))
# B = np.random.random((100,10))

# In[3]:


def ols(yin,Xin,dfcin=True):
    y = yin.copy()
    X = Xin.copy()
    dfc = bool(dfcin)
    
    if len(y.shape) == 1:
        y = y[:,None]
        
    if 0 not in np.var(X,axis=0):
        X = np.column_stack((np.ones((X.shape[0],1)),X))
        print('The data passed does not contain a constant. Automatically adding a constant')
        
    nN,nY = y.shape
    _,nK = X.shape
    
    if nY == 1:
        yX = np.column_stack((y,X))
        yXnan = np.isnan(yX).any(axis=1)
        y[yXnan,:],X[yXnan,:] = 0,0
        b = np.linalg.solve(X.T@X,X.T@y)
        e = y-X@b
        invXX = np.linalg.inv(X.T@X)
        nN = nN-np.nansum(yXnan,axis=0)
        if dfc:
            df = nN-nK
        else:
            df = nN
        S = (1/df)*(e.T@e)[0]
        V = S*invXX
        se = np.sqrt(np.diagonal(V)).T
        e[yXnan] = np.nan
    else:
#     ynan = np.isnan(y)
#     Xrownan = np.isnan(X).any(axis=1)
#     resnan = ynan | Xrownan[:,None]
#     print(ynan)
#     y_ = y[~Xnan.any(axis=1)]
#     X_ = X[~Xnan.any(axis=1),:]
#     y[resnan] = 0
#     X[Xrownan,:] = 0

        try:
            yy = y.T.reshape((1,nY*nN)).T
            XX = np.kron(np.eye(nY),X)
            yyXX = np.column_stack((yy,XX))
            yyXXnan = np.isnan(yyXX).any(axis=1)
            yy[yyXXnan,:] = 0
            XX[yyXXnan,:] = 0
            XXTXX = XX.T@XX
            bb = np.linalg.solve(XXTXX, XX.T@yy)
            ee = yy-XX@bb
            ee[yyXXnan,:] = 0
            b = bb.reshape((nY,nK)).T
            e = ee.reshape((nY,nN)).T
            b = np.linalg.solve(X.T@X,X.T@y)
            e = y-X@b
            e[resnan] = 0
            resnan = yyXXnan.reshape((nY,nN)).T
            invXX = np.array([np.linalg.inv(XXTXX[i*nK:(i+1)*nK,i*nK:(i+1)*nK]) for i in range(nY)])
        except:
            ynan = np.isnan(y)
            Xrownan = np.isnan(X).any(axis=1)
            b = np.full((nK,nY),np.nan)
            e = np.full((nN,nY),np.nan)
            resnan = np.full((nN,nY),True)
            invXX = np.full((nY,nK,nK),np.nan)
            for idy in range(nY):
                y_,X_ = y[:,idy,None],X[:,:]
                yXnan = ynan[:,idy] | Xrownan
        #         print(yXnan)
                y_[yXnan,:],X[yXnan,:] = 0,0
                b_ = np.linalg.solve(X_.T@X_, X_.T@y_)
                e_ = y_-X_@b_
        #         print(e_.shape)
                b[:,idy,None] = b_
        #         print(b)
                e[:,idy,None] = e_
                resnan[:,idy] = yXnan
                invXX[idy] = np.linalg.inv(X_.T@X_)

    #     print(e)

    #     print(yy.T.reshape((nY,nN)).T)
        nNN = np.array([nN for i in range(nY)])-np.nansum(resnan,axis=0)
    #     print(np.diagonal(e.T@e))
        #     np.full_like(y,np.nan)
    #     print(nN)
        if dfc:
            df = np.array([[min(nNN[col],nNN[row])-nK for col in range(nY)] for row in range(nY)])
        else:
            df = np.array([[min(nNN[col],nNN[row]) for col in range(nY)] for row in range(nY)])

    #     print(df)
        S = (1/df)*(e.T@e)
        V = np.diagonal(S)[:,None,None]*invXX
        se = np.sqrt([np.diagonal(V[i]) for i in range(nY)]).T
    #     for idy in range(nY):
    #         V[idy] = Sigma2[idy]*np.linalg.inv(X.T@X)
        e[resnan] = np.nan
    return b,se,V,e,S


# In[4]:


# df = pd.read_csv('testdata.csv')


# b,se,V,e,S = ols(df.values[:,0:3],df.values[:,3:13])

# In[5]:


# S


# a=[1,2,3]
# def func(A):
#     A[1] = 4
#     print(A)
# func(a)
# print(a)

# In[6]:


class nsm(object):
        
    def __init__(self,yields,tau,lam):
        
#         if len(yields.shape) == 1:
#             yields = yields[None,:]
            
        if yields.shape[1] != tau.shape[0]:
            raise SyntaxError('yields and tau must have the same length')
        
        self.yields = pd.DataFrame(yields)
        self.yields.columns = tau
        self.tau = tau
        self.lam = lam
        self.fit()
        
    def getLoadings(self,tau,lam):
        b1l = np.ones_like(tau)
        b2l = np.array((1-np.exp(-lam*tau))/(lam*tau))
        b3l = np.array((1-np.exp(-lam*tau))/(lam*tau)-np.exp(-lam*tau))
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
        mpl.scatter(tau,self.yields.loc[index].values);
        mpl.plot(ptau,self.curve.loc[index].values);


# 

# def YieldCurveLam(b1,b2,b3,lam):
#     X = getLoadings(lam,tau)
#     return X @ np.array([b1,b2,b3]).T

# def getBetas(yields,lam,tau):
#     X = getLoadings(lam,tau)
#     betas,_,_ = ols(yields,X)
#     return betas

# In[7]:


class varms(object):
    
    class dir(object):
        def __init__(self):
            pass
        
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
        self.irfs = self.dir()
    
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

        self.parameters = self.dir()
        self.parameters.c = c
        self.parameters.B = B
        self.parameters.S = S
        self.residuals = self.dir()
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
        
        self.irfs.rd = Psi
        self.irfs.rdc = np.cumsum(Psi,0)
        self.parameters.A0inv = self.dir()

        if method == 'cholesky':
            A0inv = np.linalg.cholesky(S)
            self.irfs.ch = Psi@A0inv
            self.irfs.chc = np.cumsum(Psi@A0inv,0)
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
            self.irfs.iv = Psi@A0inv
            self.irfs.ivc = np.cumsum(Psi@A0inv,0)
            self.parameters.A0inv.iv = A0inv
            self.iv = self.dir()
            self.iv.idv = idv
            self.iv.ins_names = ins_names


# In[188]:


class varm(object):
    
    class dir(object):
        def __init__(self):
            pass
        
    def __init__(self,data,nL=None,var_names=None,sample=None):
        
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
        self.model = self.dir()
        self.model.var_names = var_names
        self.model.nL = nL
        self.model.nY = len(var_names)
        self.model.svar = self.dir()
        self.irfs = self.dir()
        self.set_sample(sample)
        
    def varols(self,data,nL):
        (n0,n1) = data.shape
        nT = n0 - nL
        nY = n1
        Z = np.ones((1,n0))

        for p in range(1,1+nL):
            Z = np.row_stack((Z,np.roll(data.T,p)))

        Z = Z[:,nL:]
        Y = data.T[:,nL:]

        cB = (Y@Z.T)@(np.linalg.inv(Z@Z.T))
        c = cB[:,0]
        B = cB[:,1:].T.reshape((nL,nY,nY)).swapaxes(1,2)
        U = Y-cB@Z
        S = (1/(nT-nL*nY-1))*(U@U.T)
        return c, B, U, S

    def varsim(self,c,B,U,Y0):
        (nY,nT) = U.shape
        (_,nL) = Y0.shape
        Y = np.full(((nY,nL+nT)),np.nan)
        Y[:,:nL] = Y0
        
        for t in range(nL,nL+nT):
#             BB = B.swapaxes(1,2).reshape((nL*nY,nY)).T
#             Y[:,t] = c + (BB@Y[:,t-nL:t][:,::-1].T.reshape((nL*nY,1))).reshape((-1,)) + U[:,t-nL]
            Y_t = c + U[:,t-nL]
            for l in range(nL):
                Y_t += B[l]@Y[:,t-l-1]
            Y[:,t] = Y_t
        return Y
    
    def B2Psi(self,nL,nH,nY,B):
        Psi = np.zeros((nH+1,nY,nY))
        Psi[0] = np.eye(nY)
        for h in range(1,nH+1):
            for i in range(min(h,nL)):
                Psi[h] += Psi[h-i-1]@B[i]
        return Psi

    def A0inv_iv(self,nL,U,S,idv,instruments):
        A0inv = np.sqrt(np.diag(np.diag(S)))
        for v,ins in zip(idv,instruments.T):
            insU = np.column_stack((ins.T,U.T))
            insUnan = np.isnan(insU)
            insU = insU[~insUnan.any(axis=1),:]
            if insU.shape[0] < 10:
                raise ValueError('Not enough observations to perform SVAR-IV identification')
            insUcov = np.cov(insU,rowvar=False)
            insUstd = np.std(insU,axis=0,ddof=1).reshape(-1,1)
#                 A0inv[:,v] = (insUcov[1:,0]/insUstd[0]).T # st. dev. of explained part
            A0inv[:,v] = (insUcov[1:,0]/(insUcov[v+1,0]/insUstd[v+1])).T # st. dev. of residual
#                 A0inv[:,v] = A0inv[:,v]/A0inv[v,v] # unit
        return A0inv

    def irf2sirf(self,Psi,A0inv,impulse):
        if impulse == 'unit':
            impulse_scale = np.diag(1/np.diag(A0inv))
        if impulse == '1sd':
            impulse_scale = np.eye(nY)
        Impact = A0inv@impulse_scale
        ir = Psi@Impact
        irc = np.cumsum(Psi@Impact,0)
        return ir, irc
    
    def bs(self,Y,c,B,U,S,idv,instruments,impulse,ci,nR):
        nL = self.model.nL
        nY = self.model.nY
        nH = self.model.nH
        nT = self.model.nT
        IR = np.full((nR,nH+1,nY,nY),np.nan)
        IRC = np.full((nR,nH+1,nY,nY),np.nan)
        Uinstruments = np.column_stack((U.T,instruments))
        for r in range(nR):
            print('\r Bootstrap {}/{}'.format(r+1,nR), end='\r', flush=True)
            Y0_r = Y[:,:nL]
            idx_r = np.random.choice(nT,size=nT)
            if ci == 'bs':
                rescale = np.ones((nT,nY))
            if ci == 'wbs':
                rescale = np.random.normal(size=(nT,nY))
            for v in idv:
                rescale = np.column_stack((rescale,rescale[:,v]))
            Uinstruments_r = rescale*Uinstruments[idx_r,:]
            U_r = Uinstruments_r[:,:nY].T
            instruments_r = Uinstruments_r[:,nY:]
            Y_r = self.varsim(c,B,U_r,Y0_r)
            c_r_, B_r_, U_r_, S_r_ = self.varols(Y_r.T,nL)
            Psi_ = self.B2Psi(nL,nH,nY,B_r_)
            A0inv_ = self.A0inv_iv(nL,U_r_,S_r_,idv,instruments_r)
            ir_,irc_ = self.irf2sirf(Psi_,A0inv_,impulse)
            IR[r],IRC[r] = ir_,irc_
        Psi = self.B2Psi(nL,nH,nY,B)
        A0inv = self.A0inv_iv(nL,U,S,idv,instruments)
        irm,irmc = self.irf2sirf(Psi,A0inv,impulse)
        ir = self.dir()
        irc = self.dir()
        ir.irm = irm
        irc.irm = irmc
        ir.ir005 = np.quantile(IR, 0.005, axis=0)
        ir.ir025 = np.quantile(IR, 0.025, axis=0)
        ir.ir050 = np.quantile(IR, 0.050, axis=0)
        ir.ir100 = np.quantile(IR, 0.100, axis=0)
        ir.ir250 = np.quantile(IR, 0.250, axis=0)
        ir.ir500 = np.quantile(IR, 0.500, axis=0)
        ir.ir750 = np.quantile(IR, 0.750, axis=0)
        ir.ir900 = np.quantile(IR, 0.900, axis=0)
        ir.ir950 = np.quantile(IR, 0.950, axis=0)
        ir.ir975 = np.quantile(IR, 0.975, axis=0)
        ir.ir995 = np.quantile(IR, 0.995, axis=0)
        irc.ir005 = np.quantile(IRC, 0.005, axis=0)
        irc.ir025 = np.quantile(IRC, 0.025, axis=0)
        irc.ir050 = np.quantile(IRC, 0.050, axis=0)
        irc.ir100 = np.quantile(IRC, 0.100, axis=0)
        irc.ir250 = np.quantile(IRC, 0.250, axis=0)
        irc.ir500 = np.quantile(IRC, 0.500, axis=0)
        irc.ir750 = np.quantile(IRC, 0.750, axis=0)
        irc.ir900 = np.quantile(IRC, 0.900, axis=0)
        irc.ir950 = np.quantile(IRC, 0.950, axis=0)
        irc.ir975 = np.quantile(IRC, 0.975, axis=0)
        irc.ir995 = np.quantile(IRC, 0.995, axis=0)
        return ir,irc,Psi,A0inv
        
    def fit(self):
        isample = self.model.isample
        nL = self.model.nL
        nY = self.model.nY
        var_names = self.model.var_names
        data = self.data[var_names].iloc[isample[0]-nL:isample[1]+1,:].values
        
        c, B, U, S = self.varols(data,nL)

        self.model.parameters = self.dir()
        self.model.parameters.c = c
        self.model.parameters.B = B
        self.model.parameters.S = S
        self.model.parameters.A0inv = self.dir()
        self.model.residuals = self.dir()
        self.model.residuals.rd = U.T
        if hasattr(self.model.svar,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.svar,'iv'):
            self.irf(method='iv')

    def irf(self,nH=None,method='ch',impulse='unit',ci=None,nR=1000,idv=None,ins_names=None):
        
        if nH is None:
            if hasattr(self.model,'nH'):
                nH = self.model.nH
            else:
                nH = 50

        if ci is None:
            if hasattr(self.model,'ci'):
                ci = self.model.ci
            else:
                ci = None
        
        self.model.nH = nH
        self.model.impulse = impulse
        
        nT = self.model.nT
        nL = self.model.nL
        nY = self.model.nY
        c = self.model.parameters.c
        B = self.model.parameters.B
        S = self.model.parameters.S
        U = self.model.residuals.rd.T
        isample = self.model.isample
        var_names = self.model.var_names
        data = self.data[var_names].iloc[isample[0]-nL:isample[1]+1,:].values
        Y = data.T
            
        if method == 'ch':
            Psi = self.B2Psi(nL,nH,nY,B)
            A0inv = np.linalg.cholesky(S)
            ir,irc = self.irf2sirf(Psi,A0inv,impulse)
            self.irfs.ch = ir
            self.irfs.chc = irc
            self.model.parameters.A0inv.ch = A0inv
            self.model.residuals.ch = U.T@np.linalg.inv(A0inv)
            self.model.svar.ch = self.dir()
            self.model.svar.ch.impulse = impulse

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
                if type(idv) == int:
                    idv = np.array([idv])
                if type(idv) == list:
                    idv = np.array(idv)
            if type(ins_names) == str:
                ins_names = [ins_names]
            if type(ins_names) == list:
                pass
            if idv.shape[0] != len(ins_names):
                raise SyntaxError('The number of instruments must be equal the number of instrumented variables')
            
            instruments = self.data[ins_names].iloc[isample[0]:isample[1]+1,:].values

            if ci is None:
                Psi = self.B2Psi(nL,nH,nY,B)
                A0inv = self.A0inv_iv(nL,U,S,idv,instruments)
                irm,irmc = self.irf2sirf(Psi,A0inv,impulse)
                ir = self.dir()
                ir.irm = irm
                irc = self.dir()
                irc.irm = irmc
            if ci == 'bs' or ci == 'wbs':
                ir,irc,Psi,A0inv = self.bs(Y,c,B,U,S,idv,instruments,impulse,ci,nR)
                self.model.nR = nR
                self.model.ci = ci
        
            self.irfs.rd = self.dir()
            self.irfs.rd.ir = Psi
            self.irfs.rd.irc = np.cumsum(Psi,0)
            self.irfs.iv = self.dir()
            self.irfs.iv.ir = ir
            self.irfs.iv.irc = irc
            self.model.parameters.A0inv.iv = A0inv
            self.model.svar.iv = self.dir()
            self.model.svar.iv.idv = idv
            self.model.svar.iv.ins_names = ins_names
            self.model.svar.iv.impulse = impulse
            
    def set_sample(self,sample=None):
        
        var_names = self.model.var_names
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
                    nanoffset = [0,data.shape[0]-datarownanchange[1,0]]
            elif datarownanchange.shape[0] == 2:
                nanoffset = [datarownanchange[0,0],data.shape[0]-datarownanchange[1,0]]
            elif datarownanchange.shape[0] > 2:
                raise ValueError('Sample should not contain NaNs')

        nL = self.model.nL
        if sample is None:
            isample = (nL+nanoffset[0],self.data.shape[0]-1-nanoffset[1])
        else:
            try:
                isample = (max(self.data.index.get_loc(sample[0]),nL+nanoffset[0]),min(self.data.index.get_loc(sample[1]),self.data.shape[0]-1-nanoffset[1]))
                sample = (self.data.index[isample[0]].strftime('%Y-%m-%d'),self.data.index[isample[1]].strftime('%Y-%m-%d'))
            except KeyError:
                raise KeyError('Provided sample is outside of the available data range')

            
#         sample = (self.data.index[isample[0]],self.data.index[isample[1]])         

        self.model.sample = sample
        self.model.isample = isample
        self.model.nT = isample[1] - isample[0] + 1
        self.fit()
        
    def set_lag_length(self,nL):
        
        self.model.nL = nL
        self.fit()
        if hasattr(self.model.svar,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.svar,'iv'):
            self.irf(method='iv')

    def set_horizon(self,nH):
        
        self.model.nH = nH
        if hasattr(self.model.svar,'ch'):
            self.irf(nH=nH,method='ch')
        if hasattr(self.model.svar,'iv'):
            self.irf(nH=nH,method='iv')


# MM = varm(B,var_names=['0','1','2'],nL=4)
# MM.irf(method='iv',ci='bs',idv=2,ins_names='4')

# In[311]:


class lpm(object):
    
    class dir(object):
        def __init__(self):
            pass
        
    def __init__(self,data,nL=None,nH=None,Y_var_names=None,X_var_names=None,sample=None):
        
        if data.shape[0] < data.shape[1]:
            data = data.T
        
        if isinstance(data,pd.DataFrame):
            pass
        elif isinstance(data,(pd.Series,np.ndarray)):
            data = pd.DataFrame(data)
            
        if Y_var_names is None:
            Y_var_names = data.columns
        if X_var_names is None:
            X_var_names = data.columns
            
        (n0,n1) = data.shape
        self.data = data
        self.model = self.dir()
        self.model.Y_var_names = Y_var_names
        self.model.X_var_names = X_var_names
        self.model.var_names = list(set(Y_var_names).intersection(X_var_names))      
        self.model.nL = nL
        self.model.nH = nH
        self.model.nY = len(Y_var_names)
        self.model.nX = len(X_var_names)
        self.model.slp = self.dir()
        self.irfs = self.dir()
        self.set_sample(sample)
    
    def fit(self):
        isample = self.model.isample
        nL = self.model.nL
        nH = self.model.nH
        nY = self.model.nY
        nX = self.model.nX
        Y_var_names = self.model.Y_var_names
        X_var_names = self.model.X_var_names
        var_names = self.model.var_names
        Y_var_indices = [i for i,name in enumerate(self.data.columns) if name in Y_var_names]
        X_var_indices = [i for i,name in enumerate(self.data.columns) if name in X_var_names]
        data = self.data[var_names].iloc[isample[0]-nL:isample[1]+nH+1,:].values
        (n0,n1) = data.shape
        nT = self.model.nT
        nK = nL - 1
#         offset = nK
        
#         print(n0,n1,nT,nL,nH)
#         print(Y_var_indices,X_var_indices)
#         y = np.full((nK+nH+1,nT,nY),np.nan)
#         Y = np.full((nH+1,nT,nY),np.nan)
#         X = np.full((nK+1,nT,nY),np.nan)
        
#         # Creating y
#         for idj in range(-nK,nH+1):
#             y[offset+idj] = data[offset+idj:offset+nT+idj,Y_var_indices]
#         # Creating Y
#         for idj in range(1,nH+1):
#             Y[idj] = data[offset+idj:offset+nT+idj,Y_var_indices] #y[offset+1:offset:idh,:,:]
#         # Creating X
#         for idj in range(0,nK):
#             X[idj] = data[offset-idj:offset+nT-idj,X_var_indices] #y[offset+1:offset:idh,:,:]
#         # Creating Z
#         for idj in range(1,nK):
#             Z[idj] = np.row_stack((np.ones((1,n0)),np.roll(data.T,p)))  X[] data[offset-idj:offset+nT-idj,X_var_indices] #y[offset+1:offset:idh,:,:]
        


        Y = np.full((0,n0),np.nan)
        for h in range(1,nH+1):
            Y = np.row_stack((Y,np.roll(data[:,Y_var_indices].T,-h)))
        
        X = data[:,X_var_indices].T
        
        Z = np.ones((1,n0))
        for l in range(1,nK+1):
            Z = np.row_stack((Z,np.roll(data[:,X_var_indices].T,l)))
        
#         print(Y.shape,X.shape,Z.shape)
        
        X = X[:,nK:-nH].T
        Y = Y[:,nK:-nH].T
        Z = Z[:,nK:-nH].T
#         print(Y.shape,X.shape,Z.shape)
#         print(Y)
        Mz = np.eye(nT) - Z@np.linalg.inv(Z.T@Z)@Z.T
        B = np.linalg.inv(X.T@Mz@X)@(X.T@Mz@Y)
#         print(Y)
#         print(B)
#         cB = (Y@Z.T)@(np.linalg.inv(Z@Z.T))
        U = Mz@Y - Mz@X@B
        U = U.T.reshape((nH,nX,nT))
        S = (1/nT)*(U[0]@U[0].T)
        B = B.reshape((nX,nH,nX)).swapaxes(0,1) #.swapaxes(1,2)

        
        self.model.parameters = self.dir()
        self.model.parameters.B = B
        self.model.parameters.S = S
        self.model.parameters.A0inv = self.dir()
        self.model.residuals = self.dir()
        self.model.residuals.rd = U
        if hasattr(self.model.slp,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp,'iv'):
            self.irf(method='iv')

    def irf(self,method='ch',impulse='unit',idv=None,ins_names=None):
        
        self.model.impulse = impulse
        
        nT = self.model.nT
        nL = self.model.nL
        nX = self.model.nX
        nH = self.model.nH
        B = self.model.parameters.B
        S = self.model.parameters.S
        U = self.model.residuals.rd
        isample = self.model.isample
        data = self.data.iloc[isample[0]-nL:isample[1]+nH+1,:].values

        Psi = np.zeros((nH+1,nX,nX))
        Psi[0] = np.eye(nX)
        for h in range(1,nH+1):
            Psi[h] = B[h-1]
        
        self.irfs.rd = Psi
        self.irfs.rdc = np.cumsum(Psi,0)
        
        def irf2sirf(Psi,A0inv,impulse):
            if impulse == 'unit':
                impulse_scale = np.diag(1/np.diag(A0inv))
            if impulse == '1sd':
                impulse_scale = np.eye(nX)
            Impact = A0inv@impulse_scale
            ir = Psi@Impact
            irc = np.cumsum(Psi@Impact,0)
            return ir, irc
            
        if method == 'ch':
            A0inv = np.linalg.cholesky(S)
            ir,irc = irf2sirf(Psi,A0inv,impulse)
            self.irfs.ch = ir
            self.irfs.chc = irc
            self.model.parameters.A0inv.ch = A0inv
            self.model.residuals.ch = U[0].T@np.linalg.inv(A0inv)
            self.model.slp.ch = self.dir()
            self.model.slp.ch.impulse = impulse

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
                if type(idv) == int:
                    idv = np.array([idv])
                if type(idv) == list:
                    idv = np.array(idv)
            if type(ins_names) == str:
                ins_names = [ins_names]
            if type(ins_names) == list:
                pass
            if idv.shape[0] != len(ins_names):
                raise SyntaxError('The number of instruments must be equal the number of instrumented variables')
            
            instruments = self.data[ins_names].iloc[isample[0]-nL:isample[1]+nH+1,:].values

            A0inv = np.sqrt(np.diag(np.diag(S)))
            for v,ins in zip(idv,instruments.T):
                insU = np.column_stack((ins.T[nK:-nH],U))
                insUnan = np.isnan(insU)
                insU = insU[~insUnan.any(axis=1),:]
                if insU.shape[0] < 10:
                    raise ValueError('Not enough observations to perform SVAR-IV identification')
                insUcov = np.cov(insU,rowvar=False)
                insUstd = np.std(insU,axis=0,ddof=1).reshape(-1,1)
#                 A0inv[:,v] = (insUcov[1:,0]/insUstd[0]).T # st. dev. of explained part
                A0inv[:,v] = (insUcov[1:,0]/(insUcov[v+1,0]/insUstd[v+1])).T # st. dev. of residual
#                 A0inv[:,v] = A0inv[:,v]/A0inv[v,v] # unit
            ir,irc = irf2sirf(Psi,A0inv,impulse)
            self.irfs.iv = ir
            self.irfs.ivc = irc
            self.model.parameters.A0inv.iv = A0inv
            self.model.slp.iv = self.dir()
            self.model.slp.iv.idv = idv
            self.model.slp.iv.ins_names = ins_names
            self.model.slp.iv.impulse = impulse

    def set_sample(self,sample=None):
        
        var_names = self.model.var_names
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
                    nanoffset = [0,data.shape[0]-datarownanchange[1,0]]
            elif datarownanchange.shape[0] == 2:
                nanoffset = [datarownanchange[0,0],data.shape[0]-datarownanchange[1,0]]
            elif datarownanchange.shape[0] > 2:
                raise ValueError('Sample should not contain NaNs')

        nL = self.model.nL
        nH = self.model.nH
        if sample is None:
            isample = (nL+nanoffset[0],self.data.shape[0]-nH-nanoffset[1])
        else:
            isample = (max(self.data.index.get_loc(sample[0]),nL+nanoffset[0]),min(self.data.index.get_loc(sample[1]),self.data.shape[0]-nH-nanoffset[1]))
            sample = (self.data.index[isample[0]].strftime('%Y-%m-%d'),self.data.index[isample[1]].strftime('%Y-%m-%d'))

            
#         sample = (self.data.index[isample[0]],self.data.index[isample[1]])         

        self.model.sample = sample
        self.model.isample = isample
        self.model.nT = isample[1] - isample[0] + 1
        self.fit()
        
    def set_lag_length(self,nL):
        
        self.model.nL = nL
        self.fit()
        if hasattr(self.model.slp,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp,'iv'):
            self.irf(method='iv')

    def set_horizon(self,nH):
        
        self.model.nH = nH
        self.fit()
        if hasattr(self.model.slp,'ch'):
            self.irf(method='ch')
        if hasattr(self.model.slp,'iv'):
            self.irf(method='iv')


# df=pd.read_csv('./testdata.csv',index_col=None)
# df

# c=lpm(df.values[:,0:7],nL=1,nH=1)
# # c.model.nT
# # c.set_sample()
# # c.data
# # c.model.parameters.B[0]
# c.irf()
# # c.irfs.rd
# c.model.residuals.rd

# In[211]:


class sfm(object):
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
        eigVal, eigVec = np.linalg.eig(S_xx)
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
                axes[0].set_title('Bai & Ng Criterion');
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


# In[ ]:




