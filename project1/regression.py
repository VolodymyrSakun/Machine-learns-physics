# Cp = p + (MSEp - MSEall) / MSEall * (n - p)
# p - number of variables
# n - number of observations
# adjusted_R2 = 1 - (1-R2)*(y.size-1)/(y.size-x.shape[1]-1)
"""
A=[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,0,0]]
B = [1,1,1,1,1]
W = [1,2,3,4,5]
W = np.sqrt(np.diag(W))
Aw = np.dot(W,A)
Bw = np.dot(B,W)
X = np.linalg.lstsq(Aw, Bw)

OLS
from scipy import linalg
w1 = linalg.lstsq(X, y)[0]
print w1

WLS
weights = np.linspace(1, 2, N)
Xw = X * np.sqrt(weights)[:, None]
yw = y * np.sqrt(weights)
print linalg.lstsq(Xw, yw)[0]
"""

from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import enet_path
import matplotlib.pyplot as plt
import statsmodels.regression.linear_model as sm

def Standardize(x):
    x_scale = StandardScaler(copy=True, with_mean=True, with_std=True)
    x_scale.fit(x)
    x_std = x_scale.transform(x)
    variances = x_scale.var_
    return x_std , variances.reshape(-1)
    
def jac_exp_simple(c, x_exp, x_lin, y):
    if x_exp is None:
        len_exp = 0
    else:
        len_exp = x_exp.shape[1]
        Size = x_exp.shape[0]
    if x_lin is None:
        len_lin = 0
    else:            
        len_lin = x_lin.shape[1]
        Size = x_lin.shape[0]
    if type(c) is list:
        len_vars = len(c)
    else:
        len_vars = c.size
    J = np.empty((Size, len_vars))    
    for i in range(0, len_exp, 1):
        J[:, 2*i] = np.exp(c[2*i+1] * x_exp[:, i])# coeff * exp
    for i in range(1, len_exp, 1):
        J[:, 2*i+1] = c[2*i] * x_exp[:, i] * np.exp(c[2*i+1] * x_exp[:, i]) # e^coeff
    for i in range(0, len_lin, 1):
        J[:, i+2*len_exp] = x_lin[:, i]   
    return J

# beta*R**(-n)*e**(alpha*R)
# c - vector c[0] - alpha, c[1] - beta; before linear
# c [i] - linear coeff; last
# x_exp - 2D array of tuples (R, R**(-n))
def jac_exp(c, x_expD, x_expDn, x_lin, y):
    if x_expD is None or x_expDn is None:
        len_exp = 0 # number of exponential terms
    else:
        len_exp = x_expD.shape[1]
        Size = x_expD.shape[0]
    if x_lin is None:
        len_lin = 0 # number of linear terms
    else:            
        len_lin = x_lin.shape[1]
        Size = x_lin.shape[0]
    if type(c) is list:
        len_vars = len(c)
    else:
        len_vars = c.size # total number of fitting coeficients
    J = np.empty((Size, len_vars))   
    for i in range(0, len_exp, 1):
        J[:, 2*i] = c[2*i+1] * x_expDn[:, i] * x_expD[:, i] * np.exp(c[2*i] * x_expD[:, i]) # e^coeff [alpha]
    #   d/d_alpha = beta     * R**(-n)        * R              * e**(   alpha  * R)
    for i in range(0, len_exp, 1):
        J[:, 2*i+1] = x_expDn[:, i] * np.exp(c[2*i] * x_expD[:, i])# R**(-n)*e**(alpha*R)
    #   d/d_beta    = R**(-n)        * e**(   alpha  * R) 
    for i in range(0, len_lin, 1):
        J[:, i+2*len_exp] = x_lin[:, i]   
    #   d/d_gamma         = R**(-m)  
    return J

def residual_exp(c, x_expD, x_expDn, x_lin, y):
    y_pred = predict_exp(c, x_expD, x_expDn, x_lin, y)
    residuals = np.zeros(shape=(y.size), dtype=float)
    residuals[:] = y[:] - y_pred[:]
    return residuals

def predict_exp(c, x_expD, x_expDn, x_lin, y):
    if x_expD is None or x_expDn is None:
        len_exp = 0
    else:
        len_exp = x_expD.shape[1]
    if x_lin is None:
        len_lin = 0
    else:            
        len_lin = x_lin.shape[1]
    y_pred = np.zeros(shape=(y.size), dtype=float)
    for i in range(0, len_exp, 1):
        y_pred[:] += c[2*i+1] * x_expDn[:, i] * np.exp(c[2*i] * x_expD[:, i])
    #   E         += beta     * R**(-n)        * e**(   alpha  * R)    
    for i in range(0, len_lin, 1):
        y_pred[:] += c[i+2*len_exp] * x_lin[:, i] 
    #   E         += gamma          * R**(-m)
    return y_pred

def predict_exp_simple(c, x_exp, x_lin, y):
    if x_exp is None:
        len_exp = 0
    else:
        len_exp = x_exp.shape[1]
    if x_lin is None:
        len_lin = 0
    else:            
        len_lin = x_lin.shape[1]
    Size = y.size
    if type(c) is list:
        len_vars = len(c)
    else:
        len_vars = c.size # total number of fitting coeficients
    y_pred = np.zeros(shape=(Size), dtype=float)
    if len_exp != 0:
        vars_exp = np.zeros(shape=(2*len_exp), dtype=float)
        for i in range(0, 2*len_exp, 1):
            vars_exp[i] = c[i]     
        for i in range(0, len_exp, 1):
            y_pred += vars_exp[2*i]*np.exp(vars_exp[2*i+1]*x_exp[:, i]) 
    if len_lin != 0:                
        vars_lin = np.zeros(shape=(len_lin), dtype=float)
        for i in range(2*len_exp, len_vars, 1):
            vars_lin[i-2*len_exp] = c[i]                
        for i in range(0, len_lin, 1):
            y_pred += vars_lin[i]*x_lin[:, i]
    return y_pred

def get_Mallow(idx_nonlin, idx_lin, x_nonlin, x_lin, y, MSEall=None, MSEp=None): 
    Cp = None
    n = y.size # number of observations
    if (idx_nonlin is None) or (x_nonlin is None): # proceed only linear features
        p = len(idx_lin) # number of features from linear set
        if MSEall is None: # calculate MSE for all features using linear regression only
            lr = LR(normalize=True, LinearSolver='sklearn')
            lr.fit(x_lin, y)
            MSEall = lr.MSE_Train            
        if MSEp is None: # calculate MSE for reduced set of features using linear regression only
            results = fit_linear(idx_lin, x_lin, y, normalize=True, LinearSolver='sklearn')      
            MSEp = results['MSE Train']
    else: # proceed non-linear regression
        if (idx_lin is None) or (x_lin is None): # no linear features
            p = len(idx_nonlin)
        else:
            p = len(idx_nonlin) + len(idx_lin) # both features
        if MSEall is None: # calculate MSE for all features using non-linear regression        
            r = expRegression()
            r.fit(x_nonlin, x_lin, y, jac=None, c0=None)
            if r.MSE_Train is None:
                return None, None, None
            MSEall = r.MSE_Train
        if MSEp is None:
            results = fit_exp(idx_nonlin, idx_lin, x_nonlin, x_lin, y,\
                NonlinearFunction='exp', jac=None, c0=None)    
            MSEp = results['MSE Train']
            if MSEp is None:
                return None, None, None
    Cp = p + ((MSEp - MSEall) / MSEall) * (n - p)
    return Cp, MSEall, MSEp

def compute_Mallow(nObservations, nFeatures, MSEall, MSEp):
    Cp = nFeatures + ((MSEp - MSEall) / MSEall) * (nObservations - nFeatures)
    return Cp

def fit_linear(idx, x_train, y_train, x_test=None, y_test=None,\
        normalize=True, LinearSolver='sklearn', cond=1e-20, lapack_driver='gelsy'):
# solver = 'sklearn'
# solver = 'scipy'   
# solver = 'statsmodels'  
    if idx == [] or idx is None:
        print('Crash fit_linear')
        return
    Size_train = x_train.shape[0] # number of observations
    size = len(idx) # number of variables
    x_sel_train = np.zeros(shape=(Size_train, size), dtype=float)
# creating selected features array
    for i in range(0, size, 1):
        x_sel_train[:, i] = x_train[:, idx[i]] # copy selected features from initial set
    if (x_test is not None) and (y_test is not None):
        Size_test = x_test.shape[0] # number of observations
        x_sel_test = np.zeros(shape=(Size_test, size), dtype=float)      
        for i in range(0, size, 1):
            x_sel_test[:, i] = x_test[:, idx[i]] # copy selected features from initial set
    else:
        x_sel_test = None
    lr = LR(normalize=normalize, LinearSolver=LinearSolver, cond=cond, lapack_driver=lapack_driver)        
    lr.fit(x_sel_train, y_train, x_test=x_sel_test, y_test=y_test)
    if lr.p_Values is None:
        a = np.zeros(shape=(1, len(idx)), dtype=float)
        Table = pd.DataFrame(a, index=['Coefficient'], columns=idx, dtype=float)
    else:
        a = np.zeros(shape=(2, len(idx)), dtype=float)
        Table = pd.DataFrame(a, index=['Coefficient', 'p-Value'], columns=idx, dtype=float)        
    for i in range(0, Table.shape[1], 1):
        Table.iloc[0, i] = lr.coef_[i]
        if Table.shape[0] == 2:
            Table.iloc[1, i] = lr.p_Values[i]
    return {'Coefficients': Table, 'MSE Train': lr.MSE_Train,\
            'RMSE Train': lr.RMSE_Train, 'R2 Train': lr.R2_Train,\
            'R2 Adjusted Train': lr.R2Adj_Train,\
            'MSE Test': lr.MSE_Test, 'RMSE Test': lr.RMSE_Test,\
            'R2 Test': lr.R2_Test, 'R2 Adjusted Test': lr.R2Adj_Test}

def fit_exp(idx_exp, idx_lin, x_expD_train, x_expDn_train, x_lin_train, y_train,\
        x_expD_test=None, x_expDn_test=None, x_lin_test=None, y_test=None,\
        jac=None, c0=None, verbose=False):
    if idx_exp is None or idx_exp == [] or x_expD_train is None or x_expDn_train is None: # no exponential features
        size_exp = 0
        x_sel_expD_train = None
        x_sel_expDn_train = None
        x_sel_expD_test = None
        x_sel_expDn_test = None
    else:
        Size_train = x_expD_train.shape[0] # number of observations training set        
        size_exp = len(idx_exp) # number of variables
        x_sel_expD_train = np.zeros(shape=(Size_train, size_exp), dtype=float)
        x_sel_expDn_train = np.zeros(shape=(Size_train, size_exp), dtype=float)
# creating selected features array
        for i in range(0, size_exp, 1):
            x_sel_expD_train[:, i] = x_expD_train[:, idx_exp[i]] # copy selected features from initial set
            x_sel_expDn_train[:, i] = x_expDn_train[:, idx_exp[i]] # copy selected features from initial set        
        if (x_expD_test is not None) and (x_expDn_test is not None) and (y_test is not None): # test set exists
            Size_test = x_expD_test.shape[0]
            x_sel_expD_test = np.zeros(shape=(Size_test, size_exp), dtype=float)
            x_sel_expDn_test = np.zeros(shape=(Size_test, size_exp), dtype=float)
            for i in range(0, size_exp, 1):
                x_sel_expD_test[:, i] = x_expD_test[:, idx_exp[i]] # copy selected features from initial set
                x_sel_expDn_test[:, i] = x_expDn_test[:, idx_exp[i]] # copy selected features from initial set
        else:
            x_sel_expD_test = None
            x_sel_expDn_test = None
    if idx_lin is None or idx_lin == [] or x_lin_train is None:
        size_lin = 0
        x_sel_lin_train = None   
        x_sel_lin_test = None   
    else:         
        Size_train = x_lin_train.shape[0] # number of observations training set        
        size_lin = len(idx_lin) # number of variables
        x_sel_lin_train = np.zeros(shape=(Size_train, size_lin), dtype=float)
# creating selected features array
        for i in range(0, size_lin, 1):
            x_sel_lin_train[:, i] = x_lin_train[:, idx_lin[i]] # copy selected features from initial set
        if (x_lin_test is not None) and (y_test is not None): # test set exists
            Size_test = x_lin_test.shape[0] # number of observations test set
            x_sel_lin_test = np.zeros(shape=(Size_test, size_lin), dtype=float)
            for i in range(0, size_lin, 1):
                x_sel_lin_test[:, i] = x_lin_test[:, idx_lin[i]] # copy selected features from initial set
        else:
            x_sel_lin_test = None
    non_lr = expRegression(verbose=verbose)  
    non_lr.fit(x_sel_expD_train, x_sel_expDn_train, x_sel_lin_train, y_train,\
        x_expD_test=x_sel_expD_test, x_expDn_test=x_sel_expDn_test,\
        x_lin_test=x_sel_lin_test, y_test=y_test, jac=jac, c0=c0)
    if size_exp != 0:
#        exp_array = np.zeros(shape=(1, len(idx_exp)), dtype=float)
        exp_array = [[(0, 0) for j in range(0, len(idx_exp), 1)] for i in [0]]
        Table_exp = pd.DataFrame(exp_array, index=['Coefficient'], columns=idx_exp)
        for i in range(0, Table_exp.shape[1], 1):
            Table_exp.iloc[0, i] = non_lr.coef_exp[i]
    else:
        Table_exp = None        
    if size_lin != 0:
        lin_array = np.zeros(shape=(1, len(idx_lin)), dtype=float)
        Table_lin = pd.DataFrame(lin_array, index=['Coefficient'], columns=idx_lin, dtype=float)
        for i in range(0, Table_lin.shape[1], 1):
            Table_lin.iloc[0, i] = non_lr.coef_lin[i]
    else:
        Table_lin = None        
    return {'Fit Result': non_lr.fit_result,'Success': non_lr.success,\
            'Coefficients exponential': Table_exp,\
            'Coefficients linear': Table_lin,'MSE Train': non_lr.MSE_Train,\
            'RMSE Train': non_lr.RMSE_Train,'R2 Train': non_lr.R2_Train,\
            'R2 Adjusted Train': non_lr.R2Adj_Train,'MSE Test': non_lr.MSE_Test,\
            'RMSE Test': non_lr.RMSE_Test, 'R2 Test': non_lr.R2_Test,\
            'R2 Adjusted Test': non_lr.R2Adj_Test}

class LR(dict):

    def __init__(self, normalize=True, LinearSolver='sklearn', cond=1e-20, lapack_driver='gelsy'):
        self.normalize = True
        self.LinearSolver = LinearSolver # 'sklearn', 'scipy', 'statsmodels'
        self.cond = cond # for scipy solver
        self.lapack_driver = lapack_driver # 'gelsd', 'gelsy', 'gelss', for scipy solver
        self.p_Values = None # if solver is statsmodels
        self.coef_ = None
        self.var = None # if need to be normalized
        self.MSE_Train=None    
        self.RMSE_Train=None
        self.R2_Train=None
        self.R2Adj_Train=None
        self.MSE_Test=None
        self.RMSE_Test=None
        self.R2_Test=None
        self.R2Adj_Test=None
        return

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 
    
    def fit(self, x_train, y_train, x_test=None, y_test=None):
        if self.LinearSolver == 'sklearn': # 
            lr = LinearRegression(fit_intercept=False, normalize=self.normalize,\
                                  copy_X=True, n_jobs=1)
            lr.fit(x_train, y_train)
            self.coef_ = lr.coef_.reshape(-1)
        else:
            if self.normalize: # normalize if assigned
                x_std, self.var = Standardize(x_train)
            else:
                x_std = x_train     
            if self.LinearSolver == 'scipy':
                coef, _, _, _ = lstsq(x_std, y_train, cond=self.cond, overwrite_a=False,\
                    overwrite_b=False, check_finite=False, lapack_driver=self.lapack_driver)
            elif self.LinearSolver == 'statsmodels':
                ols = sm.OLS(endog = y_train, exog = x_std, hasconst = False).fit()
                self.p_Values = ols.pvalues
                coef = ols.params
            else:
                print('Wrong solver')
                return False
            coef = coef.reshape(-1)
            if self.normalize:
                self.coef_ = np.zeros(shape=(len(coef)), dtype=float)
                self.coef_[:] = coef[:] / np.sqrt(self.var[:])                    
            else:
                self.coef_ = coef
                
# calculating scores                
        y_pred = self.predict(x_train)
        self.MSE_Train = skm.mean_squared_error(y_train, y_pred)
        self.RMSE_Train = np.sqrt(self.MSE_Train)
        self.R2_Train = skm.r2_score(y_train, y_pred)
        self.R2Adj_Train = 1 - (1-self.R2_Train)*(y_train.size-1)/\
            (y_train.size-x_train.shape[1]-1)                
        if (x_test is not None) and (y_test is not None):
            y_pred = self.predict(x_test)
            self.MSE_Test = skm.mean_squared_error(y_test, y_pred)
            self.RMSE_Test = np.sqrt(self.MSE_Test)
            self.R2_Test = skm.r2_score(y_test, y_pred)
            self.R2Adj_Test = 1 - (1-self.R2_Test)*(y_test.size-1)/\
                (y_test.size-x_test.shape[1]-1)
        return
    
    def predict(self, x):
        y = np.dot(x, self.coef_)
        return y.reshape(-1)
    
class expRegression(dict):
    
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.coef_exp=None
        self.coef_lin=None
        self.MSE_Train=None
        self.RMSE_Train=None
        self.R2_Train=None
        self.R2Adj_Train=None
        self.Mallow_Train=None
        self.MSE_Test=None
        self.RMSE_Test=None
        self.R2_Test=None
        self.R2Adj_Test=None
        self.Mallow_Test=None
        self.fit_result=None
        self.success=None
        self.jac=None 
        return    

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 

    def fit(self, x_expD_train, x_expDn_train, x_lin_train, y_train, x_expD_test=None,\
            x_expDn_test=None, x_lin_test=None, y_test=None, jac=None, c0=None): 
        residual = residual_exp
        predict = predict_exp
        if jac is None:
            jac = '3-point'
        elif jac == 'exp':
            jac = jac_exp
        if x_expD_train is None or x_expDn_train is None:
            size_exp = 0
        else:
            size_exp = x_expD_train.shape[1]
        if x_lin_train is None:
            size_lin = 0   
        else:
            size_lin = x_lin_train.shape[1]            
        if (size_exp + size_lin) == 0:
            return
        if c0 is None: # start from 0
            c0 = np.zeros(shape=(2*size_exp + size_lin))
        results = least_squares(residual, c0, jac=jac, method='trf',\
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,\
            diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,\
            max_nfev=None, verbose=self.verbose, args=(x_expD_train, x_expDn_train, x_lin_train, y_train))
        if not results.success: # try -1
            c0 = np.ones(shape=(2*size_exp + size_lin))
            c0[:] = -1 * c0[:]
            results = least_squares(residual, c0, jac=jac, method='trf',\
                ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,\
                diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,\
                max_nfev=None, verbose=self.verbose, args=(x_expD_train, x_expDn_train, x_lin_train, y_train))
            if (not results.success) and (jac != '3-point'): # not successfull with user provided jacobian
# try jacobian estimation
                c0 = np.zeros(shape=(2*size_exp + size_lin))
                results = least_squares(residual, c0, jac='3-point', method='trf',\
                    ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,\
                    diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,\
                    max_nfev=None, verbose=self.verbose, args=(x_expD_train, x_expDn_train, x_lin_train, y_train))
                if not results.success:# try ones and 3-point estimation
                    c0 = np.ones(shape=(2*size_exp + size_lin))
                    results = least_squares(residual, c0, jac='3-point', method='trf',\
                        ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,\
                        diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,\
                        max_nfev=None, verbose=self.verbose, args=(x_expD_train, x_expDn_train, x_lin_train, y_train))
                    if not results.success:
                        self.success = False                
                        return            
        coef = results.x.reshape(-1)
        self.fit_result = results.status
        self.success = True
        y_pred = predict(coef, x_expD_train, x_expDn_train, x_lin_train, y_train)
        self.MSE_Train = skm.mean_squared_error(y_train, y_pred)
        self.RMSE_Train = np.sqrt(self.MSE_Train)
        self.R2_Train = skm.r2_score(y_train, y_pred)
        self.R2Adj_Train = 1 - (1-self.R2_Train)*(y_train.size-1)/(y_train.size-size_exp-size_lin-1)
        if (x_expD_test is not None) or (x_expDn_test is not None) or (x_lin_test is not None):
            y_pred = predict(coef, x_expD_test, x_expDn_test, x_lin_test, y_test)
            self.MSE_Test = skm.mean_squared_error(y_test, y_pred)
            self.RMSE_Test = np.sqrt(self.MSE_Test)
            self.R2_Test = skm.r2_score(y_test, y_pred)
            self.R2Adj_Test = 1 - (1-self.R2_Test)*(y_test.size-1)/(y_test.size-size_exp-size_lin-1)        
# append tuples of exponential coefficients (alpha, beta)
# alpha = power of exponent, beta = coefficiant in front of exponent
        if size_exp != 0:
            self.coef_exp = []
            for i in range(0, size_exp, 1): 
                self.coef_exp.append((coef[i*2], coef[i*2+1])) 
        if size_lin != 0:
            self.coef_lin = []
            for i in range(0, size_lin, 1):
                self.coef_lin.append(coef[i+2*size_exp])
        return

class ENet(dict):
        
    def __init__(self, L1=0.7, eps=1e-3, nAlphas=100, alphas=None, random_state=None):
        self.idx = None
        self.L1 = L1
        self.eps = eps
        self.nAlphas = nAlphas
        self.random_state = random_state
        self.alpha = None
        self.alphas = alphas
        self.var=None
        self.coefs=None
        self.mse_list=None
        self.Cp_list=None
        self.nonzero_count_list=None            

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 
    
    def fit(self, x, y, VIP_idx=None, Criterion='Mallow', max_iter=10000, tol=0.0001,\
            cv=None, n_jobs=1, selection='random', verbose=True, normalize=True):
        if normalize:
            x_std, self.var = Standardize(x)
        else:
            x_std = x
        if VIP_idx is None:
            VIP_idx = []
        if (Criterion == 'CV'):
            enet_cv = ElasticNetCV(l1_ratio=self.L1, eps=self.eps, n_alphas=self.nAlphas,\
                alphas=self.alphas, fit_intercept=False,normalize=False, precompute='auto',\
                max_iter=max_iter, tol=tol, cv=cv, copy_X=True, verbose=verbose,\
                n_jobs=n_jobs, positive=False, random_state=self.random_state, selection=selection)
            enet_cv.fit(x_std, y)
            self.coefs = enet_cv.coef_
            self.alpha = enet_cv.alpha_
            idx = []
            for j in range(0, len(self.coefs), 1):
                if abs(self.coefs[j]) > tol:
                    idx.append(j)
            for i in VIP_idx: # add missing VIP features 
                if i not in idx:
                    idx.append(i)
            self.idx = idx                 
            return 

        self.alphas , self.coefs, _ = enet_path(x_std, y, l1_ratio=self.L1,\
            eps=self.eps, n_alphas=self.nAlphas, alphas=self.alphas, precompute='auto',\
            Xy=None, copy_X=True, coef_init=None, verbose=verbose, return_n_iter=False,\
            positive=False, check_input=True)
        self.Cp_list = []
        self.mse_list = []
        self.nonzero_count_list = []
        for i in range(0, self.coefs.shape[1], 1): # columns
            nonzero_idx = []
            for j in range(0, self.coefs.shape[0], 1):
                if abs(self.coefs[j, i]) > tol:
                    nonzero_idx.append(j)
            if len(nonzero_idx) == 0:
                self.Cp_list.append(1e+100)
                self.mse_list.append(1e+100)
                continue
            self.nonzero_count_list.append(len(nonzero_idx))
            Cp, MSEall, MSEp = get_Mallow(idx_nonlin=None, idx_lin=nonzero_idx,\
                x_nonlin=None, x_lin=x_std, y=y, MSEall=None, MSEp=None)
            self.Cp_list.append(Cp)
            self.mse_list.append(MSEp)
        if Criterion == 'MSE':
            idx = np.argmin(self.mse_list)
        if Criterion == 'Mallow':
            idx = np.argmin(self.Cp_list)
        self.alpha = self.alphas[idx]
        nonzero_idx = []
        for j in range(0, self.coefs.shape[0], 1):
            if abs(self.coefs[j, idx]) > tol:
                nonzero_idx.append(j)        
        self.idx = nonzero_idx
        return
    
    def plot_path(self, fig_number, F_ENet=None, FigSize=(4,3), FileFormat='eps', Resolution=100):
        nonzero = []
        for i in range(0, self.coefs.shape[1], 1):
            nonzero_count = np.count_nonzero(self.coefs[:, i])
            nonzero.append(nonzero_count)
        fig = plt.figure(fig_number, figsize = FigSize)
        plt.subplot(211)
        subplot11 = plt.gca()
        plt.plot(self.alphas, nonzero, ':')
        subplot11.set_xscale('log')
        plt.xlabel('Alphas')
        plt.ylabel('Number of nonzero coefficients')
        plt.title('Elastic Net Path')
        plt.subplot(212)
        subplot21 = plt.gca()
        plt.plot(self.alphas, self.mse_list, ':')
        subplot21.set_xscale('log')
        subplot21.set_yscale('log')
        plt.xlabel('Alphas')
        plt.ylabel('MSE')
        plt.title('Mean squared error vs. regularization strength')
        plt.show()
        if F_ENet is not None:
            F = '{}{}{}'.format(F_ENet, '.', FileFormat)
            plt.savefig(F, bbox_inches='tight', format=FileFormat, dpi=Resolution)
            plt.close(fig) 
        return
        
    