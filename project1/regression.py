# Cp = p + (MSEp - MSEall) / MSEall * (n - p)
# p - number of variables
# n - number of observations
# adjusted_R2 = 1 - (1-R2)*(y.size-1)/(y.size-x.shape[1]-1)

from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
import numpy as np
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
    return x_std , variances
    
def jac_exp(c, x_exp, x_lin, y):
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
        J[:, 2*i] = np.exp(c[2*i+1] * x_exp[:, i])
    for i in range(1, len_exp, 1):
        J[:, 2*i+1] = c[2*i] * x_exp[:, i] * np.exp(c[2*i+1] * x_exp[:, i])
    for i in range(0, len_lin, 1):
        J[:, i+2*len_exp] = x_lin[:, i]   
    return J

def residual_exp(c, x_exp, x_lin, y):
    y_pred = predict_exp(c, x_exp, x_lin, y)
    residuals = y - y_pred
    if residuals is None:
        print('Crash: residual_exp')
    return residuals

def predict_exp(c, x_exp, x_lin, y):
    if x_exp is None:
        len_exp = 0
    else:
        len_exp = x_exp.shape[1]
    if x_lin is None:
        len_lin = 0
    else:            
        len_lin = x_lin.shape[1]
    Size = y.size
    len_vars = len(c)
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
            MSEall = lr.MSE            
        if MSEp is None: # calculate MSE for reduced set of features using linear regression only
            results = fit_linear(idx_lin, x_lin, y, normalize=True, LinearSolver='sklearn')      
            MSEp = results['MSE Train']
    else: # proceed non-linear regression
        if (idx_lin is None) or (x_lin is None): # no linear features
            p = len(idx_nonlin)
        else:
            p = len(idx_nonlin) + len(idx_lin) # both features
        if MSEall is None: # calculate MSE for all features using non-linear regression        
            r = nonLR(NonlinearFunction='exp')
            r.fit(x_nonlin, x_lin, y, jac=None, c0=None)
            if r.MSE is None:
                return None, None, None
            MSEall = r.MSE
        if MSEp is None:
            results = fit_nonlinear(idx_nonlin, idx_lin, x_nonlin, x_lin, y,\
                NonlinearFunction='exp', jac=None, c0=None)    
            MSEp = results['MSE Train']
            if MSEp is None:
                return None, None, None
    Cp = p + ((MSEp - MSEall) / MSEall) * (n - p)
    return Cp, MSEall, MSEp

def compute_Mallow(nObservations, nFeatures, MSEall, MSEp):
    Cp = nFeatures + ((MSEp - MSEall) / MSEall) * (nObservations - nFeatures)
    return Cp

def fit_linear(idx, x_train, y_train, x_test=None, y_test=None, MSEall_train=None, MSEall_test=None,\
        normalize=False, LinearSolver='sklearn', cond=1e-20, lapack_driver='gelsy'):
# solver = 'sklearn'
# solver = 'scipy'   
# solver = 'statsmodels'    
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
    if (MSEall_train is None):
        lr.fit(x_train, y_train, x_test=x_test, y_test=y_test)
        MSEall_train = lr.MSE_Train
        MSEall_test = lr.MSE_Test
    lr.fit(x_sel_train, y_train, x_test=x_sel_test, y_test=y_test)
    if (MSEall_train is not None) and (lr.MSE_Train is not None):
        Mallow_train = compute_Mallow(Size_train, size, MSEall_train, lr.MSE_Train)
    else:
        Mallow_train = None
    if (MSEall_test is not None) and (lr.MSE_Test is not None):
        Mallow_test = compute_Mallow(Size_train, size, MSEall_test, lr.MSE_Test)
    else:
        Mallow_test = None
    return {'Coefficients': lr.coef_, 'p-Values': lr.p_Values, 'MSE Train': lr.MSE_Train,\
            'RMSE Train': lr.RMSE_Train, 'R2 Train': lr.R2_Train, 'R2 Adjusted Train': lr.R2Adj_Train,\
            'Mallow Train': Mallow_train, 'MSE Test': lr.MSE_Test, 'RMSE Test': lr.RMSE_Test,\
            'R2 Test': lr.R2_Test, 'R2 Adjusted Test': lr.R2Adj_Test, 'Mallow Test': Mallow_test}

def fit_nonlinear(idx_nonlin, idx_lin, x_nonlin_train, x_lin_train, y_train,\
        x_nonlin_test=None, x_lin_test=None, y_test=None, NonlinearFunction='exp',\
        jac=None, c0=None):
    if NonlinearFunction != 'exp':
        return None, None, None, None
    if idx_nonlin is None:
        size_nonlin = 0
        x_sel_nonlin_train = None
        x_sel_nonlin_test = None
    else:
        Size_train = x_nonlin_train.shape[0] # number of observations training set        
        size_nonlin = len(idx_nonlin) # number of variables
        x_sel_nonlin_train = np.zeros(shape=(Size_train, size_nonlin), dtype=float)
# creating selected features array
        for i in range(0, size_nonlin, 1):
            x_sel_nonlin_train[:, i] = x_nonlin_train[:, idx_nonlin[i]] # copy selected features from initial set
        if (x_nonlin_test is not None) and (y_test is not None):
            Size_test = x_nonlin_test.shape[0]
            x_sel_nonlin_test = np.zeros(shape=(Size_test, size_nonlin), dtype=float)
            for i in range(0, size_nonlin, 1):
                x_sel_nonlin_test[:, i] = x_nonlin_test[:, idx_nonlin[i]] # copy selected features from initial set
        else:
            x_sel_nonlin_test = None
    if idx_lin is None:
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
        if (x_lin_test is not None) and (y_test is not None): 
            Size_test = x_lin_test.shape[0] # number of observations test set
            x_sel_lin_test = np.zeros(shape=(Size_test, size_lin), dtype=float)
            for i in range(0, size_lin, 1):
                x_sel_lin_test[:, i] = x_lin_test[:, idx_lin[i]] # copy selected features from initial set
        else:
            x_sel_lin_test = None
    lr = nonLR(NonlinearFunction=NonlinearFunction)  
    lr.fit(x_sel_nonlin_train, x_sel_lin_train, y_train, x_exp_test=x_sel_nonlin_test,\
        x_lin_test=x_sel_lin_test, y_test=y_test, jac=jac, c0=c0)
    return {'Fit Result': lr.fit_result, 'Success': lr.success, 'Coefficients': lr.coef_,\
            'MSE Train': lr.MSE_Train, 'RMSE Train': lr.RMSE_Train, 'R2 Train': lr.R2_Train,\
            'R2 Adjusted Train': lr.R2Adj_Train, 'Mallow Train': lr.Mallow_Train,\
            'MSE Test': lr.MSE_Test, 'RMSE Test': lr.RMSE_Test, 'R2 Test': lr.R2_Test,\
            'R2 Adjusted Test': lr.R2Adj_Test, 'Mallow Test': lr.Mallow_Test}

class LR:
    normalize=False
    coef_=None
    var=None # if normalize
    p_Values=None # if solver is statsmodels
    MSE_Train=None
    RMSE_Train=None
    R2_Train=None
    R2Adj_Train=None
    MSE_Test=None
    RMSE_Test=None
    R2_Test=None
    R2Adj_Test=None
    LinearSolver='sklearn' # 'sklearn', 'scipy', 'statsmodels'
    cond=1e-20 # for scipy solver
    lapack_driver='gelsy' # 'gelsd', 'gelsy', 'gelss', for scipy solver

    def __init__(self, normalize=False, LinearSolver='sklearn', cond=1e-20, lapack_driver='gelsy'):
        self.normalize = normalize
        self.LinearSolver = LinearSolver
        self.cond = cond
        self.lapack_driver = lapack_driver
        self.p_Values = None
        self.coef_ = None
        return
    
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
            if self.LinearSolver == 'statsmodels':
                ols = sm.OLS(endog = y_train, exog = x_std, hasconst = False).fit()
                self.p_Values = ols.pvalues
                coef = ols.params
            if self.normalize:
                self.coef_ = np.zeros(shape=(len(coef)), dtype=float)
                for i in range(0, len(coef), 1):
                    self.coef_[i] = coef[i] / np.sqrt(self.var[i])
            else:
                self.coef_ = coef.reshape(-1)
                
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
        return y
    
class nonLR:
    coef_=None
    MSE_Train=None
    RMSE_Train=None
    R2_Train=None
    R2Adj_Train=None
    Mallow_Train=None
    MSE_Test=None
    RMSE_Test=None
    R2_Test=None
    R2Adj_Test=None
    Mallow_Test=None
    fit_result=None
    NonlinearFunction='exp'
    success=None
    verbose=0
    jac=None # 'exp' or None
    
    def __init__(self, NonlinearFunction='exp', verbose=0):
        self.NonlinearFunction = NonlinearFunction
        self.verbose = verbose
        return    

    def fit(self, x_exp_train, x_lin_train, y_train, x_exp_test=None,\
            x_lin_test=None, y_test=None, jac=None, c0=None): 
        if self.NonlinearFunction != 'exp': # now works only with exp function
            return
        residual = residual_exp
        predict = predict_exp
        if jac is None:
            jac = '3-point'
        elif jac == 'exp':
            jac = jac_exp
        if x_exp_train is None:
            size_exp = 0
        else:
            size_exp = x_exp_train.shape[1]
        if x_lin_train is None:
            size_lin = 0   
        else:
            size_lin = x_lin_train.shape[1]            
        if (size_exp + size_lin) == 0:
            return
        if c0 is None: # start from zeros
            c0 = np.zeros(shape=(2*size_exp + size_lin)) # each exp feature requires 2 comstants 
        results = least_squares(residual, c0, jac=jac, method='trf',\
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,\
            diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,\
            max_nfev=None, verbose=self.verbose, args=(x_exp_train, x_lin_train, y_train))
        if not results.success: # try ones
            c0 = np.ones(shape=(2*size_exp + size_lin))
            results = least_squares(residual, c0, jac=jac, method='trf',\
                ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,\
                diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,\
                max_nfev=None, verbose=self.verbose, args=(x_exp_train, x_lin_train, y_train))
            if (not results.success) and (jac != '3-point'): # not successfull with user provided jacobian
# try jacobian estimation
                c0 = np.zeros(shape=(2*size_exp + size_lin))
                results = least_squares(residual, c0, jac='3-point', method='trf',\
                    ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,\
                    diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,\
                    max_nfev=None, verbose=self.verbose, args=(x_exp_train, x_lin_train, y_train))
                if not results.success:# try ones and 3-point estimation
                    c0 = np.ones(shape=(2*size_exp + size_lin))
                    results = least_squares(residual, c0, jac='3-point', method='trf',\
                        ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,\
                        diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,\
                        max_nfev=None, verbose=self.verbose, args=(x_exp_train, x_lin_train, y_train))
                    if not results.success:
                        self.success = False                
                        return            
        self.coef_ = results.x
        self.fit_result = results.status
        self.success = True
        y_pred = predict(self.coef_, x_exp_train, x_lin_train, y_train)
        self.MSE_Train = skm.mean_squared_error(y_train, y_pred)
        self.RMSE_Train = np.sqrt(self.MSE_Train)
        self.R2_Train = skm.r2_score(y_train, y_pred)
        self.R2Adj_Train = 1 - (1-self.R2_Train)*(y_train.size-1)/(y_train.size-size_exp-size_lin-1)
        if (x_exp_test is not None) or (x_lin_test is not None):
            y_pred = predict(self.coef_, x_exp_test, x_lin_test, y_test)
            self.MSE_Test = skm.mean_squared_error(y_test, y_pred)
            self.RMSE_Test = np.sqrt(self.MSE_Test)
            self.R2_Test = skm.r2_score(y_test, y_pred)
            self.R2Adj_Test = 1 - (1-self.R2_Test)*(y_test.size-1)/(y_test.size-size_exp-size_lin-1)
        return

class ENet:
    idx = None
    F_ENet = 'ENet path.png'
    L1 = 0.7
    eps = 1e-3
    nAlphas = 100
    alpha = None
    alphas = None
    var=None
    random_state=None
    coefs=None
    mse_list=None
    Cp_list=None
    nonzero_count_list=None
    
    def __init__(self, L1=0.7, eps=1e-3, nAlphas=100, alphas=None, random_state=None):
        self.L1 = L1
        self.eps = eps
        self.nAlphas = nAlphas
        self.random_state = random_state
        if alphas is not None:
            self.alphas = alphas
        return
    
    def fit(self, x, y, VIP_idx=None, Criterion='CV', max_iter=10000, tol=0.0001,\
            cv=None, n_jobs=1, selection='random', verbose=True, normalize=True):
        if normalize:
            x_std, self.var = Standardize(x)
        else:
            x_std = x
        if VIP_idx is None:
            VIP_idx = []
        if (Criterion == 'CV'):
            enet_cv = ElasticNetCV(l1_ratio=self.L1, eps=self.eps, n_alphas=self.nAlphas, alphas=self.alphas, fit_intercept=False,\
                normalize=False, precompute='auto', max_iter=max_iter, tol=tol, cv=cv, copy_X=True, \
                verbose=verbose, n_jobs=n_jobs, positive=False, random_state=self.random_state, selection=selection)
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
        MSEall = None
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
                x_nonlin=None, x_lin=x_std, y=y, MSEall=MSEall, MSEp=None)
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
    
# returns list of indices of selected features
# default setting: use ElasticNetCV assigned by Method
# x_std - standardized set of features
# y_std - response with mean = 0
# Method='CV' - use ElasticNEtCV, Method='grid' - use enet_path
# MSE_threshold - treshhold for selecting min nonzero coefficients which gives fit better than MSE_threshold
# R2_threshold - treshhold for selecting min nonzero coefficients which gives fit better than R2_threshold
# L1_ratio - portion of L1 regularization. For Method='CV' can be array of floats
# eps - Length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3. Relevant if alphas is not assigned
# N_alphas - size of grid
# Alphas - array of regularization strengths
# cv - number of cross-folds. only for Method='CV'
# selection='cyclic' or 'random'. If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default. This (setting to ‘random’) often leads to significantly faster convergence especially when tol is higher than 1e-4.
# PlotPath - plot 2 graphs: Number of nonzero coefficients vs. alpha and MSE vs. alpha. Only for Method='grid'

    def plot_path(self, fig_number, FileName=None):
        nonzero = []
        for i in range(0, self.coefs.shape[1], 1):
            nonzero_count = np.count_nonzero(self.coefs[:, i])
            nonzero.append(nonzero_count)
        fig = plt.figure(fig_number, figsize = (19, 10))
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
        if FileName is not None:
            plt.savefig(FileName, bbox_inches='tight')
            plt.close(fig)
        return
        
    