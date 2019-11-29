#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 23:00:14 2019
@author: aytekm - cindyz
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.optimize as opt
import scipy.special as scsp
from scipy.stats import describe,chi2
from lib.grad import *


def read_clean_data(filename):

    ## read data
    df  = pd.read_excel('data/'+filename+'.xlsx')

    ## calculate no. of obs.
    df['N'] = df.iloc[:,2:11].notna().sum(axis=1)
    
    df['date'] = [pd.datetime(year=y,month=m,day=1) for y,m in zip(df.Year,df.Month)]
    
    df['mean'] = df.iloc[:,2:11].mean(axis=1)
    
    ## trim below 1% and above 99%
    df.iloc[:,2:11] = np.where(df.iloc[:,2:11]<0.01,0.01,np.where(df.iloc[:,2:11]>0.99,0.99,df.iloc[:,2:11]))

    return df


def plot_summarize_data(df):
    
    #plot data
    fig,ax= plt.subplots(figsize=(10,5))
    ax.plot(df.date,df['mean'],'b',label='Mean LGD per observation month')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    
    ##summarize data 
    df_desc = describe(df['mean'].dropna())
    df_summary = pd.DataFrame(columns=['Data'],index=['Mean','Std. Deviation','Min','Max','Skewness','Kurtosis','Total Number of Observations'])

    df_summary.loc['Mean'] = df_desc[2]
    df_summary.loc['Std. Deviation'] = np.sqrt(df_desc[3])
    df_summary.loc['Min'] = df_desc[1][0]
    df_summary.loc['Max'] = df_desc[1][1]
    df_summary.loc['Skewness'] = df_desc[4]
    df_summary.loc['Kurtosis'] = df_desc[5]
    df_summary.loc['Total Number of Observations'] = df.N.sum()    

    return df_summary


def PlotDist(f_t,c,s,df):

    #num. of observations
    iN = len(f_t)
    
    #mu_t
    mu_t = Sigmoid(f_t)
    
    #initialize mean estimation array
    mean_t = np.zeros((iN,))
    
    #estimate mean over time
    for i in range(iN):
        mean_t[i] = scsp.gamma(s)/scsp.gamma(s*mu_t[i]) * scsp.gamma(s*mu_t[i]+1/c)/scsp.gamma(s+1/c)
    
    #prepare data
    df_final = pd.concat([df.iloc[:,2:11],df['date']],axis=1)
    df_final = pd.melt(df_final,id_vars=['date'])
    
    #plot
    fig,ax = plt.subplots()
    ax.plot(df.date,mean_t,'b')
    ax.plot(df_final['date'],df_final['value'],'r+')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.legend().set_visible(False)

 
def Sigmoid(x):

    return np.exp(x)/(1+np.exp(x))

def ReverseSigmoid(x,k):

    return np.log(x/(k-x))

def LogBetaDist(mu_t,c,s,vY_t_i):
    return np.log(c)+scsp.gammaln(s)-scsp.gammaln(s*mu_t)-scsp.gammaln(s*(1-mu_t))+np.log(np.power(vY_t_i,c*s*mu_t-1)) + np.log(np.power((1-np.power(vY_t_i,c)),s*(1-mu_t)-1))


def WaldTest(iY,params,cov_matrix):
    
    g = lambda params: params[3]-1
    gJ = jacobian_2sided(g,params)[0]
    test_stat = (iY*g(params)*(gJ@cov_matrix@gJ.T)*g(params))
    p_val = 1 - chi2.cdf(test_stat, 1)
    
    return [test_stat,p_val]
    

def ParamTransform(vP,bShapeAsVector=False):
    vP_transformed = [vP[0],Sigmoid(vP[1]),Sigmoid(vP[2]),5*Sigmoid(vP[3]),5*Sigmoid(vP[4])]
    if (bShapeAsVector == True):
        return np.array(vP_transformed)
    else:
        return vP_transformed

def LogLikelihood(mu_t,vP,N_t,vY_t):
    
    c = vP[3]
    s = vP[4]
    ll=0
    
    #log-likelihood contribution of each observation at time t
    for i in range(N_t):
        ll = ll + LogBetaDist(mu_t,c,s,vY_t[i])

    return ll

def Derivative(mu_t,vP,N_t,vY_t):
    
    c = vP[3]
    s = vP[4]
    deriv = 0
    
    #derivative contribution of each observation at time t
    for i in range(N_t):
        deriv += -s*scsp.digamma(s*mu_t) + s*scsp.digamma(s*(1-mu_t)) + c*s*np.log(vY_t[i]) -s*np.log(1-np.power(vY_t[i],c))

    return deriv


def LL_PredictionErrorDecomposition(vP,vY,mean_overall,vN,return_f_t = False):
    
    #transform parameters
    vP = ParamTransform(vP) 
    
    #length of data
    iN = len(vY)
    
    #initialize filter
    f_1 = np.log(mean_overall/(1-mean_overall))
    f_t = np.zeros((iN,))
    f_t[0] = f_1
    
    #initialize log-likelihoods
    vLL = np.zeros((iN,))
    
    #parameters
    omega = vP[0]
    beta = vP[1]
    alpha = vP[2]

    
    for i in range(0,iN-1):
        
        N_t = vN[i]
        vY_t = vY.loc[i]
        score = 0
        
        #calculate score
        if N_t>0:
            score =  (1/N_t) * Derivative(Sigmoid(f_t[i]),vP,N_t,vY_t) *  np.exp(f_t[i])/(np.power((1+np.exp(f_t[i])),2))
 
        #update filter
        f_t[i+1] = omega + beta*f_t[i] + alpha * score 
        
        #calculate loglikelihood
        vLL[i+1] = LogLikelihood(Sigmoid(f_t[i+1]),vP,vN[i+1],vY.loc[i+1])
    
    #return filter or loglikelihood array
    if return_f_t:
        return f_t
    else:
        return vLL


def StdErrors(fun,params,vY,mean_overall,vN,transform,cov_matrix=False):
     
    #hessian and std. errors
    hes = -hessian_2sided(fun,params)
    inv_hes = np.linalg.inv(hes)
    std_errors = list(np.sqrt(np.diag(-inv_hes)))
    
    #sandwich form robust std. errors
    inv_hes_symetric = (inv_hes + inv_hes.T)/2
    mG = jacobian_2sided(LL_PredictionErrorDecomposition,params,vY,mean_overall,vN)
    
    cov_matrix_sandwich = inv_hes_symetric @ (mG.T @ mG) @ inv_hes_symetric
    
    #calculate transformed cov_matrix in case there is reparametrization
    if transform:
        mJ = jacobian_2sided(ParamTransform, params,True)
        cov_matrix_sandwich = mJ @ cov_matrix_sandwich @ mJ.T
     
    std_errors_sandwich = list(np.sqrt(np.diag(cov_matrix_sandwich)))
    
    return [std_errors_sandwich,cov_matrix_sandwich]


def EstimateParams(df):
        
    #initial params for w,beta,alpha,c,s
    vP0_t = [0.01, 0.99, 0.01, 0.99, 0.99]
    
    #re-transform initial parameters for optimization initialization
    vP0 = [vP0_t[0],ReverseSigmoid(vP0_t[1],1),ReverseSigmoid(vP0_t[2],1),ReverseSigmoid(vP0_t[3],5),ReverseSigmoid(vP0_t[4],5)]
    
    #data extraction
    vY = df.iloc[:,2:11]

    #overall sample mean for f_1 initialization
    mean_overall = vY.sum().sum() / vY.count().sum()
    
    #array of N_t's
    vN = df['N']
    
    #function to be optimized
    sumLL= lambda vP: -np.sum(LL_PredictionErrorDecomposition(vP,vY,mean_overall,vN))  
    
    res= opt.minimize(sumLL, vP0, method='BFGS', options={'disp': True, 'maxiter':250})
    
    #transformed parameters
    params = ParamTransform(res.x)
    
    #untransformed parameters
    params_untransformed = res.x    
    
    #untransformed standard errors (original)
    std_errors_untransformed = StdErrors(sumLL,params_untransformed,vY,mean_overall,vN,transform = False)[0]
    
    #transformed standard errors (reparametrized) 
    [std_errors_transformed,cov_matrix_transformed] = StdErrors(sumLL,params_untransformed,vY,mean_overall,vN,transform = True)
    
    #filters
    f_t_optimized = LL_PredictionErrorDecomposition(params_untransformed,vY,mean_overall,vN,True)
    
    #Wald Test
    [t_stat,p_value] = WaldTest(len(vY),np.array(params),cov_matrix_transformed)
    
    return [res.fun,params,list(params_untransformed),std_errors_transformed,std_errors_untransformed,f_t_optimized,t_stat,p_value]


def main():
    
    filename = 'LGD-data'
    
    df = read_clean_data(filename)
    
    print(plot_summarize_data(df))
    
    [likelihood,params,params_original,std_err,std_err_original,f_t,t_stat,p_value] = EstimateParams(df)
    print('Log-likelihood: ',-likelihood)
    print('Optimized reparametrized parameters:\nomega: ',params[0],'\nbeta: ',params[1],'\nalpha: ',params[2],'\nc: ',params[3],'\ns: ',params[4])
    print('Optimized original parameters:\nomega: ',params_original[0],'\nbeta: ',params_original[1],'\nalpha: ',params_original[2],'\nc: ',params_original[3],'\ns: ',params_original[4])
    print('Optimized reparametrized std. errors:\nomega: ',std_err[0],'\nbeta: ',std_err[1],'\nalpha: ',std_err[2],'\nc: ',std_err[3],'\ns: ',std_err[4])
    print('Optimized original std. errors:\nomega: ',std_err_original[0],'\nbeta: ',std_err_original[1],'\nalpha: ',std_err_original[2],'\nc: ',std_err_original[3],'\ns: ',std_err_original[4])
    
    if p_value > 0.05:
        print('Wald test null hypothesis cannot be rejected with test statistic of ',t_stat,' and p-value of ',p_value) 
        print('Parameter c=',params[3],' is not significantly different from 1')
    else:
        print('Wald test null hypothesis is rejected with test statistic of ',t_stat,' and p-value of ',p_value) 
        print('Parameter c=',params[3],' is significantly different from 1')
        
    PlotDist(f_t,params[3],params[4],df)
    
    
 
### start main
if __name__ == "__main__":
    main()