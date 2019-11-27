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

from lib.grad import *


def read_clean_data(filename):
    '''
    Purpose:
        Read and clean LGD dataset
    
    Inputs:
        filename:   string, filename for LGD
        
    Return value:
        df          dataframe, data
    '''
    ## read data
    df  = pd.read_excel('data/'+filename+'.xlsx')

    ## calculate no. of obs.
    df['N'] = df.iloc[:,2:11].notna().sum(axis=1)
    
    ## trim below 1% and above 99%
    df.iloc[:,2:11] = np.where(df.iloc[:,2:11]<0.01,0.01,np.where(df.iloc[:,2:11]>0.99,0.99,df.iloc[:,2:11]))

    return df


def plot_summarize_data(df):
    '''
    Purpose:
        Plot and summarize data
    
    Inputs:
        df:                 dataframe, data
        
    Return value:
        df_summary          dataframe, descriptives       
    '''
    
    pass


def PlotDist(f_t,c,s,df):
    
    #num. of observations
    iN = len(f_t)
    
    #mu_t
    mu_t = Sigmoid(f_t)
    
    mean_t = np.zeros((iN,))
    
    #calculate mean over time
    for i in range(iN):
        mean_t[i] = scsp.gamma(s)/scsp.gamma(s*mu_t[i]) * scsp.gamma(s*mu_t[i]+1/c)/scsp.gamma(s+1/c)
    
    #prepare data
    df['date'] = [pd.datetime(year=y,month=m,day=1) for y,m in zip(df.Year,df.Month)]
    df_final = pd.concat([df.iloc[:,2:11],df['date']],axis=1)
    df_final = pd.melt(df_final,id_vars=['date'])
    
    #plot
    fig,ax = plt.subplots()
    ax.plot(df.date,mean_t,'b')
    ax.plot(df_final['date'],df_final['value'],'r+')
    ax.legend().set_visible(False)

 
def Sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def ReverseSigmoid(x,k):
    return np.log(x/(k-x))

def BetaDist(mu_t,c,s,vY_t_i):
    return (c*scsp.gamma(s))/(scsp.gamma(s*mu_t)*scsp.gamma(s*(1-mu_t))) * np.power(vY_t_i,c*s*mu_t-1) * np.power((1-np.power(vY_t_i,c)),s*(1-mu_t)-1)

def ParamTransform(vP,bShapeAsVector=False):
    
    vP_transformed = [vP[0],Sigmoid(vP[1]),Sigmoid(vP[2]),5*Sigmoid(vP[3]),5*Sigmoid(vP[4])]
    if (bShapeAsVector == True):
        return np.array(vP_transformed)
    else:
        return vP_transformed

def LogLikelihood(mu_t,vP,N_t,vY_t):
    c = vP[3]
    s = vP[4]
    ll=1
    for i in range(N_t):
        ll = ll * BetaDist(mu_t,c,s,vY_t[i])
        
    return np.log(ll)


def Derivative(mu_t,vP,N_t,vY_t):
    c = vP[3]
    s = vP[4]
    deriv = 0
    
    for i in range(N_t):
        deriv += -s*scsp.digamma(s*mu_t) + s*scsp.digamma(s*(1-mu_t)) + c*s*np.log(vY_t[i]) -s*np.log(1-np.power(vY_t[i],c))

    return deriv



def LL_PredictionErrorDecomposition(vP,vY,mean_overall,vN,return_f_t = False):
    
    #transform vP
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
        
        if N_t>0:
            score =  (1/N_t) * Derivative(Sigmoid(f_t[i]),vP,N_t,vY_t) *  np.exp(f_t[i])/(np.power((1+np.exp(f_t[i])),2))
 
            
        f_t[i+1] = omega + beta*f_t[i] + alpha * score 
        vLL[i+1] = LogLikelihood(Sigmoid(f_t[i+1]),vP,vN[i+1],vY.loc[i+1])
    
    if return_f_t:
        return f_t
    else:
        return vLL


def StdErrors(fun,params,vY,mean_overall,vN,transform):
     
    
    #hessian and std. errors
    hes = -hessian_2sided(fun,params)
    inv_hes = np.linalg.inv(hes)
    std_errors = list(np.sqrt(np.diag(-inv_hes)))
    
    #sandwich form robust std. errors
    inv_hes_symetric = (inv_hes + inv_hes.T)/2
    mG = jacobian_2sided(LL_PredictionErrorDecomposition,params,vY,mean_overall,vN)
    
    cov_matrix_sandwich = inv_hes_symetric @ (mG.T @ mG) @ inv_hes_symetric
    
    if transform:
        mJ = jacobian_2sided(ParamTransform, params,True)
        cov_matrix_sandwich = mJ @ cov_matrix_sandwich @ mJ.T
     
    std_errors_sandwich = list(np.sqrt(np.diag(cov_matrix_sandwich)))
    
    return std_errors_sandwich


def EstimateParams(df):
        
    #initial params for w,beta,alpha,c,s
    vP0_t = [0.01, 0.99, 0.01, 3, 0.99]
    
    vP0 = [vP0_t[0],ReverseSigmoid(vP0_t[1],1),ReverseSigmoid(vP0_t[2],1),ReverseSigmoid(vP0_t[3],5),ReverseSigmoid(vP0_t[4],5)]
    
    vY = df.iloc[:,2:11]

    mean_overall = vY.sum().sum() / vY.count().sum()
    vN = df['N']
    
    sumLL= lambda vP: -np.sum(LL_PredictionErrorDecomposition(vP,vY,mean_overall,vN))  
    
    
    res= opt.minimize(sumLL, vP0, method='BFGS', options={'disp': True, 'maxiter':250})
    
    params = ParamTransform(res.x)
    params_untransformed = res.x    
    
    std_errors_untransformed = StdErrors(sumLL,params_untransformed,vY,mean_overall,vN,transform = False)
    std_errors_transformed = StdErrors(sumLL,params_untransformed,vY,mean_overall,vN,transform = True)
    
    f_t_optimized = LL_PredictionErrorDecomposition(params_untransformed,vY,mean_overall,vN,True)
    
    return [res.fun,params,list(params_untransformed),std_errors_transformed,std_errors_untransformed,f_t_optimized]


def main():
    
    filename = 'LGD-data'
    
    df = read_clean_data(filename)
    
    #print(plot_summarize_data(df))
    
    [likelihood,params,params_original,std_err,std_err_original,f_t] = EstimateParams(df)
    print('Log-likelihood: ',-likelihood)
    print('Optimized reparametrized parameters:\nomega: ',params[0],'\nbeta: ',params[1],'\nalpha: ',params[2],'\nc: ',params[3],'\ns: ',params[4])
    print('Optimized original parameters:\nomega: ',params_original[0],'\nbeta: ',params_original[1],'\nalpha: ',params_original[2],'\nc: ',params_original[3],'\ns: ',params_original[4])
    print('Optimized reparametrized std. errors:\nomega: ',std_err[0],'\nbeta: ',std_err[1],'\nalpha: ',std_err[2],'\nc: ',std_err[3],'\ns: ',std_err[4])
    print('Optimized original std. errors:\nomega: ',std_err_original[0],'\nbeta: ',std_err_original[1],'\nalpha: ',std_err_original[2],'\nc: ',std_err_original[3],'\ns: ',std_err_original[4])
    
    PlotDist(f_t,params[3],params[4],df)

    
 
### start main
if __name__ == "__main__":
    main()