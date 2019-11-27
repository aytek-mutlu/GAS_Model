#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 23:00:14 2019

@author: aytekm - cindyz
"""


import pandas as pd
import numpy as np
import os
from scipy.stats import describe
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.optimize as opt
import scipy.special as scsp
import statsmodels.api as sm
import math
import copy
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


def Sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def ReverseSigmoid(x,k):
    return np.log(x/(k-x))

def BetaDist(mu_t,c,s,vY_t_i):
    return (c*scsp.gamma(s))/(scsp.gamma(s*mu_t)*scsp.gamma(s*(1-mu_t))) * np.power(vY_t_i,c*s*mu_t-1) * np.power((1-np.power(vY_t_i,c)),s*(1-mu_t)-1)

def ParamTransform(vP):
    return [vP[0],Sigmoid(vP[1]),Sigmoid(vP[2]),5*Sigmoid(vP[3]),5*Sigmoid(vP[4])]

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


def StdErrors(fun,params):
    
    #hessian
    hes = -hessian_2sided(fun,params)
    mHI = np.linalg.inv(hes)
    
    #std. errors
    std_errors = list(np.sqrt(np.diag(-mHI)))
    
    #sandwich form robust std. errors
    mHI = (mHI + mHI.T)/2
    mG = jacobian_2sided(fun,params)
    cov_matrix_sandwich = mHI @ (mG.T @ mG) @ mHI
    std_errors_sandwich = list(np.sqrt(np.diag(cov_matrix_sandwich))) 
    
    return std_errors_sandwich

def LL_PredictionErrorDecomposition(vP,vY,mean_overall,vN):
    
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

    return vLL

def EstimateParams(df):
        
    #initial params for w,beta,alpha,c,s
    vP0_t = [0.01, 0.99, 0.01, 3, 0.99]
    
    vP0 = [vP0_t[0],ReverseSigmoid(vP0_t[1],1),ReverseSigmoid(vP0_t[2],1),ReverseSigmoid(vP0_t[3],5),ReverseSigmoid(vP0_t[4],5)]
    
    vY = df.iloc[:,2:11]

    mean_overall = vY.sum().sum() / vY.count().sum()
    vN = df['N']
    
    sumLL= lambda vP: -np.sum(LL_PredictionErrorDecomposition(vP,vY,mean_overall,vN))  
    
    
    res= opt.minimize(sumLL, vP0, method='BFGS', options={'disp': True})
    params = ParamTransform(res.x)
    
    print('Optimized parameters:\nomega: ',params[0],'\nbeta: ',params[1],'\nalpha: ',params[2],'\nc: ',params[3],'\ns: ',params[4])
    
    
    return res

def main():
    
    filename = 'LGD-data'
    
    df = read_clean_data(filename)
    
    #print(plot_summarize_data(df))
    
    res = EstimateParams(df)
    

### start main
if __name__ == "__main__":
    main()