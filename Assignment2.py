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

def LogLikelihood(mu_t,vP,N_t,vY_t):
    c = vP[3]
    s = vP[4]
    
    if N_t==0: 
        return np.log(1)
    
    else: 
        ll=0
        for i in range(N_t):
            ll+= np.log(c) + scsp.gammaln(s) - scsp.gammaln(s*mu_t) - scsp.gammaln(s*(1-mu_t)) 
            + (c*s*mu_t - 1)*np.log(vY_t[i]) + (s*(1-mu_t)-1)*np.log(1-np.power(vY_t[i],c))
        
        return ll
    
def Derivative(mu_t,vP,N_t,vY_t):
    c = vP[3]
    s = vP[4]
    deriv = 0
    
    for i in range(N_t):
        deriv += -s*scsp.digamma(s*mu_t) + s*scsp.digamma(s*(1-mu_t)) + c*s*np.log(vY_t[i]) -s*np.log(1-np.power(vY_t[i],c))

    return deriv


def LL_PredictionErrorDecomposition(vP,vY,mean_overall,vN):
    
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
            score =  1/N_t * Derivative(Sigmoid(f_t[i]),vP,N_t,vY_t) *  np.exp(f_t[i])/(np.power(1+np.exp(f_t[i]),2))
                        
        f_t[i+1] = omega + beta*f_t[i] + alpha * score 
        vLL[i+1] = LogLikelihood(Sigmoid(f_t[i+1]),vP,vN[i+1],vY.loc[i+1])
        
    return vLL

def EstimateParams(df):
        
    #initial params for w,beta,alpha,c,s
    vP0 = [0.001,0.86,0.26,3,0.84]
    
    vY = df.iloc[:,2:11]

    mean_overall = vY.sum().sum() / vY.count().sum()
    vN = df['N']
    
    sumLL= lambda vP: -np.sum(LL_PredictionErrorDecomposition(vP,vY,mean_overall,vN))  
    res= opt.minimize(sumLL, vP0, method="BFGS",options={'disp': True, 'maxiter':50})
    return res

def main():
    
    filename = 'LGD-data'
    
    df = read_clean_data(filename)
    
    #print(plot_summarize_data(df))
    
    res = EstimateParams(df)
    

    


### start main
if __name__ == "__main__":
    main()