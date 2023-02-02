import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

import astropy
from astropy import stats
import math

import scipy
from scipy import stats

from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, atleast_1d, sqrt, array,
                   compress, pi, exp, ravel, count_nonzero, sin, cos,
                   arctan2, hypot)


from scipy import optimize
from scipy import special

import figurefirst as fifi
import time

import metpy
from metpy import calc
from matplotlib import gridspec

def split_df(df, chunk_size = 6000): #default chunks are 10 min
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks 

def avg_df(chunks):
    avg_df=pd.DataFrame(index=chunks[0].columns)
    N=len(chunks)
    if len(chunks[-1])< len(chunks[0]):
        N=len(chunks)-1
    testdf=np.empty((N, 0)).tolist()
    for i in range(0,N):
        testdf[i]=pd.DataFrame(chunks[i].mean(), columns=[i])
        avg_df=pd.concat([avg_df, testdf[i]], axis=1)
    return avg_df

def std_df(chunks):
    std_df=pd.DataFrame(index=chunks[0].columns)
    N=len(chunks)
    if len(chunks[-1])< len(chunks[0]):
        N=len(chunks)-1
    testdf=np.empty((N, 0)).tolist()
    for i in range(0,N):
        testdf[i]=pd.DataFrame(chunks[i].std(), columns=[i])
        std_df=pd.concat([std_df, testdf[i]], axis=1)
    return std_df

def maxchange_std (df, startval, lagarray, method='forward'):
    
    d_list=np.zeros((len(lagarray)))
    j=0
    
    def checkforwardbounds(df,startval,lag):
        if (startval+i)>=len(df):
            raise ValueError('Lag value extends beyond length of timeseries')            
    
    def checkbackwardbounds(df,startval,lag):
        if (startval-i)<0:
            raise ValueError('Lag value extends beyond length of timeseries')  

    
    for i in lagarray:
        
        if method=='forward':
            checkforwardbounds(df,startval,i)
            b=df.iloc[startval:startval+i]
            std= mycircstd(b, high=360, low=0) 
            d_list[j]=std
                
        j+=1
    return d_list

def _circfuncs_common(samples, high, low, nan_policy='propagate'):
    # Ensure samples are array-like and size is not zero
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = sin((samples - low)*2.*pi / (high - low))
    cos_samp = cos((samples - low)*2.*pi / (high - low))
    #sin_samp=sin(samples)
    #cos_samp=cos(samples)
    mask = None
    return samples, sin_samp, cos_samp, mask


def mycircstd(samples, high=360, low=0, axis=None, nan_policy='propagate'):

    samples, sin_samp, cos_samp, mask = _circfuncs_common(samples, high, low,
                                                          nan_policy=nan_policy)
    if mask is None:
        sin_mean = sin_samp.mean(axis=axis)  # [1] (2.2.3)
        cos_mean = cos_samp.mean(axis=axis)  # [1] (2.2.3)
    else:
        nsum = np.asarray(np.sum(~mask, axis=axis).astype(float))
        nsum[nsum == 0] = np.nan
        sin_mean = sin_samp.sum(axis=axis) / nsum
        cos_mean = cos_samp.sum(axis=axis) / nsum
    # hypot can go slightly above 1 due to rounding errors
    with np.errstate(invalid='ignore'):
        R = np.minimum(1, hypot(sin_mean, cos_mean))  # [1] (2.2.4)

    #res = sqrt(-2*log(R))
    #if not normalize:
    #    res *= (high-low)/(2.*pi)  # [1] (2.3.14) w/ (2.3.7)
    res = np.sqrt(2 * (1 -R))
    
    return res

def maxchange_std (df, startval, lagarray, method='forward'):
    
    d_list=np.zeros((len(lagarray)))
    j=0
    
    def checkforwardbounds(df,startval,lag):
        if (startval+i)>=len(df):
            raise ValueError('Lag value extends beyond length of timeseries')            
    
    def checkbackwardbounds(df,startval,lag):
        if (startval-i)<0:
            raise ValueError('Lag value extends beyond length of timeseries')  

    
    for i in lagarray:
        
        if method=='forward':
            checkforwardbounds(df,startval,i)
            b=df.iloc[startval:startval+i]
            std= mycircstd(b, high=360, low=0) 
            d_list[j]=std
                
        j+=1
    return d_list


def vert_temporal_analysis (vertical_vel, ucomp, vcomp, horizontal_speed, horizontal_dir, lagarray, method='forward'):
    '''
# This function takes a single starting value from a column of directional data in a pandas df or np array
# and returns a list of values at each desired lag.

# Note, directional data must be between 0-360 degrees.
# Need to convert data if in radians and/or if it goes from -180 to 180.


#inputs: df - one column or array of directional data bounded between 0-360
#        startval - first point of wind data that you want to analyze
#        lagarray - array of time lags that you wish to calculate. a lag of 1=.1 sec
#        method - time direction in which lags are calculated 
#        options are forward, backward, average, max, or min. 
#             if no method is selected, forward is the default
    '''
    
    avg_vertical_vel_list=np.zeros((len(lagarray)))
    avg_horizontal_speed_list=np.zeros((len(lagarray)))
    avg_horizontal_dir_list=np.zeros((len(lagarray)))
    
    std_vertical_vel_list=np.zeros((len(lagarray)))
    std_horizontal_speed_list=np.zeros((len(lagarray)))
    std_horizontal_dir_list=np.zeros((len(lagarray)))
    
    frictional_vel_list=np.zeros((len(lagarray)))
    j=0

    def fixangle(angle):
        if angle>=180:
            angle=360-angle
        return angle
    
    
    def checkforwardbounds(dir_df,startval,lag): 
    #compute the standard deviation backwards if the start value+lag will be greater than the length of the df
        if (startval+i)>=len(dir_df): 
            
            vertchunk=vertical_vel.iloc[(startval-i):startval]
            horizontalchunk=horizontal_speed.iloc[(startval-i):startval]
            dirchunk=horizontal_dir.iloc[(startval-i):startval]
            uchunk=ucomp.iloc[(startval-i):startval]
            vchunk=vcomp.iloc[(startval-i):startval]
            
            avg_vertical_vel_list[j]= np.mean(vertchunk)
            avg_horizontal_speed_list[j]= np.mean(horizontalchunk)
            avg_horizontal_dir_list[j]=scipy.stats.circmean(dirchunk,high=360, low=0)
            
            std_vertical_vel_list[j]= np.std(vertchunk)
            std_horizontal_speed_list[j]=np.std(horizontalchunk)
            std_horizontal_dir_list[j]=mycircstd(dirchunk, high=360, low=0)
            frictional_vel_list[j]= metpy.calc.friction_velocity(u=uchunk,v=vchunk,w=vertchunk, axis=0)   
    
    for i in lagarray:
        
        if method=='forward': #currently only forward - will eventually update?
            startval=int(vertical_vel.sample(1).index.to_numpy()) #get a random start value for each lag
            checkforwardbounds(vertical_vel,startval,i)
            vertchunk=vertical_vel.iloc[startval:(startval+i)]
            horizontalchunk=horizontal_speed.iloc[startval:(startval+i)]
            dirchunk=horizontal_dir.iloc[startval:(startval+i)]
            uchunk=ucomp.iloc[startval:(startval+i)]
            vchunk=vcomp.iloc[startval:(startval+i)]
            
            avg_vertical_vel_list[j]= np.mean(vertchunk)
            avg_horizontal_speed_list[j]= np.mean(horizontalchunk)
            avg_horizontal_dir_list[j]=scipy.stats.circmean(dirchunk,high=360, low=0)
            
            std_vertical_vel_list[j]= np.std(vertchunk)
            std_horizontal_speed_list[j]=np.std(horizontalchunk)
            std_horizontal_dir_list[j]=mycircstd(dirchunk, high=360, low=0) 
            frictional_vel_list[j]= metpy.calc.friction_velocity(u=uchunk,v=vchunk,w=vertchunk, axis=0)  
            
        j+=1
    #return uchunk, vchunk
    return avg_vertical_vel_list, avg_horizontal_speed_list, avg_horizontal_dir_list, std_vertical_vel_list,std_horizontal_speed_list,std_horizontal_dir_list, frictional_vel_list
