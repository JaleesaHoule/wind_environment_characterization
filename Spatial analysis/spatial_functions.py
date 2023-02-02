import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
import geopy
from geopy import distance
import math

import scipy
from scipy import stats
from itertools import combinations

from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, atleast_1d, sqrt, array,
                   compress, pi, exp, ravel, count_nonzero, sin, cos,
                   arctan2, hypot)


from scipy import optimize
from scipy import special
from scipy import spatial

import figurefirst as fifi
import time

from scipy import signal
import math

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import pynumdiff



def spatialchanges (dir_df, speed_df, x_position,y_position,z_position, radians=False):
    '''
this takes the difference between 2 sensors for all possible combinations. Returns a df sorted by distance
    
    dir_df - all columns of directional data from one data collection, assumed to be in range 0-360
    
    speed_df - all columns of horizontal speed data from one data collection
    
    latlons - list of tuples in form (lat, lon) which correspond to dir_df and speed_df

all inputs are expected to be organized alphabetically/numerically from sensor A (1) to I (9) 
    
    '''    
    def combine(arr, s):  #to determine total permutation of sensor pairs
        return list(combinations(arr, s)) 
    
    ########this is for direction 
    if (radians==True): #to convert into degrees if needed
        dir_df=dir_df*180/np.pi
    if (any(np.max(dir_df, axis=0)) > 360 or any(np.max(dir_df, axis=0)) < 0):
        raise ValueError ("Directional data not bounded correctly. Needs to be in range [0,360] or [0,2pi]")
    
    
    columns=np.arange(0,len(dir_df.T)) #count total number of sensor recordings in df - should be same for S2, direction and position dfs
    N=len(combine(columns, 2)) #compute total number of permutations for all 2 sensor pairs
    direction_diff_list=np.empty((N, 0)).tolist()
    speed_diff_list=np.empty((N, 0)).tolist()
    direction_avg_list=np.empty((N, 0)).tolist()
    speed_avg_list=np.empty((N, 0)).tolist()
    distances=np.empty((N, 0)).tolist()
    k=0
    
    for i in columns:
        totalcombinations=len(dir_df.T)-i
        for j in range (1, totalcombinations):
            
            direction_diff_list[k]=np.abs(dir_df.iloc[:,i]-dir_df.iloc[:,i+j]) #direction_diff
            speed_diff_list[k]=speed_df.iloc[:,i]-speed_df.iloc[:,i+j] #speed_diff
            speed_avg_list[k]=(np.abs(speed_df.iloc[:,i]+speed_df.iloc[:,i+j])/2) #gives average speed between two sensors
            
            angulardata=np.array([dir_df.iloc[:,i],dir_df.iloc[:,i+j]])
            direction_avg_list[k]=scipy.stats.circmean(angulardata*math.pi/180, axis=0)*180/math.pi #computes mean between 2 sensors at each time recording
            
            u=np.array([x_position.iloc[0,i],y_position.iloc[0,i],z_position.iloc[0,i]]) #sensor1
            v=np.array([x_position.iloc[0,i+j],y_position.iloc[0,i+j],z_position.iloc[0,i+j]]) #sensor2
            distances[k]= scipy.spatial.distance.euclidean(u,v)*1000 #distance between sensor1 and sensor2 in meters

            #distances[k]=distance.distance(latlons[i],latlons[i+j]).m #compute distance in meters from gps coords
            k=k+1

    direction_diff=pd.DataFrame(direction_diff_list, index=np.round(distances,decimals=2))
    
    #fixes the angles
    M=len(direction_diff)
    for x in range (0,M):
        for y in (np.where(direction_diff_list[x]>180)):
            direction_diff.iloc[x,y]=360-direction_diff.iloc[x,y] #maximum difference in angle can be 180 - if more than 180, will subtract to get the smaller angle
    
    direction_diff=direction_diff.sort_index(ascending=True).T
    
    direction_avg=pd.DataFrame(direction_avg_list, index=np.round(distances,decimals=2)).sort_index(ascending=True).T
    speed_diff=pd.DataFrame(speed_diff_list, index=np.round(distances,decimals=2)).sort_index(ascending=True).T  
    speed_avg=pd.DataFrame(speed_avg_list, index=np.round(distances,decimals=2)).sort_index(ascending=True).T       
    
    return (direction_diff, direction_avg, speed_diff, speed_avg)

def smoothdf (dir_df, speed_df, filt):
    #functions to deal with angle wrap around effect
    def unwrap_angle (directionarray, degrees=True):
        if (degrees==False):
            directionarray=+math.pi
        else:
            newdirection=directionarray*math.pi/180-math.pi
        return np.unwrap(newdirection)

    def rewrap_angle(unwrappedarray, degrees=False):
        rewrapped = (unwrappedarray + np.pi) % (2 * np.pi)
    #same as np.arctan(np.cos(unwrappedarray), np.sin(unwrappedarray))
        if degrees:
            return rewrapped*180/math.pi 
        else:
            return rewrapped #this is in radians not degrees
   
    N=len(dir_df.columns)
    filtered_dir=np.empty((N, 0)).tolist()
    filtered_speed=np.empty((N, 0)).tolist()
    for i in range(0,N):
        filtered_speed[i] = signal.sosfilt(filt, speed_df.iloc[:,i])
        filtered_dir[i] = rewrap_angle(signal.sosfilt(filt, unwrap_angle(dir_df.iloc[:,i].dropna())), degrees=True)
    filtered_dir_df=pd.DataFrame(filtered_dir, index=dir_df.columns).T
    filtered_speed_df=pd.DataFrame(filtered_speed, index=speed_df.columns).T
    
    return filtered_dir_df, filtered_speed_df

def smoothdf_pynum (dir_df, speed_df, dt, params, unwrap=False):
    #functions to deal with angle wrap around effect
    def unwrap_angle (directionarray, degrees=True):
        if (degrees==False):
            directionarray=+math.pi
        else:
            newdirection=directionarray*math.pi/180-math.pi
        return np.unwrap(newdirection)

    def rewrap_angle(unwrappedarray, degrees=False):
        rewrapped = (unwrappedarray + np.pi) % (2 * np.pi)
    #same as np.arctan(np.cos(unwrappedarray), np.sin(unwrappedarray))
        if degrees:
            return rewrapped*180/math.pi 
        else:
            return rewrapped #this is in radians not degrees

    N=len(dir_df.columns)
    filtered_dir=np.empty((N, 0)).tolist()
    filtered_speed=np.empty((N, 0)).tolist()
    for i in range(0,N):
        filtered_speed[i], throwaway =  pynumdiff.smooth_finite_difference._smooth_finite_difference.butterdiff(x=speed_df.iloc[:,i], dt=dt, params=params, options={'iterate': True})
        if unwrap:
            filtered_dir[i], throwaway = pynumdiff.smooth_finite_difference._smooth_finite_difference.butterdiff(x=unwrap_angle(dir_df.iloc[:,i]), dt=dt, params=params, options={'iterate': True})
            filtered_dir[i]= rewrap_angle(filtered_dir[i], degrees=True)
        else:
            filtered_dir[i], throwaway = pynumdiff.smooth_finite_difference._smooth_finite_difference.butterdiff(x=dir_df.iloc[:,i], dt=dt, params=params, options={'iterate': True})

    filtered_dir_df=pd.DataFrame(filtered_dir, index=dir_df.columns).T
    filtered_speed_df=pd.DataFrame(filtered_speed, index=speed_df.columns).T
    
    return filtered_dir_df, filtered_speed_df

#split speed and difference dfs so that you can find averages on different time chunks
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

def avg_df_direction(chunks):
    avg_df=pd.DataFrame(index=chunks[0].columns)
    N=len(chunks)
    if len(chunks[-1])< len(chunks[0]):
        N=len(chunks)-1
    testdf=np.empty((N, 0)).tolist()    
    for i in range(0,N):
        testdf[i]=pd.DataFrame((scipy.stats.circmean(chunks[i]*np.pi/180, axis=0)*180/np.pi), columns=[i], index=chunks[0].columns)
        avg_df=pd.concat([avg_df, testdf[i]], axis=1)
    return avg_df

def find_avg_values(direction_diff, direction_avg, speed_diff, speed_avg):
    new_direction_diff=avg_df(split_df(direction_diff))
    new_direction_avg=avg_df_direction(split_df(direction_avg))
    new_speed_diff=avg_df(split_df(speed_diff))
    new_speed_avg=avg_df(split_df(speed_avg))
    
    return new_direction_diff, new_direction_avg, new_speed_diff,new_speed_avg

