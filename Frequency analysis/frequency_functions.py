import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from itertools import combinations
import scipy
from scipy import signal
from scipy import fft

#split speed and difference dfs so that you can find averages on different time chunks
def split_df(df, chunk_size = 6000): #default chunks are 10 min
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks 

def get_fft(data,time):
    # Number of sample points
    N = len(time)
    # sample spacing
    time=np.array(time)
    T = 1.0 /(len(time)/(time[-1]-time[0]))
    # print(T)
    x = np.linspace(0.0, N*T, N)
    y = np.array(data)
    yf = scipy.fft.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#     p=np.multiply(yf,time[-1])
    X=np.log10(xf[1:])
    Y=np.log10(2.0/N * np.abs(yf[1:(N//2)]))
#     P=np.log10(2.0/N * np.abs(p[1:(N//2)]))
    return(X,Y)

def spectrum1(h, dt=.1):
    """
    basic spectral estimation
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = len(h)
    npositive = nt//2
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice] 
    ft = np.fft.fft(h)[pslice]
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    # Normalization for Power Spectrum
    psraw /= nt**2
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    return freqs, psraw, psdraw


def avg_psd(array):
    chunks=split_df(array)
    N=len(chunks)-1
    freqs=np.empty((N, 0)).tolist()
    ps=np.empty((N, 0)).tolist()
    psd=np.empty((N, 0)).tolist()
    
    for i in range (0, N):
        freqs[i], ps[i], psd[i] =spectrum1(chunks[i], dt=.1)
    
    
    arrays = [np.array(x) for x in freqs]
    freqs_avg= [np.mean(k,dtype=np.float64) for k in zip(*arrays)] 
    arrays2 = [np.array(x) for x in psd]
    psd_avg= [np.mean(k,dtype=np.float64) for k in zip(*arrays2)] 
    return freqs_avg,psd_avg
    #return freqs, psd

def avgdf_psd (df):
    N=len(df.columns)
    all_df_freq=np.empty((N, 0)).tolist()
    all_df_psd=np.empty((N, 0)).tolist()
    
    for i in range (0,N):
        all_df_freq[i], all_df_psd[i] =avg_psd(df.iloc[:,i])

    arrays = [np.array(x) for x in  all_df_freq]
    
    
    freqs_avg= [np.mean(k,dtype=np.float64) for k in zip(*arrays)] 
    #freqs_median= [np.median(k,dtype=np.float64) for k in zip(*arrays)] 
    
    #freqs_avgdf=pd.DataFrame(arrays)
    #freqs_avg=freqs_avgdf.mean()
    arrays2 = [np.array(x) for x in all_df_psd]
    #psd_avgdf=pd.DataFrame(arrays2)
    #psd_avg=psd_avgdf.mean()
    psd_avg= [np.mean(k,dtype=np.float64) for k in zip(*arrays2)] 


    #return freqs_avg,psd_avg
    return freqs_avg, psd_avg


#functions to deal with angle wrap around effect
def unwrap_angle (directionarray, degrees=True):
    if (degrees==False):
        directionarray=+math.pi
    else:
        newdirection=directionarray*math.pi/180-math.pi
    return np.unwrap(newdirection)

def rewrap_angle(unwrappedarray, degrees=True):
    rewrapped = (unwrappedarray + np.pi) % (2 * np.pi)
    #same as np.arctan(np.cos(unwrappedarray), np.sin(unwrappedarray))
    if degrees:
        return rewrapped #this is in radians not degrees
    else:
        return rewrapped*180/math.pi
    
def unwrap_angulardf (directiondf, degrees=True):
    M=len(directiondf.columns)
    unwrappeddf=np.empty((M, 0)).tolist()
    for i in range (0,M):
        unwrappeddf[i]=unwrap_angle(directiondf.iloc[:,i])
    return pd.DataFrame(unwrappeddf, index=directiondf.columns).T   
