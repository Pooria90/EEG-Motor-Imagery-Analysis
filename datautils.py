'''
This module contains functions to download BCIC-IV-2a,
extract data from it, and do preprocessing
'''

import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft
from scipy.signal import butter, lfilter
from scipy.io import loadmat
from pathlib import Path


def butter_bandpass (lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter (signal, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, signal)
    return y


def eog_remove (signal, eeg_ch=22, eog_ch=3):
    '''
    function for removing EOG artifact
    based on *** ref ****
    
    signal: input in the form (eeg_ch + eog_ch)*T
            in which eeg_ch is the number of EEG channels,
            eog_ch is the number of EOG channels,
            and T is the number of time samples
    '''
    
    Y = signal[:-eog_ch, :] # received by EEG sensors
    N = signal[-eog_ch:, :] # EOG singal (noise)
    
    autoN = np.cov(N)
    covNY = np.zeros((eog_ch, eeg_ch))
    
    for i in range(eog_ch):
        for j in range(eeg_ch):
            cov   = np.cov(N[i:i+1, :], Y[j:j+1, :])
            covNY = cov[0, 1]
    
    b = np.linalg.inv(autoN).dot(covNY)
    return b


def downloader (path, subjects, url="http://bnci-horizon-2020.eu/database/data-sets/001-2014/"):
    '''
    function for downloading BCIC-IV-2a dataset
    
    path     : the directory in which you want to download the data into that (as str)
    subjects : a list containing the desired subjects to download (from 1 to 9)
    url      : the link to .mat file; if my default url has expired you can simply find the new download link on
               http://bnci-horizon-2020.eu/database/data-sets
    '''
    
    data_path = Path(path)
    for sj in subjects:
          for L in ['E','T']:
                filename = "A0" + str(sj) + L + '.mat'
                if not (data_path / filename).exists():
                    content = requests.get(url + filename).content
                    (data_path / filename).open("wb").write(content)
    return data_path


def mat_extractor (path, beg=500, end=1500, remove_eog = True):
    '''
    function for extracting data from a single mat file
    
    data_path  : the directroy of the desired .mat file
    beg, end   : refer to the begining and end og the part of the 8-second 
                 signal that the motor imagery task is done
    remove_eog : decides whether we want to remove EOG or use raw EEG
    '''
    
    mat  = loadmat(path)
    data = mat['data']
    
    xx = np.empty([1, 22, end-beg]) # signals
    yy = np.empty([1])              # labels
    N  = data.shape[1]              # number of samples
    
    # EOG recordings
    if remove_eog:
        opened = data[0][0][0][0][0]
        closed = data[0][1][0][0][0]
        motion = data[0][2][0][0][0]
        allsig = np.concatenate((opened, closed, motion),axis=0).T
        b = eog_remove(allsig)
    
    # iteration on motor imagery task sessions
    for j in range(3,N):
        samples = data[0][j][0][0][0] # whole signal of the session
        trials  = data[0][j][0][0][1] # indices of successive trials
        labels  = data[0][j][0][0][2] # labels of corresponding task
        
        # iteration on tasks in each session
        for i in range(48):
            if i < 47:
                x = samples[trials[i,0]:trials[i+1,0]]
            else:
                x = samples[trials[i,0]:]
            
            if remove_eog:
                x = (x[beg: end, 0: -3] - x[beg: end, -3: x.shape[1]+1].dot(b)).T
            else:
                x = x[beg: end, 0: -3]
            
            x = x.reshape(1, x.shape[0], x.shape[1])
            y = np.array([labels[i,0]])
            
            xx = np.concatenate((xx, x),axis=0)
            yy = np.concatenate((yy, y),axis=0)
    
    return xx, yy


def dataset_extractor (train_subs, test_subs, data_path):
    '''
    a function to extract some subjects as train, and some as test for inter-subject analysis
    
    '''
    pass