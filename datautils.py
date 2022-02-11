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


def butter_bandpass (lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter (signal, lowcut, highcut, fs, order=6):
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
            covNY[i,j] = cov[0, 1]
    
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
    data_path.mkdir(parents=True, exist_ok=True)
    for sj in subjects:
          for L in ['E','T']:
                filename = "A0" + str(sj) + L + '.mat'
                if not (data_path / filename).exists():
                    content = requests.get(url + filename).content
                    (data_path / filename).open("wb").write(content)
    print (f'Subjects {np.array(subjects)} are now in {data_path}') 
    return data_path


def mat_extractor (
    path, beg = 500, end = 1500,
    remove_eog = True, bpf_dict={'apply':True, 'fs': 250, 'lc':4, 'hc':38, 'order':6},
    channel_norm = True
    ):
    '''
    function for extracting data from a single mat file and doing preprocessing on it
    
    data_path    : the directroy of the desired .mat file
    beg, end     : refer to the begining and end og the part of the 8-second 
                   signal that the motor imagery task is done
    remove_eog   : decides whether we want to remove EOG or use raw EEG
    bpf_dict     : a dictionary for specifications of bandpass filter that we want to apply
                   it should be defined with the following keys and values
                   'apply'   : deciding whether we want to apply bpf or not (boolean)
                   'lc','hc' : lowcut and highcut of butterworth bpf (float)
                   'fs'      : sampling rate (250 for bcic-IV-2a)
                   'order'   : the order of butterworth bpf
    channel_norm : z-score normalization for each channel (boolean)
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
        allsig = np.concatenate((opened, closed, motion),axis=0)
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
            
            if bpf_dict['apply']:
                x = butter_bandpass_filter(
                    signal  = x,
                    lowcut  = bpf_dict['lc'],
                    highcut = bpf_dict['hc'],
                    fs      = bpf_dict['fs'],
                    order   = bpf_dict['order']
                )
                
            if channel_norm:
                x = (x - np.mean(x))/np.std(x)
            
            x = np.expand_dims(x, axis=0)
            y = np.array([labels[i,0]])
            
            xx = np.concatenate((xx, x),axis=0)
            yy = np.concatenate((yy, y),axis=0)
    
    return xx[1:], yy[1:] # first sample is empty


def cropper (signals, labels, window, step):
    '''
    function to crop signals with a window and step for data augmentation
    
    signals : in format (N, ch, T) where N is the number of samples,
              ch is the number of channels, and T is the number of time points
    labels  : labels of samples
    window  : the window for cropping signals in terms of time points (int)
    step    : the step for moving the window in term of time points (int)
    '''
    
    time   = signals.shape[2]             # number of time points
    begs   = list(range(0, time, step))   # begining indices of sliding windows
    crops  = list()                       # list contaiing croppend signals
    annots = list()                       # labels of cropped signals
    
    for i in range(signals.shape[0]):
        for j in begs:
            if j + window <= time:
                crops .append(signals[i:i+1, :, j:j+window])
                annots.append(labels[i])
    
    crops  = np.concatenate(crops, axis=0)
    annots = np.array(annots)
    
    return crops, annots


def dataset_extractor (train_subs, test_subs, data_path):
    '''
    a function to extract some subjects as train, and some as test for inter-subject analysis
    
    '''
    pass