# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:04:18 2017

@author: gao
"""
import librosa as lr
import numpy as np

resample_rate=16000
def read_sphere_wav(filename):#read .wav into array
    wav_data, fs =lr.load(filename,sr=None)
    wav_data= lr.resample(wav_data, fs,resample_rate )     # resample to 16k
    wav_data = np.array(wav_data)
    
    return wav_data