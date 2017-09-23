# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:24:17 2017

@author: gao

得到语音和噪声的混合数据 以.pkl的形式存放在mix_data中
"""
import librosa as lr
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from ams_extract import ams_extractor

resample_rate=16000
nfft = 512
offset = int(nfft/2)


def genAndSave_trainData(speech_fileName):#给纯净语音打标签
    
    s, fs = lr.load(speech_fileName, sr=None)    
    s = lr.resample(s, fs, resample_rate)     # resample to 16k
    s = np.array(s)
    
    s_tf = lr.core.stft(s, n_fft=nfft, hop_length=offset)
    s_db = lr.core.amplitude_to_db(np.abs(s_tf))
    #s_angle = np.angle(s_tf)
    
    x_label = np.ones(s_db.shape[1]) # initialize x_label to all one
    xmean = s_db.mean(axis=0)
    for i in range(s_db.shape[1]):
        if xmean[i] < -40:
            x_label[i] = 0
    #xstd =  np.std(s_db, axis =0)       
    #x_data = (s_db-xmean)/(xstd+1e-10)      # normalize train data to zero mean and unit std
    return x_label#,x_data,s_angle,xmean,xstd

'''
    #label&data图形显示
    plt.subplot(211) 
    plt.plot(xmean)
    plt.subplot(212)
    plt.plot(x_label)
    plt.show()
'''

def gen_vadWav(x_data,x_label,s_angle,xmean,xstd):#标记处理后的波形显示函数   
    for i in range(x_data.shape[1]):
        x_data[:,i] = x_data[:,i] * xstd[i] + xmean[i]  # 逆归一化处理
    
    speech_amp = lr.core.db_to_amplitude(x_data)      
    for i in range(len(x_label)):
        if x_label[i]==0:
            speech_amp[:,i]=1#是data的地方就都保留 不是的地方全都置成0
    speech_tf = speech_amp * np.exp(s_angle*1j)
    speech = lr.core.istft(speech_tf, hop_length = offset)#加入角度信息傅里叶反变换得到图谱
    return speech  


def Read_wav(fileName):#定义的不好 未使用
    
    s, fs = lr.load(fileName, sr=None)    
    s = lr.resample(s, fs, resample_rate)     # resample to 16k
    s = np.array(s)
    
    s_tf = lr.core.stft(s, n_fft=nfft, hop_length=offset)#对数据进行快速傅里叶变化
    s_db = lr.core.amplitude_to_db(np.abs(s_tf))#幅值转化为dB
    s_angle = np.angle(s_tf)#记录傅里叶变换后的角度
    
    xmean = s_db.mean(axis=0)#求傅里叶变化后的均值
    xstd =  np.std(s_db, axis =0)  #标准差    
    x_data = (s_db-xmean)/(xstd+1e-10)   # normalize train data to zero mean and unit std
    return x_data,s_angle

def MFCC(y,sr):#提取20D的MFCC
    return lr.feature.mfcc(y=y,n_fft=nfft,hop_length=offset,n_mfcc=20)

def Pitch(y,sr):
    pitches, magnitudes = lr.core.pitch.piptrack(y=y,n_fft=nfft,hop_length=offset)
    # Select out pitches with high energy
    pitches = pitches[magnitudes > np.median(magnitudes)]
    return lr.core.pitch_tuning(pitches)


  
def Mix_wav(speechName,noiseName,mix_snr=5,train=True):#对数据加一个5db的噪声
    s,fs=lr.load(speechName,sr=None)
    n,fsn=lr.load(noiseName,sr=None)
    s=lr.resample(s,fs,resample_rate)
    n=lr.resample(n,fsn,resample_rate)
    s=np.array(s)
    n=np.array(n)
    len_s=len(s)
    len_n=len(n)
    if len_s<=len_n:
        n=n[0:len_s]
    else:
        n_extend_num=int(len_s/len_n)+1
        n=n.repeat(n_extend_num)
        n=n[0:len_s]
    alpha=np.sqrt((s**2).sum()/((n**2).sum()*10**(mix_snr/10)))  
    mix=s+alpha*n
    mix_tf=lr.core.stft(mix,n_fft=nfft,hop_length=offset) 
    mix_db=lr.core.amplitude_to_db(np.abs(mix_tf))
    
    mfcc=MFCC(mix,sr=fs)
    
    pitch=Pitch(mix,sr=fs)
    
    ams=ams_extractor(mix,sr=fs,win_len=nfft,shift_len=offset,order=1)
    
    if train==True:
        return mix_db, mfcc, pitch,ams#mix_db
    else:
        mix_angle=np.angle(mix_tf)
        return mix_db,mix_angle
   
  
    
def I_Mix_wav(mix_db,mix_angle):# 加性噪声语音波形逆变换
    
    mix_amp=lr.core.db_to_amplitude(mix_db)
    mix_tf=mix_amp*np.exp(mix_angle*1j)
    mix=lr.core.istft(mix_tf,hop_length=offset)
    return mix  

def label_save(mix_data,mix_label,fn):#fn为文件要存储的位置
    dic_data={'mix_data':mix_data,
              'mix_label':mix_label} #将数据存入字典中
    with open(fn,'wb') as f:
       pickle.dump(dic_data,f)
       
def _create_zero_indexes(speech_fileName):#过零率
    s, fs = lr.load(speech_fileName, sr=None)    
    s = lr.resample(s, fs, resample_rate)     # resample to 16k
    zero_indexes=[]
    zero_crossings=lr.zero_crossings(s)
    zero_index=np.nonzero(zero_crossings)[0]
    zero_indexes.append(zero_index)


  
#得到加噪的语音信号
filepath="C:/Users/gao/vad/863_IBM_test/"#添加路径1
#filepath="D:/863语音数据/863_data/863_IBM_train1/"#添加路径2
dirname=os.listdir(filepath)#获取全部文件
filepath1="C:/Users/gao/vad/noise/"
dirnoise=os.listdir(filepath1)
for i in range(0,len(dirname)):
        for j in range(0,len(dirnoise)):
            noise_fileName=filepath1+dirnoise[j]
            #print(noise_fileName)#读取噪声音频
            speech_fileName=filepath+dirname[i]
            #print(speech_fileName)#读取原始数据
            
         
            mix_label=genAndSave_trainData(speech_fileName)#用原始音频得到数据的标签
            #plt.plot(mix_label)
            #plt.show()
            
            #得到5dB下的混合音频
            #mix_db,mix_angle=Mix_wav(speech_fileName,noise_fileName,mix_snr=5,train=False)
            
            mix_db, mfcc, pitch,ams=Mix_wav(speech_fileName,noise_fileName,mix_snr=0,train=True)
            
            #print(mix_tf.shape,mfcc.shape,pitch.shape,ams.shape)
            
            mix=np.row_stack((mix_db,mfcc,ams))
            #print(mix.shape)
                                              
           
            #mixdata的归一化
            mixmean= mix.mean(axis=0)#求出均值
            mixstd = np.std(mix, axis =0) #求出标准差      
            mix_data= (mix-mixmean)/(mixstd+1e-10) #归一化后的数据
            print(mix_data.shape)
            print(mix_label)
            
            #a=label_save(mix_data,mix_label,'C:/Users/gao/vad/test_data/0dB/factory_test/'+dirname[i].strip('.WAV')+dirnoise[j].strip('.wav')+'0'+'.pkl')
            
            """
            x=I_Mix_wav(mix_db,mix_angle)
            lr.output.write_wav('C:/Users/gao/vad/test_data/'+dirname[i].strip('.WAV')+'__5'+dirnoise[j], x, resample_rate)
            #显示加性噪声语音
            plt.plot(x)
            plt.show()
            """






     
















     