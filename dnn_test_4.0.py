# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:30:57 2017

@author: gao
"""
import tensorflow as tf
import os
import numpy as np
import pickle

sess=tf.Session()
x=tf.placeholder(tf.float32,[None,412])

with open('C:/Users/gao/vad/wb.pkl','rb') as wb_data:
        data_dict=pickle.load(wb_data)
        W1=data_dict['W1']
        W2=data_dict['W2']
        b1=data_dict['b1']
        b2=data_dict['b2']
   

h1=tf.nn.relu(tf.matmul(x,W1)+b1)
y=tf.nn.sigmoid(tf.matmul(h1,W2)+b2)
y_=tf.placeholder(tf.float32,[None,1])

def test(file,dirname):
    result=0
    for i in range(0,len(dirname)):
        data1=open(file+dirname[i],"rb")
        data_dict1=pickle.load(data1)
        test_data=data_dict1["mix_data"]
        test_label=data_dict1["mix_label"]
        a=len(test_label)
        test_label=np.array(test_label).reshape((a,1))
        data1.close()
        
        correct_prediction=tf.equal(tf.nn.relu(tf.sign(y-0.5)),test_label)
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        total=sess.run(accuracy,{x:test_data.T,y_:test_label})
        
        '''
        Y=y.eval({x:test_data.T})
        Y1=tf.nn.elu(tf.sign(Y-0.5))
        y2=sess.run(Y1)
        print(y2)
        ''' 
        
        #print(total)
        
        result+=total
        
    print(file,result/len(dirname)) 
    
if __name__ == '__main__':
    #factory 
    file1='test_data/5dB/factory_test/'
    dirname1=os.listdir(file1)
    factory5=test(file1,dirname1)
    file2='test_data/_5dB/factory_test/'
    dirname2=os.listdir(file2)
    factory_5=test(file2,dirname2)
    file3='test_data/0dB/factory_test/'
    dirname3=os.listdir(file3)
    factory0=test(file3,dirname3)
    
    #car
    file1='test_data/5dB/car_test/'
    dirname1=os.listdir(file1)
    car5=test(file1,dirname1)
    file2='test_data/_5dB/car_test/'
    dirname2=os.listdir(file2)
    car_5=test(file2,dirname2)
    file3='test_data/0dB/car_test/'
    dirname3=os.listdir(file3)
    car0=test(file3,dirname3)
    
    #babble
    file1='test_data/5dB/babble_test/'
    dirname1=os.listdir(file1)
    babble5=test(file1,dirname1)
    file2='test_data/_5dB/babble_test/'
    dirname2=os.listdir(file2)
    babble_5=test(file2,dirname2)
    file3='test_data/0dB/babble_test/'
    dirname3=os.listdir(file3)
    babble0=test(file3,dirname3)
    
sess.close()
