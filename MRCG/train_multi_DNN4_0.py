# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:41:23 2017

@author: gao
multi_perceptron-based VAD train_net 
"""
import tensorflow as tf
import pickle
import numpy as np
import os

input_units=786
h1=800
h2=200
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,input_units])
W1=tf.Variable(tf.truncated_normal([input_units,h1],stddev=0.1))
W2=tf.Variable(tf.zeros([h1,h2]))
W3=tf.Variable(tf.zeros([h2,1]))
b1=tf.Variable(tf.zeros([h1]))
b2=tf.Variable(tf.zeros([h2]))
b3=tf.Variable(tf.zeros([1]))


h1=tf.nn.relu(tf.matmul(x,W1)+b1)#N*200
h2=tf.nn.relu(tf.matmul(h1,W2)+b2)
y=tf.nn.sigmoid(tf.matmul(h2,W3)+b3)#N*1

y_=tf.placeholder(tf.float32,[None,1])#N*1
#定义反向传播函数 损失函数用来刻画预测值与真实值的差距
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
mse=tf.reduce_mean(tf.square(y_-y))
#定义反向传播算法来优化神经网络中的参数
#train_step=tf.train.GradientDescentOptimizer(0.08).minimize(cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(mse)


init=tf.global_variables_initializer()
sess.run(init)
filepath='MRCG_mixdata/'
dirname=os.listdir(filepath)#获取全部文件共300*3*3 约一小时
for j in range(10):
    loss=0
    for i in range(0,len(dirname)):
        data=open(filepath+dirname[i],"rb")
        data_dict=pickle.load(data)
        train_data=data_dict['mix_data']
        train_label=data_dict['mix_label']
        
        data.close()
        a=len(train_label)
        train_label=np.array(train_label).reshape((a,1))
        #print(train_label.shape)
        #train_data,label_data=(data_dict['mix_data'].T ,data_dict['mix_label'])
        loss1=sess.run(mse,{x:train_data.T,y_:train_label})
        sess.run(train_step,{x:train_data.T,y_:train_label})
        
        #print(sess.run(cross_entropy,{x:train_data.T,y_:train_label}))
        loss+=loss1
    print(loss/len(dirname))

dic_wb={'W1':W1.eval(),
        'W2':W2.eval(),
        'W3':W3.eval(),
        'b1':b1.eval(),
        'b2':b2.eval(),
        'b3':b3.eval(),}  
with open ('C:/Users/gao/vad/wb.pkl','wb') as f1:
    pickle.dump(dic_wb,f1) 
  

 

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
   

#factory 
file1='MRCG_testdata/5dB/factory_test/'
dirname1=os.listdir(file1)
factory5=test(file1,dirname1)
file2='MRCG_testdata/_5dB/factory_test/'
dirname2=os.listdir(file2)
factory_5=test(file2,dirname2)
file3='MRCG_testdata/0dB/factory_test/'
dirname3=os.listdir(file3)
factory0=test(file3,dirname3)

#car
file1='MRCG_testdata/5dB/car_test/'
dirname1=os.listdir(file1)
car5=test(file1,dirname1)
file2='MRCG_testdata/_5dB/car_test/'
dirname2=os.listdir(file2)
car_5=test(file2,dirname2)
file3='MRCG_testdata/0dB/car_test/'
dirname3=os.listdir(file3)
car0=test(file3,dirname3)

#babble
file1='MRCG_testdata/5dB/babble_test/'
dirname1=os.listdir(file1)
babble5=test(file1,dirname1)
file2='MRCG_testdata/_5dB/babble_test/'
dirname2=os.listdir(file2)
babble_5=test(file2,dirname2)
file3='MRCG_testdata/0dB/babble_test/'
dirname3=os.listdir(file3)
babble0=test(file3,dirname3)

sess.close()

